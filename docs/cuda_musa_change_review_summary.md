# CUDA/MUSA 精度对齐与评测链路改动 Review 说明

本文用于代码review, 主要围绕`CosyVoice`的MUSA适配、CUDA/MUSA from-scratch 对齐、`Seed-TTS-Eval`评测脚本所做的改动讲清楚。

## 0. 基线与范围说明

本次总结涉及两个独立 git 仓库，基线分别如下:

CosyVoice 主仓：

```text
仓库路径：/home/cosyvoice-test/CosyVoice
基线 commit：074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
```

seed-tts-eval 仓库：

```text
仓库路径：/home/cosyvoice-test/seed-tts-eval
基线 commit：752f4297f090c46bb1a55a1f7439e5944ddefe8d
```

因此本文分两部分总结：

- CosyVoice 主仓：按当前 `precision-alignment` 分支相对 `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` 之后的改动总结。
- seed-tts-eval：按 `/home/cosyvoice-test/seed-tts-eval` 仓库中 `752f4297f090c46bb1a55a1f7439e5944ddefe8d` 之后的改动总结。

`CosyVoice`仓库中`docs`目录下还有其他文档可参考查阅

## 1. 总体目标

这批改动主要解决四类问题。

1. CosyVoice3 可以在 CUDA 和 MUSA 两类后端上使用尽量一致的训练、推理和评测入口。

2. 训练过程具备更强的可复现性和可恢复性，包括固定随机源、可恢复 dataloader 进度、保存 optimizer/scheduler/scaler/RNG 状态，以及支持跨平台 checkpoint 迁移。

3. 为 CUDA/MUSA from-scratch 对齐建立更可解释的实验基线。这里的目标不是承诺跨硬件 bitwise identical，而是尽量让两边输入构造、随机数消费、注意力后端、AMP 路径、optimizer step 边界等变量受控。

4. 把 Seed-TTS-Eval 适配MUSA平台以及新增测试脚本，将其从一个单纯指标计算仓库扩展为 CosyVoice checkpoint 的评测工具，支持 CUDA/MUSA 多卡推理、WER、SIM计算等功能。

## 2. CosyVoice 主仓改动

### 2.1 MUSA/CUDA 后端抽象

相关文件：

```text
cosyvoice/bin/train.py
cosyvoice/utils/device.py
cosyvoice/utils/onnx.py
cosyvoice/cli/cosyvoice.py
cosyvoice/cli/frontend.py
cosyvoice/cli/model.py
tools/extract_speech_token.py
requirements.txt
```

主要改动：

- 新增 `cosyvoice/utils/device.py`，集中判断 `torch.musa`、`torch.cuda`、CPU，并封装：
  - `get_available_device()`
  - `is_gpu_available()`
  - `stream_context(device)`
  - `empty_cache_and_sync(device)`
- `train.py` 在导入 `torch` 前尝试导入 `torchada`。为了同时适配CUDA和MUSA设备引入torchada库（相关说明详见：https://github.com/MooreThreads/torchada ），当前训练路径减少适配代码改动仍保留绝大部分使用 `.cuda()`、`torch.cuda.*` 等 CUDA API，MUSA 侧依赖 `torchada` 做兼容 patch。
- CLI 推理路径不再直接用 `torch.cuda.is_available()` 判断设备，而是使用统一的 device helper。
- ONNX speech tokenizer provider 从固定 `CUDAExecutionProvider` 改成按 `COSYVOICE_ONNX_PROVIDER` 选择：
  - `cpu`
  - `cuda`
  - `musa`
  - `auto`
- `tools/extract_speech_token.py` 也复用同一套 ONNX provider 选择逻辑。
- `requirements.txt` 不再强行安装 CUDA PyTorch、TensorRT 和 onnxruntime-gpu，改为说明这些平台相关包应由 CUDA/MUSA 基础镜像或本地安装脚本提供。

原理与合理性：

原始 CosyVoice 代码默认 CUDA。若在 MUSA 上直接运行，最容易出问题的是三类地方：PyTorch 设备判断、`torch.cuda` API、ONNX Runtime provider。把这些能力集中到 `device.py` 和 `onnx.py` 后，业务代码不需要到处写分支，且 CUDA 路径仍保持原有语义。

需要 review 的点：

- `example.py` 中无条件 `import torchada`，如果某些 CUDA-only 环境没有安装 torchada，示例脚本会失败。更稳妥的方式是像 `train.py` 一样按环境判断后再导入。
- 当前训练主路径仍依赖 `torchada` 兼容 `.cuda()`，并没有完全改成 `.to(device)`。这是工程上快速适配 MUSA 的方案，不是彻底设备无关化。

### 2.2 训练随机性与数据顺序可复现

相关文件：

```text
cosyvoice/bin/train.py
cosyvoice/utils/train_utils.py
cosyvoice/utils/executor.py
```

主要改动：

- 新增训练参数：
  - `--seed`
  - `--data_seed`
  - `--deterministic/--no-deterministic`
- 新增 `set_global_random_seed()`，统一设置：
  - Python `random`
  - NumPy
  - `torch.manual_seed`
  - CUDA/MUSA accelerator RNG
  - cuDNN deterministic/benchmark
  - `torch.use_deterministic_algorithms`
- DataLoader 增加 `torch.Generator`，并通过 `seed_worker()` 初始化 worker 内的 Python/NumPy 随机源。
- 每个 epoch 调用 `seed_dataloader_for_epoch()`，用 `data_seed + epoch * 100000 + rank` 重新设定 dataloader generator。
- `batch_forward()` 将当前 epoch 写入 `batch['_epoch']`，供 LLM deterministic bi-stream 使用。

原理与合理性：

跨平台对齐时，第一层必须保证“同一个 rank、同一个 epoch、同一个 batch_idx 拿到的是同一批样本”，否则后面讨论 loss 差异没有意义。DataLoader 的 worker 随机源、shuffle、epoch seed 如果不固定，CUDA/MUSA 的输入样本顺序可能已经不同。

这套改动把数据顺序和模型随机数拆成两套 seed：`seed` 控制模型与训练随机性，`data_seed` 控制数据迭代。这样 resume 或调试时更容易定位是哪条随机链路出了问题。

### 2.3 DDP uneven dataloader 修复

相关文件：

```text
cosyvoice/utils/executor.py
cosyvoice/utils/train_utils.py
docs/gloo_uneven_dataloader_fix.md
```

主要改动：

- 移除旧的 `cosyvoice_join()`/`monitored_barrier` 式 uneven workload 处理。
- 新增 `distributed_batch_available(has_batch, control_group)`。
- 每个 rank 取 batch 后，用 control group 做 `all_reduce(MIN)`：
  - 所有 rank 都有 batch，继续训练。
  - 任意 rank 已经 StopIteration，所有 rank 统一停止当前 epoch。
- 如果 epoch 结束时处于未完成的 gradient accumulation window，主动 `optimizer.zero_grad()` 并丢弃半窗口梯度。

原理与合理性：

旧逻辑依赖 barrier 超时来“发现”某些 rank 已经没数据，这种方式容易污染进程组，表现为卡死、误判或后续 collective 失败。新逻辑将“是否还有 batch”变成显式的低成本 control collective，所有 rank 在同一语义点做一致决策。

这对动态 batch、不同 rank 数据长度不完全一致、MUSA/MCCL 调试都更稳。

### 2.4 Checkpoint resume 与容错训练

相关文件：

```text
cosyvoice/bin/train.py
cosyvoice/utils/train_utils.py
cosyvoice/utils/executor.py
docs/cosyvoice_resume_training.md
README.md
```

主要改动：

- 新增训练参数：
  - `--resume [PATH|auto]`
  - `--resume_mode exact|cross_platform|weights_only`
  - `--save_per_step`
  - `--early_stop_file`
  - `--deepspeed.save_states model+optimizer`
- DDP checkpoint 除原本模型权重外，新增旁路状态文件：
  - `epoch_xxx.pt`：模型权重，保留 `epoch`、`step` 元信息。
  - `epoch_xxx.train.pt`：optimizer、scheduler、scaler、resume signature、train batch index 等训练状态。
  - `epoch_xxx.rankN.rng.pt`：每个 rank 的 Python/NumPy/Torch/accelerator RNG 状态。
  - `latest_ddp`：最新 DDP checkpoint 指针。
- 保存改为 atomic 写入，避免中途失败产生半截 checkpoint。
- 支持 intra-epoch checkpoint，通过 `train_batch_idx` 和 `_skip_batches()` 恢复到 epoch 内对应 batch。
- 支持三种 resume 语义：
  - `exact`：要求配置、数据、world size、RNG 等严格匹配，目标是尽可能 bit-exact 续训。
  - `cross_platform`：允许 CUDA/MUSA 平台差异，迁移模型、optimizer、scheduler 等可迁移状态，但 accelerator RNG 重置。
  - `weights_only`：只恢复权重，optimizer/scheduler/RNG 重新开始。
- Resume signature 记录环境和训练配置：
  - 模型组件、train_engine、seed、data_seed、dtype、optim、scheduler、数据文件 sha256、world size、ONNX provider、CUDA/MUSA runtime 版本等。

原理与合理性：

训练恢复不是只加载模型权重。Adam 动量、scheduler step、AMP scaler、RNG 状态和 dataloader 位置都会影响后续 loss。对于精度对齐实验，缺任何一块都会让“续训后差异”不可解释。

`exact` 和 `cross_platform` 分开是合理的：CUDA 到 CUDA 同环境续训可以追求严格一致；CUDA 到 MUSA 则不应恢复 CUDA accelerator RNG，也不应要求 CUDA/MUSA runtime 字段相同。

需要 review 的点：

- `cross_platform` 目前要求使用 `epoch_*_whole.pt`，这是保守设计，因为跨平台恢复到 epoch 中间时，dataloader prefetch、worker 状态和 accelerator RNG 更难解释。
- 如果未来需要跨平台 intra-epoch resume，需要额外定义数据迭代状态和 RNG 迁移边界。

### 2.5 LLM CUDA/MUSA 精度对齐

相关文件：

```text
cosyvoice/llm/llm.py
examples/libritts/cosyvoice3/conf/cosyvoice3.yaml
cosyvoice/utils/train_utils.py
```

主要改动：

- `Qwen2Encoder` 增加 `attn_implementation` 参数，可显式指定 Qwen attention backend。
- CosyVoice3 配置中将 Qwen attention backend 设置为 `eager`。
- `Qwen2LM` 和 `CosyVoice3LM` 增加：
  - `bistream_mode`
  - `bistream_seed`
- 原始 bi-stream 选择逻辑：

```python
random.random() < 0.5
```

改为支持 deterministic 模式：

```python
digest = blake2b(f'{seed}:{epoch}:{utt}', digest_size=8)
int.from_bytes(digest, byteorder='little') < (1 << 63)
```

- deterministic 模式要求 batch 中提供 `utts`，否则抛错。
- `forward()` 和 `forward_dpo()` 将 `utts`、`epoch` 传入 `prepare_lm_input_target()`。
- CosyVoice3 yaml 中设置：

```yaml
bistream_mode: deterministic
bistream_seed: !ref <seed>
attn_implementation: eager
```

原理与合理性：

CosyVoice3 LLM 的核心随机差异不在 dropout，而在输入构造。bi-stream 和 uni-stream 不是微小数值差异，而是完全不同的 `lm_input/lm_target` 布局。如果 CUDA 端某个样本走 uni-stream，MUSA 端同一个样本走 bi-stream，那么后续 loss 已经无法比较。

用 `seed + epoch + utt` 做 hash 有两个好处：

- 与 Python RNG 消费顺序解耦。
- 对同一个样本、同一个 epoch，在任意 rank、任意后端上决策一致。

从概率分布看，这个改动保持了原逻辑的 0.5 选择概率。原始代码 `random.random() < 0.5` 是从 `[0, 1)` 上取均匀随机数，小于 0.5 时走 bi-stream，因此边际概率是 1/2。当前实现使用 `blake2b(seed, epoch, utt)` 得到 8 字节哈希值，可以视为在 `[0, 2^64 - 1]` 上近似均匀的确定性伪随机整数；判断它是否小于 `2^63`，正好把 64-bit 取值空间均分成两半，因此 bi-stream 的边际概率仍是 1/2。

需要注意的是，它和原始代码是“分布等价”，不是“逐样本轨迹等价”。同一批样本中 bi-stream 的整体比例期望仍约为 50%，但具体哪些 utterance 被选中不再由 Python RNG 流决定，而由 `seed/epoch/utt` 唯一决定。这个变化正是 CUDA/MUSA 对齐需要的：保留原训练中约一半样本使用 bi-stream 的数据增强语义，同时消除 Python RNG 消费顺序差异导致的跨平台输入布局分叉。

将 Qwen attention 设为 `eager` 是为了建立保守 FP32 对齐基线，避免 CUDA/MUSA 分别走不同 fused SDPA kernel 后把算子差异混入实验。

需要 review 的点：

- deterministic bi-stream 是对齐实验友好的改动，也可以作为长期可复现训练选项保留。
- `eager` attention 和 LLM FP32 baseline 更偏实验开关，会牺牲性能，不一定适合作为最终高性能训练配置。

### 2.6 Flow CUDA/MUSA 随机数对齐 guard

相关文件：

```text
cosyvoice/bin/train.py
cosyvoice/utils/train_utils.py
cosyvoice/utils/executor.py
examples/libritts/cosyvoice3/conf/cosyvoice3.yaml
examples/libritts/cosyvoice3/run_from_scratch_with_log_on_cuda.sh
examples/libritts/cosyvoice3/run_from_scratch_with_log_on_musa.sh
docs/cuda_musa_precision_alignment.md
```

主要改动：

- 新增参数 `--rng_alignment_max_speech_feat_numel`。
- 如果某个 batch 的 `speech_feat.numel()` 超过阈值，则认为它可能触发 CUDA/MUSA 大随机 kernel 的 Philox counter 消费差异。
- 分布式下用 `all_reduce(MAX)` 判断任意 rank 是否超限；只要一个 rank 超限，所有 rank 同步跳过。
- 训练阶段不是只跳过单个 micro-batch，而是丢弃完整 gradient accumulation window。
- CV 阶段超限 batch 直接跳过；如果全部 CV batch 都被跳过则报错。
- CosyVoice3 Flow DiT 配置中将 `dropout` 设置为 `0.0`。
- from-scratch 脚本中 Flow 默认启用阈值 `184320`。

原理与合理性：

Flow 的 `compute_loss()` 中存在 `torch.rand` timestep、`torch.randn_like` noise、CFG mask 等随机项；DiT dropout 也可能产生大 mask。跨硬件后端中，大 tensor 随机 kernel 的 launch grid 可能不同，导致 Philox counter 消费轨迹不同。一次大随机 tensor 分叉后，后续所有随机数都可能偏离。

阈值 guard 的目的不是证明随机数完全一致，而是避免最容易触发随机轨迹分叉的大 batch。丢弃完整 accumulation window 是因为只丢单个 micro-batch 会导致 optimizer.step 的梯度内容不同，Adam 动量和 scheduler 后续轨迹仍然分叉。

需要 review 的点：

- 这是实验 guard，不是通用训练策略。它会改变有效训练数据分布。
- Flow 里仍有 Python `random.random/random.randint` 控制 streaming 和 condition，还有 diffusion timestep/noise/CFG 随机。当前 guard 主要覆盖大 tensor RNG 风险，并不等于 Flow 全链路 bitwise 对齐。
- 阈值 `184320` 是经验阈值，应在文档中明确其硬件/版本上下文。

### 2.7 CosyVoice3 from-scratch 脚本与日志

相关文件：

```text
examples/libritts/cosyvoice3/run.sh
examples/libritts/cosyvoice3/run_with_log.sh
examples/libritts/cosyvoice3/run_from_scratch_with_log_on_cuda.sh
examples/libritts/cosyvoice3/run_from_scratch_with_log_on_musa.sh
examples/libritts/cosyvoice3/conf/cosyvoice3.yaml
.gitignore
```

主要改动：

- 新增 CUDA/MUSA 两套 from-scratch 启动脚本。
- 脚本支持：
  - `MODELS=llm|flow|hifigan`，且一次只允许一个组件。
  - `shared_init_checkpoint`，让 CUDA/MUSA 从同一个初始化权重开始。
  - `--resume auto`
  - `--save_per_step 16000`
  - `--deepspeed.save_states model+optimizer`
  - Flow RNG alignment guard。
  - LLM 默认 FP32 alignment mode。
- MUSA 脚本设置 MUSA/MCCL 相关环境变量和 `ACCELERATOR_BACKEND=musa`。
- CUDA 脚本设置 `ACCELERATOR_BACKEND=cuda` 和 `COSYVOICE_ONNX_PROVIDER=cuda`。
- `.gitignore` 忽略 `examples/libritts/cosyvoice3/logs/*`。
- CosyVoice3 yaml 从原 SFT 配置调整为 from-scratch 默认配置，并保留 SFT/Flow from-scratch 配置注释。

原理与合理性：

将 CUDA/MUSA 启动脚本拆开，可以让平台相关环境变量显式化，避免在同一脚本里混用 `CUDA_VISIBLE_DEVICES` 和 `MUSA_VISIBLE_DEVICES`。同时通过 `shared_init_checkpoint` 避免“初始权重不同”污染对齐实验。

#### `shared_init_checkpoint` 的作用

`shared_init_checkpoint` 是 from-scratch 对齐实验里的“共同起跑线”开关。它的目的不是续训，而是让 CUDA 和 MUSA 两边在 step 0 拥有完全相同的一份模型参数。

默认情况下，如果不传 `shared_init_checkpoint`，脚本会使用：

```bash
checkpoint_args=(--resume auto)
```

这表示优先从当前 `model_dir` 下的已有 checkpoint 恢复；如果没有 checkpoint，则由 `train.py` 按当前 seed 和初始化逻辑创建新模型。即使 CUDA 和 MUSA 设置了同一个 seed，也仍然可能因为环境、导入顺序、模型构造细节或之前目录中是否已有 checkpoint 等因素，让两边的初始状态不再是同一个可审计对象。

设置 `shared_init_checkpoint` 后，脚本会改成：

```bash
checkpoint_args=(--checkpoint "${shared_init_checkpoint}")
```

也就是说，训练入口只加载这份 checkpoint 中的模型权重，然后从新的训练状态开始跑。它不会恢复这份 checkpoint 对应的 optimizer、scheduler、AMP scaler 或 RNG 状态；这些状态会按当前实验重新初始化。因此它和 `--resume` 的语义不同：

```text
--checkpoint shared_init_checkpoint
  只提供初始模型权重，用于开始一轮新实验。

--resume / --resume auto
  恢复已有训练过程，目标是接着之前的 step/epoch 继续训练。
```

这种设计对 CUDA/MUSA 对齐很重要。from-scratch 实验想比较的是“同一初始权重、同一数据顺序、同一训练配置下，后端数值路径会如何分叉”。如果两边各自随机初始化，就算 seed 相同，也很难在会议或 review 中证明初始权重确实一致；如果两边都从同一个 `shared_init_checkpoint` 加载，则初始权重可以通过同一个文件路径、文件 hash 或参数 hash 来审计。

推荐使用方式是先生成一份只用于对齐实验的 init checkpoint，例如在某一端保存 `init.pt`，然后 CUDA 和 MUSA 都显式指定同一个文件：

```bash
shared_init_checkpoint=/path/to/init.pt MODELS=llm bash run_from_scratch_with_log_on_cuda.sh
shared_init_checkpoint=/path/to/init.pt MODELS=llm bash run_from_scratch_with_log_on_musa.sh
```

需要注意，`shared_init_checkpoint` 只能保证模型参数起点一致。它不自动保证后续训练完全一致；后续仍依赖数据顺序、LLM bi-stream 决策、Flow RNG guard、AMP/FP32 设置、attention backend、optimizer step 边界等其它对齐措施。

需要 review 的点：

- 当前 CUDA from-scratch 脚本默认 `MODELS=flow`，MUSA 脚本默认 `MODELS=llm`。做对齐实验时应显式传入 `MODELS=llm` 或 `MODELS=flow`，否则容易比较了不同组件。
- `run.sh` 中包含本机硬编码路径和 MUSA 默认设备，更像个人实验脚本，不一定适合上游合入。

### 2.8 推理与模型加载兼容

相关文件：

```text
cosyvoice/cli/model.py
cosyvoice/hifigan/generator.py
example.py
model_load.py
runtime/triton_trtllm/token2wav_dit.py
```

主要改动：

- 模型加载时新增 `_load_weight_state_dict()`，过滤训练 checkpoint 中的 `epoch/step/state_dict` 等元信息，使训练保存的 checkpoint 可被推理模型加载。
- 推理 cache 清理和 stream 上下文改为调用 device helper。
- HiFT `f0_predictor` 从 float64 改为 float32，因为当前 MUSA 对 fp64 支持不足。
- `example.py` 指向本机 `Fun-CosyVoice3-0.5B-test` 软链接模型目录。
- 新增 `model_load.py`，用于从 modelscope 下载 Fun-CosyVoice3 模型。
- `runtime/triton_trtllm/token2wav_dit.py` 从符号链接变成实际文件。

原理与合理性：

训练 checkpoint 常带有 `epoch/step` 等元信息，直接 `strict=True` 加载到推理模型会失败；过滤非 Tensor 项后，训练产物可以直接参与 Seed-TTS 推理评测。

HiFT float32 是 MUSA 平台兼容性让步。它可能带来轻微数值变化，但比 fp64 不支持导致推理不可用更合理。

需要 review 的点：

- `example.py` 中硬编码本机路径不适合长期保留。
- `runtime/triton_trtllm/token2wav_dit.py` 由 symlink 变实体文件，需确认这是否是预期提交，避免引入上游重复代码维护成本。

## 3. seed-tts-eval 改动

seed-tts-eval 的改动严格基于：

```text
752f4297f090c46bb1a55a1f7439e5944ddefe8d
```

### 3.1 CosyVoice 推理入口

相关文件：

```text
infer_cosyvoice.sh
infer_cosyvoice_seedtts.py
```

主要改动：

- 新增 `infer_cosyvoice_seedtts.py`，读取 Seed-TTS meta 文件，调用 CosyVoice `inference_zero_shot()` 生成 wav。
- 支持：
  - `--model-dir`
  - `--cosyvoice-root`
  - `--prompt-prefix`
  - `--num-shards/--shard-index`
  - `--limit`
  - `--overwrite`
  - `--device-backend auto|musa|cuda`
  - 每条样本使用 `seed + meta_index` 固定推理随机性。
- 加载 CosyVoice 前可尝试导入 `torchada`。
- 推理加载 checkpoint 时过滤训练元信息 `epoch/step`。
- 新增 `infer_cosyvoice.sh`，通过 `MUSA_VISIBLE_DEVICES` 或 `CUDA_VISIBLE_DEVICES` 自动决定后端和 worker 数。

原理与合理性：

Seed-TTS-Eval 原本只算指标，不负责 CosyVoice 推理。新增推理入口后，可以把训练 checkpoint 直接转成 Seed-TTS wav 输出，形成闭环。

每张卡一个进程、每个进程设置单独 visible device，比在进程内手动管理多设备更简单，也更适配 CUDA/MUSA 两套后端。

### 3.2 WER 计算改造

相关文件：

```text
cal_wer.sh
run_wer.py
prepare_ckpt.py
cuda_cal_wer.sh
musa_cal_wer.sh
average_wer.py
```

主要改动：

- `cal_wer.sh` 支持 CUDA/MUSA 后端自动识别，按 visible device 列表切分任务。
- 支持 `limit`，便于 smoke test。
- `prepare_ckpt.py` 改为按语言加载 Whisper 或 Paraformer，并验证当前设备可用。
- `run_wer.py`：
  - 支持 `EVAL_DEVICE=cuda:0|musa:0`。
  - 优先使用本地 `data/models/whisper-large-v3`，否则回退 HuggingFace 模型名。
  - English WER 使用 Whisper-large-v3，并传 attention mask。
  - `jiwer` 从旧 `compute_measures` 切换到 `process_words`。
  - Chinese WER 使用 FunASR Paraformer，并传入 device。
- 新增 CUDA/MUSA wrapper，默认单机 8 卡。

原理与合理性：

WER 评测是后处理，但如果其设备管理仍写死 CUDA，就无法在 MUSA 上跑完整指标。将 WER 的设备、worker、模型路径外部化后，推理和评测可以在同一套 CUDA/MUSA 设备规则下执行。

### 3.3 SIM 计算改造

相关文件：

```text
cal_sim.sh
prepare_wavlm.py
cuda_cal_sim.sh
musa_cal_sim.sh
thirdparty/UniSpeech/downstreams/speaker_verification/models/ecapa_tdnn.py
thirdparty/UniSpeech/downstreams/speaker_verification/verification.py
thirdparty/UniSpeech/downstreams/speaker_verification/verification_pair_list_v2.py
```

主要改动：

- `cal_sim.sh` 支持 CUDA/MUSA 后端和多卡切分。
- 新增 `prepare_wavlm.py`：
  - 优先使用本地 s3prl repo。
  - 支持本地 WavLM checkpoint。
  - 支持 offline 模式，避免评测时临时访问 GitHub。
- UniSpeech speaker verification 改造：
  - `model.cuda(device)` 改为 `model.to(device)`。
  - wav tensor 也使用 `.to(device)`。
  - 支持 `musa:0`。
  - 对过短音频做显式报错。
  - 记录失败样本到 `.failures.tsv`。
  - 如果没有任何有效 SIM 分数则报错。
- 新增 CUDA/MUSA SIM wrapper。

原理与合理性：

SIM 依赖 WavLM/s3prl，原始脚本容易隐式下载或写死 CUDA。对离线服务器和 MUSA 评测来说，这会导致评测不稳定。新增本地 repo/checkpoint 优先级和失败样本记录后，评测结果更可复查。

### 3.4 Checkpoint 自动评测与早停

相关文件：

```text
eval_cosyvoice_checkpoint.sh
watch_cosyvoice_checkpoints.sh
```

主要改动：

- `eval_cosyvoice_checkpoint.sh` 评测单个 CosyVoice checkpoint：
  - 创建临时 model directory。
  - 固定 base model 资源。
  - 只替换当前训练组件的 `llm.pt` 或 `flow.pt`。
  - 多卡生成 wav。
  - 自动计算 English WER。
  - 输出 `.complete`、`wer.txt`、`wav_res_ref_text`。
- `watch_cosyvoice_checkpoints.sh` 持续监听训练目录中的 `epoch_*_whole.pt`：
  - 按间隔评测，例如每 N 个 completed epoch 评一次。
  - 维护 `wer_history.tsv`。
  - 若最近窗口 WER 波动小于阈值，则写入 `STOP_TRAIN` 文件。
- CosyVoice 训练侧的 `--early_stop_file` 会在 epoch 边界读取该文件并优雅停止。

原理与合理性：

这将训练和评测连接成闭环，但保持松耦合：训练进程不直接调用评测，只检查一个 stop file；评测进程不侵入训练，只读取已保存 checkpoint。这样失败隔离更好，也方便在另一张卡或另一台机器评测。

需要 review 的点：

- `eval_cosyvoice_checkpoint.sh` 的策略是“只替换一个组件，其余组件复用 base model”。这适合评测单组件 fine-tune 的相对变化；如果 LLM/Flow 是 from scratch 或两者耦合变化很大，WER 可能反映的是组件不匹配，而不只是平台差异。

### 3.5 文档与依赖

相关文件：

```text
README.md
docs/MUSA_COSYVOICE_EVAL_GUIDE.md
requirements.txt
.gitignore
```

主要改动：

- README 增加 CosyVoice + CUDA/MUSA 评测快速开始。
- 新增 MUSA 评测详细指南。
- requirements 增加 WER/SIM 所需依赖，注释掉 torch/torchaudio，避免覆盖已有 CUDA/MUSA PyTorch 环境。
- `.gitignore` 增加评测输出/缓存忽略项。

原理与合理性：

评测环境最怕隐式依赖和路径不清。文档把工作区目录、模型目录、meta 文件、wrapper 脚本和指标边界写清楚，可以显著降低复现实验的沟通成本。
