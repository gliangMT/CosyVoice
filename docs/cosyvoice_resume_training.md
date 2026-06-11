# CosyVoice 断点续训（Resume）实现与使用说明

本文说明本仓库为 CosyVoice 训练增加的断点续训能力，包括原实现的限制、修改后的 checkpoint 结构、DDP 和 DeepSpeed 恢复流程、使用方法、日志判断以及常见问题。

## 1. 背景

训练机器或容器可能因为设备故障、通信异常、资源抢占等原因中断。重新启动训练时，仅恢复模型参数并不足以还原训练现场。

一次可继续的训练状态至少包括：

1. 模型参数。
2. Optimizer 状态，例如 Adam/AdamW 的一阶和二阶动量。
3. LR scheduler 状态。
4. AMP GradScaler 状态。
5. 当前 epoch 和全局 optimizer step。
6. 当前 epoch 已处理到的 batch 位置。
7. 各个 rank 的随机数状态。
8. 数据划分所依赖的 world size 和随机种子。

本次修改同时支持：

- 从完整 epoch checkpoint 继续下一 epoch。
- 从 epoch 中途 checkpoint 继续当前 epoch。
- 自动寻找最新 DDP 或 DeepSpeed checkpoint。
- 兼容原仓库生成的旧 checkpoint。

## 2. 原实现为什么不是真正的 Resume

原来的 `--checkpoint` 会执行类似以下逻辑：

```python
state_dict = torch.load(args.checkpoint, map_location='cpu')
model.load_state_dict(state_dict, strict=False)
start_step = state_dict.get('step', 0)
start_epoch = state_dict.get('epoch', -1)
```

DDP checkpoint 只保存：

```python
{
    **model.module.state_dict(),
    'epoch': epoch,
    'step': step,
}
```

因此原来的 `--checkpoint` 本质上是模型权重加载或 warm start，而不是完整断点续训。

### 2.1 Optimizer 动量丢失

Adam/AdamW 为每个参数维护 `exp_avg`、`exp_avg_sq` 和内部 step。重新创建 optimizer 会清空这些状态。模型虽然从旧权重开始，但后续参数更新轨迹已经发生变化。

### 2.2 Scheduler 没有完整恢复

原实现调用：

```python
scheduler.set_step(start_step)
```

CosyVoice scheduler 的 `set_step()` 只修改 `last_epoch`，不会立即把对应学习率写回 `optimizer.param_groups`。旧 checkpoint 恢复后的第一步可能使用错误的学习率。

### 2.3 AMP GradScaler 丢失

使用 `--use_amp` 时，GradScaler 会维护动态 loss scale 和 overflow 历史。原实现每次启动都会重新创建 scaler。

### 2.4 Epoch 中途 checkpoint 会跳过剩余数据

原主循环从以下位置开始：

```python
for epoch in range(start_epoch + 1, max_epoch):
```

如果 checkpoint 是 `epoch_5_step_12000.pt`，恢复后会直接从 epoch 6 开始，epoch 5 在 checkpoint 之后的数据不会再训练。

### 2.5 DeepSpeed 只有保存，没有恢复

原代码会调用 `model.save_checkpoint()`，但训练入口没有对应的 `model.load_checkpoint()`。即使磁盘上存在 DeepSpeed optimizer 状态，重启后也不会加载。

## 3. 参数语义

### 3.1 `--checkpoint`

只用于加载模型权重，适合：

- 加载官方 `llm.pt`、`flow.pt` 或 `hift.pt`。
- 从预训练模型开始微调。
- 不要求恢复 optimizer 训练轨迹的 warm start。

### 3.2 `--resume`

用于恢复训练状态：

```bash
--resume auto
```

或者指定 checkpoint：

```bash
--resume /path/to/epoch_1_step_8000.pt
```

也可以传入 checkpoint 目录：

```bash
--resume /path/to/model_dir
```

当同时提供 `--checkpoint` 和 `--resume` 时，`--resume` 优先。

`--resume auto` 的行为是：

- 找到 checkpoint：自动恢复。
- 第一次运行、目录中没有 checkpoint：正常开始新训练。

显式指定的 checkpoint 不存在时会报错，避免路径拼错后意外从头训练。

### 3.3 `--save_per_step`

覆盖 YAML 中的 `train_conf.save_per_step`：

```bash
--save_per_step 1000
```

这里的 step 是 optimizer 更新次数，不是 DataLoader micro-batch 数。

当 `accum_grad=2` 时：

```text
2 个 micro-batch = 1 个 optimizer step
```

设置为 `0` 或负数会关闭 epoch 中途 checkpoint，但 epoch 结束时仍会保存 `epoch_*_whole`。

### 3.4 `--deepspeed.save_states`

虽然参数名称来自 DeepSpeed，本次修改也用它控制 Torch DDP 的保存内容：

```bash
--deepspeed.save_states model+optimizer
```

要实现完整 DDP resume，必须使用 `model+optimizer`。使用 `model_only` 时只保存模型权重，无法恢复 Adam 动量和 AMP scaler。

### 3.5 `--data_seed`

默认值为 `1986`。它用于为每个 epoch 和 rank 构造可复现的数据管线种子：

```text
epoch_seed = data_seed + epoch * 100000 + rank
```

断点前后应保持相同的 `data_seed`。

## 4. DDP Checkpoint 文件结构

在 step checkpoint 保存后，目录中会出现：

```text
epoch_1_step_8000.pt
epoch_1_step_8000.train.pt
epoch_1_step_8000.rank0.rng.pt
epoch_1_step_8000.rank1.rng.pt
...
epoch_1_step_8000.yaml
latest_ddp
```

### 4.1 模型文件 `.pt`

保存模型参数、epoch 和 step：

```python
{
    **model.state_dict(),
    'epoch': 1,
    'step': 8000,
}
```

该格式保持兼容，可继续用于推理、模型平均或普通 warm start。

### 4.2 训练状态文件 `.train.pt`

保存：

```python
{
    'format_version': 1,
    'model_checkpoint': 'epoch_1_step_8000.pt',
    'epoch': 1,
    'step': 8000,
    'train_batch_idx': 1725,
    'epoch_complete': False,
    'world_size': 4,
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'scaler': scaler.state_dict(),
}
```

GAN 训练还会保存 discriminator optimizer 和 scheduler。

### 4.3 Rank RNG 文件

每个 rank 单独保存 CPU Torch RNG 和 MUSA/CUDA RNG：

```text
epoch_1_step_8000.rankN.rng.pt
```

### 4.4 `latest_ddp`

这是自动恢复指针，内容例如：

```text
epoch_1_step_8000.pt
```

`--resume auto` 优先读取它。旧目录没有 `latest_ddp` 时，会扫描 `epoch_*.pt` 并选择修改时间最新的文件。

### 4.5 YAML 信息文件

YAML 保存 loss、学习率、epoch、step、batch 位置、保存时间等可读元数据，便于检查 checkpoint。

## 5. DDP 原子保存

模型、训练状态和指针先写入临时文件：

```text
epoch_1_step_8000.pt.tmp.PID
```

写完后再通过 `os.replace()` 原子替换正式路径。

这样机器在保存过程中异常退出时：

- 上一个 checkpoint 不会被覆盖损坏。
- `latest_ddp` 不会指向只写了一部分的模型文件。
- 临时文件不会被自动恢复逻辑当作 checkpoint。

如果发现遗留的 `*.tmp.*` 文件，说明上次保存可能在写入过程中中断。自动恢复仍会使用最后一个正式 checkpoint。

## 6. Resume 执行流程

### 6.1 完整 epoch checkpoint

对于：

```text
epoch_3_whole.pt
```

其中 `epoch_complete=True`，恢复后从 epoch 4 开始，不需要跳过 epoch 内 batch。

### 6.2 Epoch 中途 checkpoint

对于：

```text
epoch_1_step_8000.pt
```

恢复流程如下：

1. 加载模型参数。
2. 创建 DDP 模型。
3. 创建 optimizer、scheduler 和 AMP scaler。
4. 从 `.train.pt` 恢复其完整状态。
5. 恢复各 rank 的 RNG。
6. 保持当前 epoch 为 1。
7. 使用相同 epoch seed 重建 DataLoader。
8. 从 epoch 开头重新执行数据管线。
9. 跳过 checkpoint 之前已经完成的 batch。
10. 从 checkpoint 后的第一个 batch 继续前向和反向。

checkpoint 只在梯度累积边界保存：

```python
(batch_idx + 1) % accum_grad == 0
```

因此恢复点不存在“只积累了一半梯度”的情况。

## 7. 为什么 Resume 时会长时间显示 `Skipping`

典型日志：

```text
start step 8000 start epoch 1
Epoch 1 TRAIN info lr 8.001e-05
Skipping 1726 already completed training batches
```

这通常不是卡死。

为了重新得到 checkpoint 后完全相同的数据顺序，跳过阶段仍需要执行数据管线，包括：

- 打开并读取 parquet。
- 音频解码和重采样。
- mel 和 Whisper 特征计算。
- embedding 解析或在线提取。
- shuffle、sort、动态 batching 和 padding。

跳过阶段不会运行模型前向、反向或 optimizer step，所以常见现象是：

- 显存占用基本不变化。
- GPU/MUSA 计算利用率比训练阶段低。
- 原版本没有中间日志，看起来像卡死。

本次修改增加了每 100 个 batch 的进度和 ETA：

```text
Resume skip progress 100/1726 batches, elapsed 28.4s, ETA 461.4s
Resume skip progress 200/1726 batches, elapsed 56.1s, ETA 428.0s
```

注意：正在运行的 Python 进程不会热加载新代码，进度日志会在下次启动时生效。

本仓库一次实际恢复记录为：

```text
16:51:07 Skipping 1726 already completed training batches
16:59:20 TRAIN Epoch 1 Batch 1800 Step 8037
```

约 8 分钟后恢复正常训练。

## 8. Step、Batch 和梯度累积

训练日志现在输出：

```text
TRAIN Epoch 1 Batch 1800 Step 8037 loss ...
```

含义：

- `Epoch 1`：当前 epoch。
- `Batch 1800`：当前 rank 在这个 epoch 中处理的 micro-batch 编号。
- `Step 8037`：已经完成的全局 optimizer 更新次数。

对于中途恢复：

```text
checkpoint step = 8000
checkpoint 后已处理 micro-batch = 1800 - 1726 = 74
accum_grad = 2
```

因此：

```text
current step = 8000 + 74 / 2 = 8037
```

## 9. CV 与 Checkpoint 保存

CV 表示 Cross Validation，即验证集评估。

每次触发 step checkpoint 时，当前实现会：

1. 暂停训练。
2. 遍历 `--cv_data` 指定的完整验证集。
3. 统计验证 loss。
4. 保存模型和训练状态。
5. 切回 `model.train()` 继续训练。

典型日志：

```text
Epoch 1 Step 8000 on_batch_end False CV rank 0
CV Epoch 1 Batch 100 Step 8000 loss ...
Epoch 1 Step 8000 CV info ...
Checkpoint: save to checkpoint .../epoch_1_step_8000.pt
```

因此 `save_per_step` 设置过小会频繁运行完整 CV，并产生较大的 checkpoint I/O 开销。

当前 Flow checkpoint 的实际大小大约为：

```text
模型文件:       1.3 GB
训练状态文件:   2.5 GB
```

建议从以下间隔开始：

```bash
--save_per_step 1000
```

或者：

```bash
--save_per_step 2000
```

## 10. 使用方法

### 10.1 CosyVoice3 普通训练

脚本已经固定传入 resume 参数：

```bash
cd examples/libritts/cosyvoice3
bash run_with_log.sh
```

其中包含：

```bash
--resume auto \
--save_per_step 1000 \
--deepspeed.save_states model+optimizer
```

首次启动会加载 `--checkpoint` 指定的预训练模型；当 `model_dir` 中存在训练 checkpoint 时，`--resume auto` 优先恢复训练状态。

### 10.2 从零训练

```bash
cd examples/libritts/cosyvoice3
MODELS=flow bash run_from_scratch_with_log.sh
```

脚本不会加载预训练的 CosyVoice 模块权重，但允许恢复该 scratch 实验自己生成的 checkpoint。

指定 LLM：

```bash
MODELS=llm bash run_from_scratch_with_log.sh
```

### 10.3 使用 `run.sh`

`run.sh` 会把命令行参数通过 `"$@"` 传给 `train.py`：

```bash
bash run.sh \
  --resume auto \
  --save_per_step 1000
```

### 10.4 显式指定 checkpoint

```bash
bash run_from_scratch_with_log.sh \
  --resume exp/cosyvoice3_scratch/flow/torch_ddp/epoch_1_step_8000.pt
```

也可以传 `.train.pt`，程序会自动映射到对应的模型 `.pt`：

```bash
--resume exp/.../epoch_1_step_8000.train.pt
```

### 10.5 直接使用 `torchrun`

`--resume` 必须放在 `train.py` 后面，因为它是训练脚本参数，不是 `torchrun` 参数：

```bash
torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv_id=1986 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:1234 \
  ../../../cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config conf/cosyvoice3.yaml \
  --train_data data/train.data.list \
  --cv_data data/dev.data.list \
  --model flow \
  --model_dir "$(pwd)/exp/cosyvoice3_scratch/flow/torch_ddp" \
  --tensorboard_dir "$(pwd)/tensorboard/cosyvoice3_scratch/flow/torch_ddp" \
  --ddp.dist_backend mccl \
  --resume auto \
  --save_per_step 1000 \
  --deepspeed.save_states model+optimizer
```

## 11. 如何判断恢复类型

### 11.1 完整 Resume

日志包含：

```text
Resuming DDP training from .../epoch_1_step_8000.pt
start step 8000 start epoch 1
Skipping 1726 already completed training batches
```

并且没有：

```text
Training state ... is missing
```

同时目录中存在：

```text
epoch_1_step_8000.train.pt
epoch_1_step_8000.rankN.rng.pt
```

这表示模型、optimizer、scheduler、AMP scaler 和 batch 位置均已恢复。

### 11.2 旧 Checkpoint 兼容恢复

日志包含：

```text
Training state ...train.pt is missing; resuming weights, epoch, and step only.
Optimizer momentum and AMP scaler will be reset.
```

此时恢复了：

- 模型参数。
- epoch 和 step。
- 根据 step 重建的 scheduler 学习率。

但没有恢复：

- Adam/AdamW 动量。
- AMP GradScaler。
- RNG 状态。

这可以继续训练，但不是严格连续的完整 resume。

### 11.3 仅权重 Warm Start

只使用：

```bash
--checkpoint pretrained_models/.../flow.pt
```

表示从预训练权重开始一个新训练，不属于 resume。

## 12. 兼容性与限制

### 12.1 旧完整 epoch checkpoint

旧的 `epoch_*_whole.pt` 没有 `.train.pt`，仍可兼容恢复到下一 epoch，但 optimizer 和 AMP 状态会重置。

### 12.2 旧 epoch 中途 checkpoint

旧的 `epoch_*_step_*.pt` 没有 batch 位置，无法安全判断需要跳过多少数据。程序会拒绝恢复并提示改用 `epoch_*_whole.pt`。

### 12.3 World size

epoch 中途恢复要求 world size 与保存 checkpoint 时相同：

```text
original world_size = 4
resume world_size = 4
```

改变卡数会改变 shard 和 batch 划分，因此程序会报错。

从 `epoch_*_whole.pt` 开始下一 epoch 时，可以重新评估是否允许改变 world size，但这不保证训练轨迹完全一致。

### 12.4 数据和配置

恢复前后应保持：

- 相同的 `train_data` 内容及顺序。
- 相同的数据 pipeline。
- 相同的 `accum_grad`。
- 相同的 `data_seed`。
- 相同的模型结构。
- epoch 中途恢复时相同的 world size 和 worker 配置。

修改数据集、过滤规则、动态 batch 配置或 parquet 内容后，旧的 `train_batch_idx` 可能不再对应同一批数据。

### 12.5 DeepSpeed

DeepSpeed 恢复调用：

```python
model.load_checkpoint(
    load_dir,
    tag=tag,
    load_optimizer_states=True,
    load_lr_scheduler_states=True,
)
```

`--resume auto` 使用 DeepSpeed 的 `latest` 文件。DeepSpeed 路径已实现，但当前主要验证环境是 Torch DDP + MUSA/MCCL；切换到新的 ZeRO 配置或不同 DeepSpeed 版本时应先做短训练保存/恢复测试。

## 13. 常见问题排查

### 13.1 日志停在 `Skipping`

先等待数据 replay 完成。新版本会每 100 batch 输出进度。

检查日志是否继续增长：

```bash
stat logs/.../flow_*.log
tail -f logs/.../flow_*.log
```

如果持续超过正常 replay 时间，再检查：

- parquet 是否可读。
- DataLoader worker 是否存活。
- 存储 I/O 是否异常。
- 所有 rank 是否都打印了相同的 skip 数量。

### 13.2 显存没有变化

skip 阶段不执行模型前反向，显存稳定是正常现象。恢复到第一个新 batch 后，日志会重新出现：

```text
TRAIN Epoch ... Batch ... Step ...
```

### 13.3 没有生成 `.train.pt`

确认命令包含：

```bash
--deepspeed.save_states model+optimizer
```

并且已经真正触发 checkpoint。`save_per_step=1000` 使用绝对 step，例如从 step 7137 启动时，下一个保存点是 step 8000。

### 13.4 触发 checkpoint 后长时间没有 TRAIN 日志

可能正在执行完整 CV 和写入数 GB 的 checkpoint。观察：

```text
CV Epoch ...
Checkpoint: save to checkpoint ...
```

### 13.5 如何确认当前 step

新日志直接打印：

```text
TRAIN Epoch 1 Batch 1800 Step 8037
```

旧日志可按以下公式估算：

```text
current_step =
    resume_start_step
    + (current_batch - resumed_batch_count) / accum_grad
```

对于从完整 epoch 开始的训练，`resumed_batch_count=0`。

## 14. 代码修改位置

主要改动如下：

| 文件 | 作用 |
| --- | --- |
| `cosyvoice/bin/train.py` | 增加 resume 参数解析、自动 checkpoint 发现、DDP/DeepSpeed 状态加载、恢复 epoch 和 batch 决策 |
| `cosyvoice/utils/train_utils.py` | 完整训练状态保存/加载、原子写入、RNG 保存、scheduler 兼容恢复、数据 seed |
| `cosyvoice/utils/executor.py` | epoch 内 batch replay、skip 进度、step checkpoint、CV 后保存完整状态 |
| `examples/libritts/cosyvoice3/run.sh` | 支持将额外 resume 参数传给 `train.py` |
| `examples/libritts/cosyvoice3/run_with_log.sh` | 默认启用自动 resume 和 step checkpoint |
| `examples/libritts/cosyvoice3/run_from_scratch_with_log.sh` | scratch 训练自动 resume，并拒绝预训练 checkpoint warm start |

## 15. 推荐配置

对于当前 CosyVoice3 Flow 四卡训练：

```bash
--resume auto \
--save_per_step 1000 \
--deepspeed.save_states model+optimizer \
--data_seed 1986
```

如果验证集较大或磁盘写入较慢，可将保存间隔提高到：

```bash
--save_per_step 2000
```

关键原则是：

1. 始终使用固定 `model_dir`。
2. 完整 resume 必须保存 optimizer 状态。
3. epoch 中途恢复不要改变卡数、数据或 batch 配置。
4. 将 `Skipping` 视为数据 replay 阶段，结合进度日志判断是否正常。
5. 定期保留最近几个完整 checkpoint，并清理过旧文件控制磁盘占用。

## 16. 验证情况

本次实现已经完成以下验证：

1. `train.py`、`train_utils.py` 和 `executor.py` Python 语法检查。
2. CosyVoice3 训练 Shell 脚本语法检查。
3. 小模型 checkpoint round-trip：
   - 模型权重一致。
   - AdamW optimizer state 一致。
   - Scheduler step 和学习率一致。
   - epoch、step 和 `train_batch_idx` 一致。
   - rank RNG 文件可加载。
4. 旧 `epoch_*_whole.pt` 自动发现和兼容恢复。
5. 四 rank MUSA/MCCL Flow 实际恢复：

```text
Resuming DDP training from .../epoch_1_step_8000.pt
start step 8000 start epoch 1
Skipping 1726 already completed training batches
TRAIN Epoch 1 Batch 1800 Step 8037
```

该记录确认以下状态已经在实际训练中工作：

- 模型参数恢复。
- AdamW optimizer 状态恢复。
- Scheduler 学习率恢复为 `8.001e-05`。
- AMP scaler 状态加载。
- epoch 中途 batch replay。
- replay 完成后继续前向、反向和 optimizer 更新。

当前实现没有把 DataLoader iterator 本身序列化进 checkpoint，而是采用“确定性重建 + replay”的方式。这是恢复时出现 skip 等待的根本原因，也是后续若要优化恢复速度时最值得改进的部分。
