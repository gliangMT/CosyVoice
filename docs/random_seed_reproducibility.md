# CosyVoice 随机种子固定说明

本文简单解释 CosyVoice 当前代码里为什么要固定随机种子、固定了哪些随机源、训练和评测时应该怎么使用，主要基于当前 MUSA 训练和 seed-tts-eval 推理链路。

## 1. CosyVoice 中哪些地方会用到随机数

CosyVoice 里会用到随机数的地方主要有这些：

1. 模型随机初始化。
   从零训练 LLM、Flow 或 HifiGAN 时，模型参数一开始会随机初始化。

2. 数据顺序和数据增强。
   训练集会 shuffle，音频裁剪等数据处理也可能调用 `random.randint`。

3. DataLoader worker。
   多进程读取数据时，每个 worker 都有自己的随机状态。如果不固定，数据处理顺序可能每次不同。

4. Flow 推理噪声。
   扩散或 flow matching 推理通常会从随机噪声开始，例如 `torch.randn_like(...)`。

5. LLM token sampling。
   语音 token 生成可能使用采样策略，不固定 seed 时同一句文本可能生成不同 token。

6. 断点续训。
   如果中途保存 checkpoint 后继续训练，不仅要恢复模型参数，还要恢复随机数状态，否则恢复后的下一步训练可能已经走到另一条随机路径。

所以，只固定一个 `random.seed(...)` 是不够的。

## 2. 当前固定了哪些随机源

当前代码把训练和推理中最关键的随机源都纳入了控制。

| 随机源 | 作用 | 固定方式 |
| --- | --- | --- |
| Python `random` | 数据 shuffle、随机裁剪、训练中的普通 Python 随机逻辑 | `random.seed(seed)` |
| NumPy | 部分数据处理或第三方逻辑 | `numpy.random.seed(seed)` |
| PyTorch CPU RNG | CPU tensor 随机初始化或随机采样 | `torch.manual_seed(seed)` |
| MUSA 设备 RNG | MUSA 上的 tensor 随机初始化、Flow noise 等 | `torch.musa.manual_seed_all(seed)` |
| DataLoader generator | 控制 DataLoader 的迭代随机性 | `torch.Generator().manual_seed(...)` |
| DataLoader worker | 控制每个 worker 内部的 Python/NumPy 随机性 | `worker_init_fn=seed_worker` |
| checkpoint RNG state | 支持更精确的 resume | 保存和恢复 Python、NumPy、Torch、MUSA RNG 状态 |
| seed-tts 推理样本 | 控制每条评测样本的推理随机性 | `base_seed + meta_index` |



## 3. 训练入口如何固定 seed

训练入口在 `cosyvoice/bin/train.py`。

关键参数有三个：

```bash
--seed 1986
--data_seed 1986
--deterministic
```

其中：

- `--seed` 控制模型初始化和训练中的全局随机源。
- `--data_seed` 控制训练数据每个 epoch 的数据顺序。
- `--deterministic` 控制 PyTorch 尽量使用确定性算法；如果遇到某些没有确定性实现的算子，可能会报错。可以用 `--no-deterministic` 关闭。

训练启动后，`train.py` 会先调用：

```python
set_global_random_seed(args.seed, args.deterministic)
```

这个函数位于 `cosyvoice/utils/train_utils.py`，当前逻辑是：

```python
def set_global_random_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if hasattr(torch, 'musa') and torch.musa.is_available():
        torch.musa.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
```

注意调用时机很重要：它发生在加载 yaml、创建模型之前。这样从零训练时，模型随机初始化也会受 `--seed` 控制。

## 4. yaml 配置中的 seed

CosyVoice3 的训练配置在：

```text
examples/libritts/cosyvoice3/conf/cosyvoice3.yaml
```

文件顶部定义了：

```yaml
seed: 1986
```

并用这个 seed 初始化 Python、NumPy 和 Torch 的随机源。训练入口会把命令行里的 `--seed` 写入 yaml overrides：

```python
override_dict['seed'] = args.seed
```

因此，如果你执行：

```bash
--seed 2026
```

yaml 里的 `seed: 1986` 会被覆盖成 `2026`。这样命令行、训练入口和 yaml 配置里的 seed 是统一的。

## 5. `seed` 和 `data_seed` 有什么区别

这两个参数易混淆。

`seed` 更像“模型和计算随机性”的总开关，影响：

- 模型初始化。
- Torch 随机张量。
- 训练中模型内部随机逻辑。
- MUSA 上的随机计算。

`data_seed` 更像“数据读取顺序”的开关，影响：

- 每个 epoch 的数据 shuffle。
- 多卡训练时每个 rank 的数据流。
- DataLoader worker 中的数据处理随机性。

训练中每个 epoch 会派生一个 epoch seed：

```python
epoch_seed = data_seed + epoch * 100000 + rank
```

含义是：

- 不同 epoch 使用不同数据顺序。
- 不同 rank 使用不同但可预测的数据随机状态。
- 只要 `data_seed`、epoch、rank 不变，数据顺序就能复现。

验证集 CV 也会单独派生 seed，避免和训练数据流混在一起：

```python
data_seed + 10000000
```

## 6. DataLoader worker 为什么也要固定

当 `num_workers > 0` 时，PyTorch 会启动多个子进程读取和处理数据。每个 worker 都可能调用 Python 或 NumPy 随机函数。

当前代码给 DataLoader 设置了：

```python
worker_init_fn=seed_worker
```

`seed_worker` 会根据 PyTorch 给 worker 分配的初始 seed，再设置 Python 和 NumPy：

```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
```

这样每个 worker 的随机行为可控，而且不同 worker 不会完全使用同一条随机序列。

## 7. 断点续训时为什么要保存 RNG state

假设训练在第 3 个 epoch 的第 8000 step 中断。如果只恢复模型参数，那么下一次训练虽然从同一个模型开始，但随机数生成器的位置已经不一样。

这会导致：

- 下一次随机裁剪位置可能不同。
- 下一次 Flow noise 可能不同。
- 下一次采样或 dropout 类随机逻辑可能不同。

所以完整 resume 至少要恢复：

1. 模型参数。
2. optimizer。
3. scheduler。
4. AMP scaler。
5. epoch、step、batch index。
6. Python RNG state。
7. NumPy RNG state。
8. Torch RNG state。
9. MUSA RNG state。

当前 `capture_rng_state()` 会保存这些随机状态，`restore_rng_state()` 会在恢复训练时把它们恢复回来。

这也是为什么完整 resume 要使用：

```bash
--deepspeed.save_states model+optimizer
```

如果只保存 `model_only`，就不能完整恢复 optimizer 和随机状态。

## 8. 推理评测如何固定 seed

seed-tts-eval 的推理脚本在：

```text
seed-tts-eval/infer_cosyvoice_seedtts.py
```

它支持：

```bash
--seed 1986
```

也支持环境变量：

```bash
COSYVOICE_SEED=1986
```

推理时不是只在进程启动时设一次 seed，而是每条样本都会重新设一次：

```python
sample_seed = base_seed + meta_index
```

例如：

- 第 0 条样本使用 `1986 + 0`。
- 第 1 条样本使用 `1986 + 1`。
- 第 100 条样本使用 `1986 + 100`。

这样做有一个重要好处：即使你改变分片数、跳过已有文件，或者某个 shard 先跑完，同一条 meta 样本仍然会使用同一个 seed。

包装脚本 `seed-tts-eval/infer_cosyvoice.sh` 会把 `COSYVOICE_SEED` 传给每个 shard：

```bash
COSYVOICE_SEED=1986 bash seed-tts-eval/infer_cosyvoice.sh
```

## 9. 常用命令示例

从零训练时固定模型和数据 seed：

```bash
cd examples/libritts/cosyvoice3

seed=1986 data_seed=1986 \
bash run_from_scratch_with_log.sh
```

换一组随机初始化和数据顺序：

```bash
seed=2026 data_seed=2026 \
bash run_from_scratch_with_log.sh
```

只改变模型初始化，保持数据顺序：

```bash
seed=2026 data_seed=1986 \
bash run_from_scratch_with_log.sh
```

只改变数据顺序，保持模型初始化：

```bash
seed=1986 data_seed=2026 \
bash run_from_scratch_with_log.sh
```

seed-tts-eval 推理固定 seed：

```bash
COSYVOICE_SEED=1986 \
bash /home/cosyvoice-test/seed-tts-eval/infer_cosyvoice.sh
```

直接调用单进程推理脚本：

```bash
python3 /home/cosyvoice-test/seed-tts-eval/infer_cosyvoice_seedtts.py \
  /home/cosyvoice-test/data/seedtts_testset/en/meta.lst \
  /home/cosyvoice-test/outputs/seedtts_eval/en \
  --seed 1986
```

## 10. 为什么 seed 一样，结果仍可能不完全一致

固定 seed 能消除很多随机性，但它不是魔法。下面这些因素仍可能影响结果：

1. 代码变了。
   随机函数调用顺序只要变了，后续随机序列就会变。

2. 模型、数据或配置变了。
   同一个 seed 在不同模型权重、不同训练数据、不同 yaml 配置下没有可比性。

3. 分布式 world size 变了。
   例如原来 8 卡训练，中途改成 4 卡 resume，rank 和数据划分都会改变。严格复现应保持相同 world size。

4. 某些底层算子不是完全确定性的。
   `torch.use_deterministic_algorithms(True)` 可以帮助发现一部分问题，但不能保证所有硬件和后端都逐 bit 一致。

5. 第三方库内部有自己的随机逻辑。
   如果第三方库没有使用 Python、NumPy 或 Torch 的全局 RNG，可能还需要单独设置。

6. 浮点计算顺序不同。
   多卡通信、并行归约和不同后端可能导致极小的浮点差异。这种差异经过长时间训练后可能被放大。

因此更准确的说法是：固定 seed 可以让“同一代码、同一环境、同一配置、同一数据、同一并行方式”下的结果尽量可复现。

## 11. 排查随机性问题的清单

如果你怀疑结果仍然不稳定，可以按这个顺序检查：

1. 训练命令里是否显式设置了 `seed` 和 `data_seed`。
2. `run_from_scratch_with_log.sh` 日志里是否打印了相同的 seed。
3. 是否更换过模型目录、训练数据、yaml 配置或 checkpoint。
4. resume 时 world size 是否和 checkpoint 保存时一致。
5. 是否使用了 `--deepspeed.save_states model+optimizer`。
6. 是否从旧 checkpoint resume。旧 checkpoint 可能缺少 Python/NumPy RNG state。
7. 推理时是否设置了 `COSYVOICE_SEED` 或 `--seed`。
8. 推理是否开启了 `--overwrite`。如果没有 overwrite，旧 wav 可能不是当前 seed 生成的。
9. 分片推理时，同一份 meta 文件的行顺序是否发生变化。
10. 是否有第三方组件在内部使用了未受控的随机源。

## 12. 代码位置速查

| 文件 | 作用 |
| --- | --- |
| `cosyvoice/bin/train.py` | 解析 `--seed`、`--data_seed`、`--deterministic`，并在模型创建前固定全局 seed |
| `cosyvoice/utils/train_utils.py` | 实现全局 seed、DataLoader worker seed、epoch seed、RNG state 保存和恢复 |
| `examples/libritts/cosyvoice3/conf/cosyvoice3.yaml` | 定义可被命令行覆盖的 yaml seed |
| `examples/libritts/cosyvoice3/run_from_scratch_with_log.sh` | scratch 训练脚本，透传 `seed` 和 `data_seed` |
| `seed-tts-eval/infer_cosyvoice_seedtts.py` | seed-tts 单进程或单 shard 推理，按样本固定 seed |
| `seed-tts-eval/infer_cosyvoice.sh` | 8 shard 推理包装脚本，读取 `COSYVOICE_SEED` |
| `jd-seed-tts-eval/infer_seed_tts_cosyvoice.py` | JD seed-tts 推理脚本，按样本固定 seed |

## 13. 最短结论

训练时：

```bash
seed=1986 data_seed=1986 bash run_from_scratch_with_log.sh
```

推理时：

```bash
COSYVOICE_SEED=1986 bash infer_cosyvoice.sh
```

想要更严格复现，还要保持代码、模型、数据、配置、checkpoint、world size 和运行环境一致。
