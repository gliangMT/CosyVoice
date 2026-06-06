# CosyVoice 多卡训练 Gloo 问题修复说明

本文介绍 CosyVoice 多卡训练在一个 epoch 结束时出现 Gloo 超时的问题，以及最终采用的修复方案。内容尽量从基础概念开始，适合第一次接触 PyTorch 分布式训练的读者。

## 1. 问题现象

LLM 单卡训练可以正常运行，但四卡或八卡训练经常在一个 epoch 接近结束时出现以下问题：

```text
Rank N failed to pass monitoredBarrier
DistStoreError: wait timeout
ChildFailedError
```

有时日志看起来已经跑完训练 batch，甚至完成了 CV 和 checkpoint 保存，随后却在进入下一个 epoch 时失败。

旧日志中的典型过程是：

```text
TRAIN 最后一批数据
monitoredBarrier 等待 60 秒
某个 rank 没有进入 barrier
所有 rank 退出当前训练循环
完成 CV 和 checkpoint
下一 epoch 创建 Gloo group 超时
```

这容易让人误以为是 MCCL、MUSA 设备或者模型计算出错。实际上，这一类错误首先来自不同 rank 的 DataLoader 长度不一致，以及不合理的退出协议。

## 2. 需要理解的几个概念

### 2.1 rank 是什么

多卡训练会启动多个 Python 进程。每个进程负责一张卡，并获得一个编号，这个编号叫作 rank。

八卡训练时通常是：

```text
rank 0 -> 第 0 张可见设备
rank 1 -> 第 1 张可见设备
...
rank 7 -> 第 7 张可见设备
```

每个 rank 都会独立读取数据、执行前向计算和反向传播。DDP 会在反向传播时同步所有 rank 的梯度。

### 2.2 为什么各 rank 的 batch 数可能不同

训练集首先以 parquet shard 为单位分配给不同 rank，但以下因素会继续改变每个 rank 的数据量：

1. shard 总数不一定能被卡数整除。
2. 不同 shard 中的有效样本数可能不同。
3. `filter` 会根据语音和文本长度丢弃样本。
4. CosyVoice 使用动态 batch，batch 大小取决于语音帧数。
5. shuffle 后的数据排列会影响动态 batch 的组合结果。

因此，即使每个 rank 分到的 shard 数接近，最终得到的 batch 数仍可能不同。

例如：

```text
rank 0: 6950 batches
rank 1: 6951 batches
rank 2: 6950 batches
rank 6: 6947 batches
```

这属于正常的数据管线现象，不应该通过触发通信异常来处理。

### 2.3 MCCL 和 Gloo 分别做什么

当前训练同时使用两个通信后端：

- MCCL：在 MUSA 设备之间同步模型梯度等大数据。
- Gloo：在 CPU 上同步少量控制信息。

修复后的 Gloo group 只传递一个很小的整数，用于表示当前 rank 是否还有 batch。

## 3. 旧实现为什么有风险

旧实现每处理一个 batch 都调用：

```python
dist.monitored_barrier(group=group_join, timeout=...)
```

可以把 barrier 理解为一次集体签到。只有所有 rank 都到达，程序才能继续。

当某个 rank 的 DataLoader 先耗尽时，它不再进入训练循环，也不会继续签到。其他 rank 会等待到超时：

```text
rank 0: 到达 barrier
rank 1: 到达 barrier
rank 2: 到达 barrier
rank 3: 数据耗尽，没有到达
结果: 60 秒后超时
```

程序捕获这个异常并把它当作“数据不均匀”的信号。问题在于：

> 通信超时代表 process group 出现错误，不是一种正常的流程控制方式。

超时后的 Gloo group 可能已经无法继续可靠工作。旧代码在每个 epoch 结束后销毁它，再在下一个 epoch 重新创建，最终可能出现 `DistStoreError`。

旧代码还同时使用了 DDP 的 `model.join()`。这样就存在两套不均匀输入处理逻辑：

1. PyTorch DDP `model.join()`。
2. CosyVoice 的 Gloo `monitored_barrier()`。

两套机制叠加后，rank 的退出和 collective 调用顺序更难保持一致。

另外，下面的兼容改动也存在维护风险：

```python
group_join._get_backend(torch.device("cpu")).options._timeout
```

`_get_backend` 和 `_timeout` 都是私有接口，PyTorch 不保证它们在不同版本之间保持兼容。`torch.device("cpu")` 本身没有问题，它只是选择 Gloo 的 CPU backend；真正的问题是依赖私有 API，以及通过超时完成正常退出。

## 4. 最终修复方案

最终方案不再等待超时，而是让每个 rank 主动报告自己是否还有数据。

### 4.1 显式同步 batch 状态

核心函数位于 `cosyvoice/utils/train_utils.py`：

```python
def distributed_batch_available(has_batch, control_group):
    if dist.get_world_size() == 1:
        return has_batch

    available = torch.tensor(int(has_batch), dtype=torch.int32)
    dist.all_reduce(available, op=dist.ReduceOp.MIN, group=control_group)
    return bool(available.item())
```

每个 rank 用一个整数表示状态：

```text
1: 当前 rank 还有 batch
0: 当前 rank 已经没有 batch
```

`all_reduce(MIN)` 会取所有 rank 中的最小值，并把结果发回所有 rank：

```text
[1, 1, 1, 1] -> 1，继续训练
[1, 1, 0, 1] -> 0，所有 rank 一起停止
```

这样不需要抛出异常，也不会破坏 Gloo group。

### 4.2 手动读取 DataLoader

普通的 `for` 循环遇到 `StopIteration` 会直接结束，无法先通知其他 rank。因此训练循环改为显式调用 `next()`：

```python
data_iter = iter(train_data_loader)

while True:
    try:
        batch_dict = next(data_iter)
        has_batch = True
    except StopIteration:
        batch_dict = None
        has_batch = False

    if not distributed_batch_available(has_batch, control_group):
        break

    # forward, backward, optimizer step
```

当任意 rank 数据耗尽时，所有 rank 在相同位置退出，因此 DDP collective 的调用次数保持一致。

### 4.3 在最短 rank 处结束 epoch

该方案选择以最短 rank 为准，而不是让数据较多的 rank 继续训练。

这样做的优点是：

- 所有 rank 的 optimizer step 数相同。
- scheduler 状态相同。
- Adam 等优化器的内部状态相同。
- 更适合 CPU ONNX 与 MUSA ONNX 的精度对比实验。

代价是其他 rank 最后可能有少量预取数据没有在本 epoch 使用。下一 epoch 会重新 shuffle 和分片。

### 4.4 丢弃不完整的梯度累积

当前配置使用：

```yaml
accum_grad: 2
```

也就是两个 batch 才执行一次参数更新。如果 epoch 在只累计了一个 batch 时结束，这部分梯度不能带到下一 epoch。

修复代码会调用：

```python
optimizer.zero_grad()
```

GAN 训练有两个 optimizer，因此会同时清理 generator 和 discriminator 的梯度。

日志中的以下信息是正常的：

```text
Epoch 0 discarded an incomplete gradient accumulation on rank N
```

它表示程序主动丢弃了不足一个完整累积周期的梯度，不是训练异常。

### 4.5 Gloo control group 只创建一次

旧代码每个 epoch 都创建和销毁 Gloo group。新代码在整个训练任务中只创建一次：

```python
control_group = dist.new_group(backend="gloo", timeout=...)

try:
    for epoch in ...:
        train_one_epoch(..., control_group)
finally:
    dist.destroy_process_group(control_group)
```

由于正常流程中不再故意触发超时，这个 group 可以在多个 epoch 之间持续复用。

### 4.6 进入 CV 前释放训练 DataLoader worker

引入手动 iterator 后，较长 rank 会提前停止，iterator 仍可能持有正在预取数据的 worker。

如果立即创建 CV iterator，就可能同时存在：

```text
训练 DataLoader worker
+ CV DataLoader worker
+ tokenizer 的 Rust Rayon 线程
```

八卡训练时进程和线程数量会快速增长，曾出现：

```text
ThreadPoolBuildError
Resource temporarily unavailable
DataLoader worker exited unexpectedly
```

因此在进入 CV 前显式释放：

```python
del data_iter
```

PyTorch DataLoader iterator 的析构逻辑会关闭对应 worker。最小测试结果为：

```text
workers_before_release=2
workers_after_release=0
```

### 4.7 限制 tokenizer 线程池

训练入口还设置了默认值：

```python
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
```

作用是防止每个 DataLoader worker 都创建较大的 tokenizer 线程池。使用 `setdefault` 意味着用户仍可通过环境变量显式覆盖。

## 5. 修复后的完整流程

```text
每个 rank 尝试读取一个 batch
              |
              v
生成 has_batch，值为 0 或 1
              |
              v
Gloo all_reduce(MIN)
              |
       +------+------+
       |             |
   结果为 1       结果为 0
       |             |
继续前向和反向   所有 rank 同步停止
                     |
                     v
            清理不完整梯度累积
                     |
                     v
             释放训练 iterator
                     |
                     v
               运行 CV
                     |
                     v
              保存 checkpoint
                     |
                     v
              进入下一 epoch
```

## 6. 如何从日志判断修复成功

一个完整 epoch 应该依次看到：

```text
Epoch 0 input exhausted first on rank 6
Epoch 0 discarded an incomplete gradient accumulation on rank ...
Epoch 0 Step ... on_batch_end True CV rank ...
Epoch 0 Step ... CV info ...
Checkpoint: save to checkpoint .../epoch_0_whole.pt
Epoch 1 TRAIN info ...
```

2026 年 6 月 7 日的八卡验证日志已经满足这些条件：

- epoch 0 在 rank 6 数据耗尽时同步结束。
- 8 个 rank 均完成 CV。
- `epoch_0_whole.pt` 成功保存。
- 8 个 rank 均进入 epoch 1 并继续训练。
- 没有 Gloo/MCCL timeout。
- 没有 DataLoader worker 或 Rayon thread pool 错误。
- 没有新增 MUSA core dump。

## 7. 常见疑问

### 固定随机种子是否导致了数据不均匀

不是根本原因。

固定种子会固定 shuffle 顺序，因此可能固定“哪个 rank 最先结束”，但 shard 数量、过滤规则和动态 batch 本来就可能造成不均匀。固定种子主要用于让实验可复现。

### 为什么不只使用 `model.join()`

`model.join()` 允许数据较多的 rank 继续训练，并模拟已经结束 rank 的 DDP 通信。该行为适合某些吞吐优先的训练，但当前实验更关注精度可比性，因此选择所有 rank 在最短输入处同时停止。

### 每个 batch 做一次 Gloo all-reduce 会不会很慢

同步内容只有一个 CPU `int32`，数据量非常小。旧实现本来每个 batch 就会执行一次 `monitored_barrier`，因此新方案通常不会比旧方案增加明显通信负担。

### `torch.device("cpu")` 是不是把模型放到了 CPU

不是。旧代码中的 CPU device 只是用于选择 Gloo backend，不会移动模型参数或训练 tensor。

## 8. 相关提交

```text
68d56f8 Make training data loading deterministic
Fix uneven distributed batch synchronization
```

其中：

- `68d56f8` 负责固定 Python、NumPy、Torch、MUSA 和 DataLoader worker 的随机状态。
- `Fix uneven distributed batch synchronization` 负责多卡不均匀输入同步、Gloo group 生命周期、梯度清理和 DataLoader worker 释放。
