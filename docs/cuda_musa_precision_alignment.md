# CosyVoice CUDA/MUSA 精度对齐改动说明

本文总结本分支为 CosyVoice3 训练增加的 CUDA/MUSA 精度对齐方案，重点覆盖 Flow 与 LLM 两部分的改动、参数含义、运行方式、`init.pt` 的使用方式，以及后续排查 loss 不一致时应该优先检查的点。

这里的“精度对齐”主要包含两件事：

1. 随机数轨迹对齐：同一个 step 内，CUDA 和 MUSA 尽量消费相同数量、相同顺序的随机数。
2. 数值路径对齐：尽量让两边走相同或更接近的算子实现，降低 fused kernel、AMP、SDPA 等实现差异带来的 loss 偏移。

需要注意，本文方案的目标是让工程对比实验尽量可解释、可复现，并不等价于承诺任意 CUDA/MUSA 组合在长训练中完全 bitwise identical。跨硬件、跨后端仍可能因为 GEMM、归约顺序、BF16/FP32 实现细节出现微小差异。

## 1. 背景问题

参考 `cosyvoice随机数精度对齐方案.pdf` 的分析，CUDA 与 MUSA 的随机数不一致，核心原因不是 seed 本身不同，而是 Philox 随机数状态在大 tensor 随机 kernel 中会受到 launch grid 形态影响。

在 S5000 与 A100/H100 的对比中，经验阈值是：

```text
numel > 184320
```

当一次随机 tensor 的元素数超过该阈值后，CUDA 与 MUSA 可能因为 kernel 切分不同而消费不同的 Philox counter。更麻烦的是，这类差异具有传播性：

```text
某一次大随机 tensor 不一致
-> 当前 tensor 内容不同
-> 后续 RNG state 也不同
-> 后续 dropout、randn、mask 全部可能继续偏离
-> loss 曲线变得不可直接对比
```

因此本次改动采用两类策略：

- 能关掉的训练随机项直接关掉，例如 Flow DiT dropout。
- 不能轻易关掉的随机项，避免触发超大随机 tensor，必要时跳过对应 batch 或完整梯度累积窗口。

LLM 的主要问题略有不同：当前 CosyVoice3 LLM 本身没有明显的大 Dropout 随机 mask，更多风险来自输入构造中的 Python `random.random()` 分支，以及 Qwen 注意力后端、AMP 等数值路径差异。因此 LLM 侧重点是“确定性输入构造 + 更保守的 FP32/eager 基线”。

## 2. 本次改动涉及的文件

主要改动文件如下：

```text
cosyvoice/bin/train.py
cosyvoice/utils/train_utils.py
cosyvoice/utils/executor.py
cosyvoice/llm/llm.py
examples/libritts/cosyvoice3/conf/cosyvoice3.yaml
examples/libritts/cosyvoice3/run_from_scratch_with_log_on_cuda.sh
examples/libritts/cosyvoice3/run_from_scratch_with_log_on_musa.sh
```

其中：

- `train.py` 新增命令行参数。
- `train_utils.py` 增加分布式 RNG 对齐检查、epoch 信息注入、resume signature 记录。
- `executor.py` 在训练和 CV 阶段执行 over-limit batch 的跳过逻辑。
- `llm.py` 增加 LLM deterministic bi-stream 选择和 Qwen attention backend 控制。
- `cosyvoice3.yaml` 打开 LLM 对齐配置，关闭 Flow dropout。
- CUDA/MUSA 启动脚本增加 Flow RNG guard、LLM FP32 对齐开关和共享 init checkpoint 入口。

## 3. Flow 侧改动

### 3.1 关闭 DiT dropout

配置位置：

```text
examples/libritts/cosyvoice3/conf/cosyvoice3.yaml
```

Flow 的 DiT estimator 中设置：

```yaml
estimator: !new:cosyvoice.flow.DiT.dit.DiT
    ...
    dropout: 0.0
```

原因是 DiT 内部 dropout mask 可能超过 `184320`，属于最容易触发 CUDA/MUSA RNG 轨迹分叉的随机项。对精度对齐实验来说，先关闭 dropout 可以显著减少随机数消费路径差异。

这项改动的含义是：

- 对 Flow 精度对齐实验生效。
- 会改变训练正则化行为，因此它是一个“对齐实验配置”，不是无条件推荐的最终生产训练配置。
- 如果后续要恢复原始训练策略，可以把 `dropout` 改回原值，但 CUDA/MUSA loss 对齐难度会明显增加。

### 3.2 新增 `--rng_alignment_max_speech_feat_numel`

命令行参数位置：

```text
cosyvoice/bin/train.py
```

新增参数：

```bash
--rng_alignment_max_speech_feat_numel 184320
```

语义：

```text
当任意 rank 的 batch_dict["speech_feat"].numel() 大于该阈值时，
认为这个 batch 有较高概率触发 CUDA/MUSA 大随机 kernel 不一致，
于是跳过当前完整梯度累积窗口。
```

默认值为 `0`，表示关闭该 guard。

当前 CUDA/MUSA 脚本只在 `MODELS=flow` 时自动传入该参数：

```bash
rng_alignment_args=()
if [ "${model}" = "flow" ]; then
  rng_alignment_args=(
    --rng_alignment_max_speech_feat_numel
    "${rng_alignment_max_speech_feat_numel:-184320}"
  )
fi
```

也就是说：

- Flow 默认启用阈值 `184320`。
- LLM 默认不启用该 guard。
- 可以通过环境变量覆盖：

```bash
rng_alignment_max_speech_feat_numel=0 MODELS=flow bash run_from_scratch_with_log_on_cuda.sh
```

或：

```bash
rng_alignment_max_speech_feat_numel=200000 MODELS=flow bash run_from_scratch_with_log_on_musa.sh
```

### 3.3 为什么跳过完整梯度累积窗口

当前训练配置存在 `accum_grad`。如果只跳过单个 micro-batch，很容易出现如下不一致：

```text
CUDA:
  micro-batch A -> 累积梯度
  micro-batch B -> 累积梯度
  optimizer.step()

MUSA:
  micro-batch A -> 被跳过
  micro-batch B -> 累积梯度
  optimizer.step()
```

这会让 optimizer step 虽然数量相同，但梯度内容不同，Adam 动量、scheduler 后续轨迹全部偏离。

因此实现上选择了更保守的策略：只要一个 batch 超过阈值，就丢弃它所在的完整梯度累积窗口，并在窗口边界前持续 `optimizer.zero_grad()`。

训练循环中的行为可以理解为：

```text
发现 over-limit batch
-> 标记 skip_accumulation_window=True
-> 当前窗口内所有 micro-batch 都不 forward/backward
-> 每个 micro-batch 前清空梯度
-> 到达 accum_grad 边界后恢复正常训练
```

这样做的好处是：

- 所有 rank 在同一个位置跳过。
- CUDA 和 MUSA optimizer step 边界保持一致。
- 不会把半个累积窗口的梯度带入下一次更新。

代价是会丢弃少量训练 batch。对精度对齐实验来说，这是为了换取随机数轨迹的可控性。

### 3.4 分布式 over-limit 判断

判断函数位于：

```text
cosyvoice/utils/train_utils.py
```

核心逻辑是：

```python
local_exceeded = int(
    torch.is_tensor(speech_feat) and speech_feat.numel() > max_speech_feat_numel
)
dist.all_reduce(exceeded, op=dist.ReduceOp.MAX, group=control_group)
```

也就是只要任意一个 rank 超过阈值，所有 rank 都认为当前 batch over-limit。

这点很重要。DDP 训练中不能让某些 rank forward/backward，另一些 rank 跳过，否则 collective 调用顺序会不一致，轻则 loss 对不上，重则直接卡死或通信报错。

当前该参数只支持 `torch_ddp`，配置检查中会拒绝其它训练引擎：

```python
if args.rng_alignment_max_speech_feat_numel > 0 and args.train_engine != 'torch_ddp':
    raise ValueError(...)
```

### 3.5 CV 阶段的处理

CV 不涉及 optimizer 更新，因此处理策略更简单：

```text
CV batch over-limit
-> 当前 batch 直接跳过
-> 继续统计其它 CV batch 的 loss
```

如果所有 CV batch 都因为阈值被跳过，会抛出错误：

```text
All CV batches were skipped by rng_alignment_max_speech_feat_numel=...
```

这通常说明阈值太低，或者数据 batch 太大，需要调小 batch 规模、调高阈值，或者关闭该 guard 后重新确认风险。

## 4. LLM 侧改动

### 4.1 LLM 为什么也需要改

CosyVoice3 LLM 当前并没有像 Flow DiT 那样明显的大 Dropout mask：

- Qwen 配置中 `attention_dropout` 为 `0.0`。
- 模型中没有常规 `torch.nn.Dropout` 模块参与训练随机。
- HF SDPA 在 dropout 为 0 时本身不消费 dropout 随机数。

因此 LLM 的 CUDA/MUSA 对齐风险主要来自：

1. 输入构造阶段使用 Python `random.random()` 随机决定 bi-stream / uni-stream。
2. 注意力后端可能在 CUDA/MUSA 上走不同 fused kernel。
3. `--use_amp` 下 BF16/AMP 路径更容易放大后端实现差异。

本次 LLM 改动就是针对这三点。

### 4.2 deterministic bi-stream 选择

这一节展开说明两个问题：

1. CosyVoice LLM 里的 bi-stream 到底是什么。
2. 为什么本次把 `random.random()` 改成了基于 hash 的 deterministic 选择。

先澄清一个容易混淆的点：这里的 bi-stream 不是 CUDA stream，也不是 DDP 通信里的 stream。它指的是 LLM 训练时的一种序列组织方式，即把 text token 流和 speech token 流交错起来训练。

CosyVoice3 LLM 并不直接预测波形，而是预测离散语音 token。一个训练样本里通常有：

```text
instruct_token  指令，例如 You are a helpful assistant...
text_token      文本 token
speech_token    离散语音 token
```

LLM 的训练目标可以概括为：

```text
给定文本 token，以及已经生成的历史 speech token
-> 预测后续 speech token
```

文本位置本身通常不计算 speech-token loss，因此 target 中会填 `IGNORE_ID`。

#### 4.2.1 uni-stream：完整文本在前，完整语音在后

uni-stream 是最直观的 TTS 训练格式：

```text
输入:
[SOS] [instruct] [完整文本] [TASK_ID] [speech_1] [speech_2] ... [speech_N]

目标:
忽略  忽略       忽略文本     speech_1 speech_2 ... speech_N EOS
```

假设文本 token 是：

```text
T1 T2 T3 T4 T5
```

语音 token 是：

```text
S1 S2 S3 S4 S5 S6
```

那么 uni-stream 的输入可以理解成：

```text
[SOS] T1 T2 T3 T4 T5 [TASK] S1 S2 S3 S4 S5 S6
```

target 可以理解成：

```text
IGNORE IGNORE IGNORE IGNORE IGNORE IGNORE S1 S2 S3 S4 S5 S6 EOS
```

由于自回归语言模型通常是在当前位置预测下一个 token，所以它的训练含义是：

```text
先看到完整文本 T1~T5
-> 从 TASK 位置开始预测 S1
-> 看到 S1 后预测 S2
-> 看到 S2 后预测 S3
-> ...
-> 最后预测 EOS
```

这种格式的优点是上下文完整、逻辑简单。缺点是它更像非流式 TTS：模型习惯先拿到完整文本，再开始生成完整语音。

#### 4.2.2 bi-stream：文本块和语音块交错推进

bi-stream 则是另一种训练格式：文本和语音不再是“文本全部在前，语音全部在后”，而是按比例交错。

当前 CosyVoice3 配置中：

```yaml
mix_ratio: [5, 15]
```

可以理解为：

```text
每 5 个 text token
对应约 15 个 speech token
```

代码中会按这个比例切块：

```python
this_text_token = text_token[i][j * self.mix_ratio[0]: (j + 1) * self.mix_ratio[0]]
this_speech_token = speech_token[i][j * self.mix_ratio[1]: (j + 1) * self.mix_ratio[1]]
```

假设：

```text
mix_ratio = [5, 15]
```

文本是：

```text
T1 T2 T3 T4 T5 T6 T7 T8 T9 T10
```

语音是：

```text
S1 ... S30
```

那么 bi-stream 的输入结构大致是：

```text
[SOS]
T1 T2 T3 T4 T5
S1 S2 ... S15
T6 T7 T8 T9 T10
S16 S17 ... S30
```

也就是：

```text
文本块 -> 对应语音块 -> 文本块 -> 对应语音块
```

它让模型学习：

```text
读一小段文本
-> 生成一小段语音
-> 再读下一小段文本
-> 再生成下一小段语音
```

这和 uni-stream 的“完整文本 -> 完整语音”是两种不同的训练布局。

#### 4.2.3 bi-stream 的 target 如何对齐

以一个完整 chunk 为例，输入是：

```text
T1 T2 T3 T4 T5 S1 S2 ... S15
```

target 大致是：

```text
IGNORE IGNORE IGNORE IGNORE S1 S2 S3 ... S15 FILL
```

这里有一个细节：5 个文本 token 中，前 4 个位置不计算 speech loss，第 5 个文本 token 的输出开始预测第一个 speech token。

也就是：

```text
看到 T5 的位置，预测 S1
看到 S1 的位置，预测 S2
看到 S2 的位置，预测 S3
...
看到 S15 的位置，预测 FILL
```

`FILL` 是一个特殊 token，可以理解成当前 speech chunk 的边界标记。它告诉模型：这一段语音块结束了，后面还会接下一段文本/语音结构。

对应代码是：

```python
this_lm_target += [IGNORE_ID] * (self.mix_ratio[0] - 1)
this_lm_target += this_speech_token
this_lm_target.append(self.fill_token)

this_lm_input.append(text_token_emb[i][...])
this_lm_input.append(speech_token_emb[i][...])
```

这个设计的效果是：

```text
模型读完一个文本小块后，马上开始预测对应的语音小块。
```

#### 4.2.4 为什么训练中要使用 bi-stream

使用 bi-stream 主要有三个目的。

第一，它更接近流式或低延迟生成。

如果只用 uni-stream，模型训练时永远看到：

```text
完整文本 -> 完整语音
```

它更习惯等文本全部到齐后再开始生成语音。而 bi-stream 训练让模型学习：

```text
文本块 -> 语音块 -> 文本块 -> 语音块
```

这更接近 streaming TTS 中“读一段，说一段”的使用方式。

第二，它增强了文本和语音的局部对齐。

TTS 中文本和语音通常近似单调对齐：

```text
前面的文字 -> 前面的声音
后面的文字 -> 后面的声音
```

uni-stream 把完整文本都放在前面，模型需要自己从较长上下文中学习局部对应关系。bi-stream 则显式提供一种局部结构：

```text
T1~T5   -> S1~S15
T6~T10  -> S16~S30
```

这个比例不是严格的语言学对齐，而是工程上的近似；当前配置里使用的是 `5 text token : 15 speech token`。

第三，它相当于一种训练数据增强。

原始代码中，每个样本大约 50% 概率走 bi-stream，另外 50% 走 uni-stream：

```python
random.random() < 0.5
```

这让模型同时适应两类输入布局：

```text
一部分样本训练完整文本条件生成
一部分样本训练文本/语音交错生成
```

因此本次改动没有取消 bi-stream，而是保留这个训练形式，只是把“样本走 bi-stream 还是 uni-stream”的决策改成跨平台可复现。

#### 4.2.5 什么时候会走 bi-stream

代码中的实际判断是：

```python
if self._use_bistream(utt, epoch) and speech_token_len[i] / text_token_len[i] > self.mix_ratio[1] / self.mix_ratio[0]:
    ...
```

也就是说，需要同时满足：

```text
1. 当前样本被选择使用 bi-stream。
2. speech_token_len / text_token_len > 15 / 5。
```

第二个条件是为了避免语音 token 太短时强行切块。因为 bi-stream 要按 `5:15` 的比例组织序列，如果 speech token 数量不够，硬切会破坏样本结构，所以这种情况下会回退到 uni-stream。

#### 4.2.6 为什么不能继续用 `random.random()`

原始逻辑是：

```python
if random.random() < 0.5:
    用 bi-stream
else:
    用 uni-stream
```

这在普通训练中没有问题，但在 CUDA/MUSA loss 对齐实验中风险很大。

因为 bi-stream 和 uni-stream 的差异不是一个很小的数值扰动，而是整个 `lm_input` 和 `lm_target` 布局都不同。

同一个样本，如果 CUDA 走 uni-stream，MUSA 走 bi-stream，那么两边实际训练的已经不是同一个输入：

```text
CUDA:
[SOS] [完整文本] [TASK] [完整语音...]

MUSA:
[SOS] [文本块1] [语音块1] [文本块2] [语音块2] ...
```

这种情况下 loss 不对齐是必然的，而且它不是算子精度问题，而是数据构造路径已经分叉了。

`random.random()` 依赖 Python RNG 的全局状态。它本质上是：

```text
第 1 次调用 -> 拿第 1 个随机数
第 2 次调用 -> 拿第 2 个随机数
第 3 次调用 -> 拿第 3 个随机数
...
```

如果 CUDA/MUSA 任意一侧在前面多消费或少消费了一次 Python random，后面的所有 bi-stream 选择都会整体错位：

```text
原本样本 A 用第 100 个随机数
错位后样本 A 用第 101 个随机数
-> bi-stream / uni-stream 选择可能不同
-> lm_input / lm_target 构造不同
-> 后续 loss 不再具备可比性
```

因此，本次不是为了“去掉随机性”，而是为了去掉“依赖调用顺序的随机性”。

#### 4.2.7 hash 替代随机数的原理

配置位置：

```text
examples/libritts/cosyvoice3/conf/cosyvoice3.yaml
```

新增配置：

```yaml
llm: !new:cosyvoice.llm.llm.CosyVoice3LM
    ...
    bistream_mode: deterministic
    bistream_seed: !ref <seed>
```

代码位置：

```text
cosyvoice/llm/llm.py
```

`Qwen2LM` 与 `CosyVoice3LM` 新增两个参数：

```python
bistream_mode: str = 'random'
bistream_seed: int = 1986
```

支持两种模式：

```text
random         保持原始行为，使用 random.random() < 0.5
deterministic  使用 seed + epoch + utt id 做稳定 hash
```

deterministic 模式下，是否使用 bi-stream 由以下信息决定：

```text
bistream_seed
epoch
utt id
```

代码逻辑是：

```python
digest = hashlib.blake2b(
    f'{self.bistream_seed}:{epoch}:{utt}'.encode('utf-8'),
    digest_size=8,
).digest()
return int.from_bytes(digest, byteorder='little') < (1 << 63)
```

可以把它理解成一个无状态的伪随机抛硬币函数：

```text
(bistream_seed, epoch, utt_id)
-> blake2b hash
-> 64-bit 整数
-> 与 2^63 比较
-> True / False
```

`digest_size=8` 表示取 8 字节，也就是 64 bit。64-bit 整数空间中大约一半小于 `2^63`，因此：

```text
hash_value < 2^63
```

就近似对应 50% 概率。也就是说，它仍然保留“约一半样本走 bi-stream”的行为，但不再消费任何全局 RNG 状态。

两种方式的差异可以概括为：

```text
原始 random.random():
  Python RNG 当前状态 -> 下一个随机数 -> 消费 RNG 状态 -> 得到 True/False

deterministic hash:
  seed + epoch + utt_id -> hash -> 得到 True/False
```

前者依赖“这是第几次调用 random”，后者只依赖“这是哪个样本、哪个 epoch、哪个 seed”。

因此同一个 utterance 在同一个 epoch 上：

```text
CUDA 一定得到同一个 bi-stream 决策
MUSA 一定得到同一个 bi-stream 决策
```

只要 `bistream_seed`、`epoch` 和 `utt id` 相同，两边就不会因为 Python random consumption order 不同而选出不同输入布局。

#### 4.2.8 为什么使用 `utt_id` 和 `epoch`

使用 `utt_id` 是因为它代表样本身份。

如果使用 batch 内下标，例如 `i = 0, 1, 2...`，同一个样本换了 batch 位置，结果就会变化：

```text
这次样本 A 在 batch[0] -> bi-stream
下次样本 A 在 batch[3] -> uni-stream
```

这不利于跨平台对齐，也不利于 resume 后保持可解释性。

使用 `utt_id` 后：

```text
样本 A 不管在哪个 batch、哪个 rank
只要 epoch 和 seed 一样
结果都一样
```

使用 `epoch` 是为了保留原始随机数据增强的味道。

如果只使用：

```text
seed + utt_id
```

那么同一个样本在所有 epoch 都永远固定成同一种格式。加入 `epoch` 后：

```text
epoch 0: 样本 A 可能是 bi-stream
epoch 1: 样本 A 可能变成 uni-stream
epoch 2: 样本 A 又可能是 bi-stream
```

这样每个 epoch 内对 CUDA/MUSA 是稳定的，不同 epoch 之间仍然可以变化。

总结来说，deterministic bi-stream 不是取消 bi-stream，也不是固定所有样本都走一种格式，而是把这个决策改成：

```text
伪随机
无状态
可复现
与调用顺序无关
由样本身份和 epoch 决定
```

这可以先把 LLM 输入构造层面的离散不确定性拿掉，让后续 CUDA/MUSA loss 差异更像真正的数值或算子问题。

### 4.3 batch 注入 `_epoch`

deterministic bi-stream 需要知道当前 epoch。训练入口在统一 forward 前注入：

```python
batch['_epoch'] = info_dict.get('epoch', 0)
```

位置：

```text
cosyvoice/utils/train_utils.py
```

LLM forward 中会把它传给输入构造函数：

```python
utts=batch.get('utts')
epoch=batch.get('_epoch', 0)
```

DPO 路径中，batch 会被 repeat 两份，因此也同步复制了 `utts`，避免 chosen/rejected 的输入构造发生错位。

### 4.4 deterministic 模式要求 batch 中有 `utts`

deterministic 模式依赖 utterance id。如果 batch 中没有 `utts`，或者数量和当前 batch 样本数不一致，会直接报错：

```text
deterministic bistream mode requires one utterance id per LLM input
```

这是故意设计成 fail-fast。否则如果退回到 index 或随机逻辑，表面上能跑，但 CUDA/MUSA 对齐结果会变得不可解释。

### 4.5 Qwen attention backend 改为可配置

代码位置：

```text
cosyvoice/llm/llm.py
```

`Qwen2Encoder` 新增：

```python
def __init__(self, pretrain_path, attn_implementation='sdpa'):
    self.model = Qwen2ForCausalLM.from_pretrained(
        pretrain_path,
        attn_implementation=attn_implementation,
    )
```

配置中设置：

```yaml
llm: !new:cosyvoice.llm.llm.Qwen2Encoder
    pretrain_path: !ref <qwen_pretrain_path>
    attn_implementation: eager
```

这样做的目的不是追求最快，而是先建立一个更保守的跨平台 FP32 对齐基线：

- `sdpa` 可能在不同后端走不同 fused kernel。
- `eager` 更偏向 unfused PyTorch 路径，便于对齐和排查。
- 如果确认 loss 已经对齐，再考虑切回 `sdpa` 做性能实验。

启动时可能看到 Transformers 关于 sliding window attention 与 `eager` 的 warning。这是后端能力提示，不代表代码没有生效。做最终效果或性能训练时，需要额外确认 `eager` 与 `sdpa` 的模型行为差异是否可接受。

### 4.6 LLM 默认关闭 AMP，使用 FP32 对齐基线

CUDA/MUSA 启动脚本中新增：

```bash
llm_alignment_fp32="${llm_alignment_fp32:-1}"
```

训练参数构造逻辑是：

```bash
precision_args=(--use_amp)
if [ "${model}" = "llm" ] && [ "${llm_alignment_fp32}" = "1" ]; then
  precision_args=()
fi
```

因此：

- Flow 默认仍使用 `--use_amp`。
- LLM 默认不传 `--use_amp`，即走 FP32。
- 如需恢复 LLM AMP 训练，可以显式设置：

```bash
llm_alignment_fp32=0 MODELS=llm bash run_from_scratch_with_log_on_cuda.sh
```

或：

```bash
llm_alignment_fp32=0 MODELS=llm bash run_from_scratch_with_log_on_musa.sh
```

建议排查对齐问题时先使用默认 `llm_alignment_fp32=1`。如果 FP32 下已经不对齐，优先排查数据、初始化、输入构造和 checkpoint；如果 FP32 对齐但 AMP 不对齐，再进入 BF16/AMP kernel 差异分析。

## 5. 启动脚本改动

两个脚本都做了对齐相关改造：

```text
examples/libritts/cosyvoice3/run_from_scratch_with_log_on_cuda.sh
examples/libritts/cosyvoice3/run_from_scratch_with_log_on_musa.sh
```

### 5.1 每次只训练一个模块

脚本要求每次只选择一个模型：

```bash
MODELS=llm
```

或：

```bash
MODELS=flow
```

如果传入多个模型会退出：

```text
Select exactly one model per scratch run: MODELS=llm or MODELS=flow.
```

原因是 LLM 和 Flow 的对齐策略不同：

- LLM：默认 FP32、deterministic bi-stream、Qwen eager attention。
- Flow：默认 AMP、关闭 DiT dropout、开启 speech_feat numel guard。

把多个模块混在一次 launch 里，不利于定位偏差来源。

### 5.2 Flow 自动带 RNG guard

`MODELS=flow` 时脚本会自动传：

```bash
--rng_alignment_max_speech_feat_numel 184320
```

可通过环境变量覆盖：

```bash
rng_alignment_max_speech_feat_numel=0 MODELS=flow bash run_from_scratch_with_log_on_cuda.sh
```

### 5.3 LLM 默认 FP32

`MODELS=llm` 且未额外设置时：

```bash
llm_alignment_fp32=1
```

脚本不会传 `--use_amp`。

如果要跑原 AMP 路径：

```bash
llm_alignment_fp32=0 MODELS=llm bash run_from_scratch_with_log_on_cuda.sh
```

### 5.4 新增 `shared_init_checkpoint`

两个脚本都支持：

```bash
shared_init_checkpoint=/path/to/init.pt
```

如果该变量非空，脚本会检查文件是否存在，并把训练入口参数改为：

```bash
--checkpoint /path/to/init.pt
```

如果不设置该变量：

- CUDA 脚本默认使用 `--resume auto`。
- MUSA 脚本默认使用 `--resume auto --resume_mode cross_platform`。

也就是说，`shared_init_checkpoint` 优先级高于 `--resume auto`，它用于“从相同初始权重 warm start”，不是用于“完整断点续训”。

## 6. `init.pt` 的正确使用方式

### 6.1 为什么需要共享 `init.pt`

做 CUDA/MUSA 对齐时，两边必须从完全相同的初始权重开始。

虽然设置相同 seed 后，很多 CPU 初始化路径可能刚好一致，但这不是足够稳妥的工程假设。只要某个模块初始化使用了不同后端、不同库版本或不同随机数消费顺序，初始参数就可能不同，后续 loss 对齐就没有意义。

因此推荐流程是：

```text
先在一侧生成 init.pt
-> 拷贝到另一侧机器
-> 另一侧显式通过 shared_init_checkpoint 加载同一个 init.pt
```

### 6.2 `init.pt` 不会被自动读取

`init.pt` 不会被 `--resume auto` 自动发现。

原因是 checkpoint 保存逻辑中，`latest_ddp` 不会指向 `init.pt`：

```python
if info_dict["train_engine"] == "torch_ddp" and model_name != 'init':
    _atomic_write_text(
        os.path.basename(save_model_path),
        os.path.join(model_dir, 'latest_ddp'))
```

`--resume auto` 查找的是当前机器、当前 `model_dir` 下的 resume 指针或 epoch/step checkpoint。它不会跨机器寻找文件，也不会把 `init.pt` 当成自动恢复目标。

所以跨机器时必须显式传：

```bash
shared_init_checkpoint=/absolute/path/to/init.pt
```

### 6.3 推荐 LLM 对齐启动流程

假设先在 CUDA 机器上生成或使用初始 checkpoint：

```bash
cd /home/cosyvoice-test/CosyVoice/examples/libritts/cosyvoice3
MODELS=llm bash run_from_scratch_with_log_on_cuda.sh
```

默认实验目录为：

```text
examples/libritts/cosyvoice3/exp/cosyvoice3_scratch/llm/torch_ddp
```

如果已经生成：

```text
init.pt
```

建议先计算校验值：

```bash
sha256sum exp/cosyvoice3_scratch/llm/torch_ddp/init.pt
```

然后把该文件拷贝到 MUSA 机器，并在 MUSA 机器上确认校验值一致：

```bash
sha256sum /path/on/musa/init.pt
```

MUSA 侧启动：

```bash
cd /home/cosyvoice-test/CosyVoice/examples/libritts/cosyvoice3
MODELS=llm \
shared_init_checkpoint=/path/on/musa/init.pt \
bash run_from_scratch_with_log_on_musa.sh
```

如果 CUDA 和 MUSA 共用同一块 NFS，也仍然建议显式设置：

```bash
shared_init_checkpoint=/shared/path/init.pt
```

这样日志里会明确打印：

```text
LLM FP32 alignment mode: 1; shared init: /shared/path/init.pt
```

### 6.4 `--checkpoint` 与 `--resume` 的区别

`shared_init_checkpoint` 最终走的是 `--checkpoint`，它的语义是 warm start：

```text
只加载模型权重
不恢复 optimizer
不恢复 scheduler
不恢复 AMP GradScaler
不恢复 DataLoader batch 位置
不恢复各 rank RNG state
```

如果目标是从中断位置继续训练，应该使用：

```bash
--resume auto
```

或显式指定某个 epoch/step checkpoint。

简单区分：

```text
相同随机初始化起跑线 -> shared_init_checkpoint / --checkpoint init.pt
中断后继续同一条训练轨迹 -> --resume auto
```

不要用 `init.pt` 当断点续训 checkpoint；也不要指望 `--resume auto` 自动找到另一台机器上的 `init.pt`。

## 7. 推荐排查顺序

当 CUDA/MUSA loss 仍然不对齐时，建议按下面顺序排查。这个顺序是刻意设计的：先检查最容易造成巨大偏差的离散条件，再看数值误差。

### 7.1 确认起始权重一致

检查两边是否确实加载同一个 `init.pt`：

```bash
sha256sum /path/to/init.pt
```

同时检查日志中是否出现：

```text
shared init: /path/to/init.pt
```

如果两边起始权重不同，后面所有 loss 对比都不成立。

### 7.2 确认数据顺序一致

关键参数包括：

```text
seed
data_seed
num_workers
prefetch
world_size
train_data
cv_data
```

这些参数已经进入 `resume_signature`，resume 时会用于一致性检查。

如果 CUDA/MUSA 使用不同数据列表、不同 rank 数或不同 batch 过滤结果，loss 会从第一个 step 就开始偏离。

### 7.3 LLM 先看输入构造

确认配置中是：

```yaml
bistream_mode: deterministic
bistream_seed: !ref <seed>
attn_implementation: eager
```

确认启动脚本中没有关闭 FP32 对齐：

```text
LLM FP32 alignment mode: 1
```

如果 deterministic 模式报缺少 `utts`，应该修数据管线，不建议临时退回随机模式。

### 7.4 Flow 先看 over-limit 日志

Flow 训练日志中可能出现：

```text
exceeds CUDA/MUSA RNG alignment limit 184320;
discarding this gradient-accumulation window
```

这是预期行为，表示脚本为了对齐跳过了高风险 batch。

如果日志大量出现该信息，说明当前数据 batch 太大或阈值过低。可以考虑：

- 降低动态 batch 尺寸。
- 调整数据过滤。
- 临时提高阈值做对比实验。
- 关闭 guard 后确认是否出现 RNG 轨迹分叉。

### 7.5 再看 AMP/BF16 和 attention 后端

如果 LLM FP32 + eager 已经对齐，再尝试：

```bash
llm_alignment_fp32=0
```

或把：

```yaml
attn_implementation: eager
```

改回：

```yaml
attn_implementation: sdpa
```

这类改动主要用于性能或最终训练配置验证，不建议在基础对齐尚未完成时一起打开。

## 8. 参数速查

| 参数或环境变量 | 默认值 | 作用范围 | 说明 |
| --- | --- | --- | --- |
| `--rng_alignment_max_speech_feat_numel` | `0` | train.py | 大于 0 时启用 Flow RNG guard；当前仅支持 `torch_ddp`。 |
| `rng_alignment_max_speech_feat_numel` | `184320` | shell 脚本 Flow 分支 | 脚本传给 `--rng_alignment_max_speech_feat_numel` 的默认阈值。 |
| `bistream_mode` | `deterministic` in CosyVoice3 config | LLM | 控制 bi-stream/uni-stream 分支是否依赖稳定 hash。 |
| `bistream_seed` | `seed` | LLM | deterministic bi-stream 使用的 hash seed。 |
| `attn_implementation` | `eager` in CosyVoice3 config | Qwen2Encoder | 控制 Qwen attention 后端，便于建立保守对齐基线。 |
| `llm_alignment_fp32` | `1` | shell 脚本 LLM 分支 | 为 1 时 LLM 不传 `--use_amp`，走 FP32。 |
| `shared_init_checkpoint` | 空 | shell 脚本 | 非空时使用 `--checkpoint` warm start 到指定 init 权重。 |
| `resume_mode` | `cross_platform` in MUSA script | MUSA shell 脚本 | MUSA `--resume auto` 时使用的跨平台恢复模式。 |

## 9. 当前方案的边界

### 9.1 不能保证所有随机项都完全消失

Flow 中仍然有 timestep、noise、mask 等随机项。当前方案是通过关闭大 dropout 和跳过高风险 batch 来降低 CUDA/MUSA RNG 分叉概率，而不是把训练完全改成无随机训练。

### 9.2 不建议把对齐配置直接等同于最优训练配置

例如：

- Flow `dropout: 0.0` 会减少正则。
- LLM `attn_implementation: eager` 可能比 `sdpa` 慢。
- LLM FP32 会比 AMP 占用更多显存、速度更慢。

这些都是为了对齐实验做出的工程取舍。完成定位后，可以逐项恢复性能配置，并观察 loss 偏差从哪一步开始出现。

### 9.3 `shared_init_checkpoint` 不是 resume

这点非常重要：

```text
shared_init_checkpoint 只保证模型初始权重一致。
resume 才保证 optimizer/scheduler/RNG/DataLoader 训练现场恢复。
```

如果已经训练了若干 step 后中断，应该使用 `--resume auto`，不要再拿 `init.pt` 重新 warm start，否则会回到初始权重重新训练。

## 10. 已做的本地验证

本次改动已经做过以下静态和轻量级验证：

```bash
python -m py_compile \
  cosyvoice/llm/llm.py \
  cosyvoice/utils/train_utils.py \
  cosyvoice/utils/executor.py \
  cosyvoice/bin/train.py
```

```bash
bash -n \
  examples/libritts/cosyvoice3/run_from_scratch_with_log_on_cuda.sh \
  examples/libritts/cosyvoice3/run_from_scratch_with_log_on_musa.sh
```

```bash
git diff --check
```

另外也验证过：

- CosyVoice3 配置可以实例化。
- `bistream_mode=deterministic` 生效。
- Qwen attention backend 配置为 `eager`。
- Qwen attention dropout 为 `0.0`。
- 模型中没有常规 `torch.nn.Dropout` 模块参与 LLM 训练随机。
- deterministic bi-stream 在消耗额外 Python random 后仍保持稳定。

当前环境只有 MUSA 可用，不能在同一台机器上完成 CUDA/MUSA 双端 loss 对比。因此最终对齐效果仍需要结合 CUDA 机器与 MUSA 机器的实际训练日志确认。

## 11. 建议的对齐实验记录格式

后续做 CUDA/MUSA 对比时，建议每次记录下面信息，避免排查时丢上下文：

```text
git commit:
model: llm / flow
platform: CUDA / MUSA
world_size:
visible devices:
init.pt sha256:
train_data sha256 or line count:
cv_data sha256 or line count:
seed:
data_seed:
llm_alignment_fp32:
attn_implementation:
rng_alignment_max_speech_feat_numel:
resume or checkpoint args:
first 10 train loss:
first 10 cv loss:
over-limit skip count:
```

尤其要记录 `init.pt sha256` 和启动日志中的 `shared init`。这两个点是最常见、也最容易忽略的对齐前提。
