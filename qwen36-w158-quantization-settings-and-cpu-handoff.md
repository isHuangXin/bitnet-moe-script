# Qwen3.6-35B-A3B：w1.58 量化位置实验与 CPU 推理交接计划

更新日期：2026-09-17，公开权重自包含版 v2。本文是本任务的**单文件交接规范**：
接手 agent **没有本机文件、私有 checkpoint、修改版 Megatron/vLLM 或此前对话**，
只能取得本 Markdown、官方模型架构/权重，以及公开的 Python/PyTorch 等依赖。
关键量化规则、可运行 CPU 参考代码、模块识别、实验矩阵和验收计划都包含在本文。
模型公开架构不重复展开；仅保留确定分组、权重映射和空间所需的尺寸。

**S1 是官方原始 BF16 权重与官方 forward 的基础 baseline，必须参加全部主对比。**
下文 S2–S9 是从同一官方权重构建的公开 PTQ 候选，不是本机 iteration 2000 的副本。

## 1. 已确定的决策与当前状态

- 本文统一写 **w1.58**。它表示三值权重量化，**物理存储是
  2 bit/weight 的 packed2**，不是 1.58 bit 文件，也不是 INT8 权重精度。
- w1.58 的固定 recipe：沿每个输出行的输入维度 K 按 128 分组，abs-mean scale，
  fixed threshold `0.5`，三值 codes，BF16 weight scale。
- S2–S9 的 Embedding 和 LM head 为 **INT8 codes + BF16 reciprocal scale，group-128**；
  它们不是 w1.58，也不因为专家使用 A8 就自动改成 INT8 GEMM。
- **S1：官方原始 BF16**，包括 BF16 experts、embedding、LM head，不启用任何 fake quant。
  旧讨论中删除的“只量化 experts”方案不再保留；S1 现在专指原版 BF16 baseline。
- S3/S4/S5/S6/S7 新增模块全部使用 **w1.58**，不是 INT8。
- S8/S9 是**直接基于 S2 的独立分支**：把其余大 BF16 Linear 分别转换成
  FP8/INT8 + block scale，不能先变成 S7 的三值，再“升回”8 bit。
- 本轮目标是明确格式、导出候选、评估量化位置对下游质量的影响，并交给其他 agent
  优化 CPU 吞吐。本文不是 CPU kernel 已实现或各方案已跑分的声明。
- 此前本机存在 QAT 模型和 FP32-scale packed2 导出，它们仅解释 recipe 的来源，
  **不是接手所需输入或可下载基准**。公开 S1–S9 的导出、适配和评测仍需执行。
  不能只改配置字符串，就声称加载时已使用新格式。

所有“scale 改 BF16”指**落盘和常驻的权重量化 metadata**。计算 scale 的 reduction、
反量化中的乘除和必要的累加仍采用 FP32。动态 A8 activation scale 属于临时运行数据，
本轮使用第 3 节定义的 FP32 流程；不将其悄悄改成 BF16，混入另一个误差变量。

### 1.1 唯一必需的模型来源：官方 BF16 snapshot

官方模型：<https://huggingface.co/Qwen/Qwen3.6-35B-A3B>。
不要下载同名 FP8 变体作为 S1，也不要把已经量化过的权重当作原始 BF16 来源。
本文空间依据的官方 revision 为
`995ad96eacd98c81ed38be0c5b274b04031597b0`；应固定 commit，而不是浮动的 `main`。
示例使用公开 Hugging Face CLI，路径由接手 agent 在自己的环境创建：

```bash
hf download Qwen/Qwen3.6-35B-A3B \
  --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --local-dir ./qwen36-bf16-source
```

若该 revision 无法取得，应明确报告并固定另一个官方 commit、重新审计 shape/字节数，
不能默默用新权重沿用旧结果。读取官方 `config.json`、safetensors index、tokenizer、
chat template 和模型卡，记录文件 hash 及实际 Transformers/PyTorch/backend 版本。
依照官方 config 选择模型实现；模型名包含 3.6，不代表其 `model_type` 必须叫 3.6。
此 snapshot 使用 `qwen3_5_moe` / `qwen3_5_moe_text` 类型。

本实验只使用 text model，排除 `model.visual.*` 和 `mtp.*`，保留文本 embedding 与独立
`lm_head.weight`。原始 sharding 可能混放不同部分，应按 tensor key 过滤，不按 shard
文件名猜测。保持官方 RoPE、attention gate、GDN、tokenizer/chat template 语义，
不要为了加载方便把模型改成另一种公开 Qwen 架构。

### 1.2 公开主实验与不可复现的私有 QAT 结果分开

| 项目 | 本文的公开主实验 | 如果以后另外获得私有 QAT 权重 |
| --- | --- | --- |
| source | 同一官方 BF16 revision | 独立声明的私有 source/hash |
| S1 | 官方 BF16 权重、官方 forward | 仍可作外部参考，但不是同一训练结果 |
| S2–S9 | 按本文从官方浮点权重独立生成 codes/scales | 可另建 `source_track=qat` 的矩阵 |
| 是否需要本机代码/导出/golden | **不需要** | 只有实际交付后才能使用，不能推断存在 |
| 可以声称的结果 | 此次官方权重 PTQ 的质量、吞吐、内存 | 此次提供的 QAT checkpoint 的结果 |

**仅凭官方 BF16 权重和量化公式，无法恢复私有 iteration 2000 的 QAT 学习结果。**
直接将 32B 级专家权重三值化可能显著降低质量，甚至无法正常完成任务。这是公开 PTQ
需要测量的风险，不是可以默认通过的步骤，也不能据此判断经过 QAT 的 recipe 必然无效。
如果需要研究 QAT，则另外固定公开训练数据、预算、seed 和训练实现，形成新的、
可复现训练实验；不能声称它复现了没有获得的原训练 checkpoint。

公开 S2–S9 未量化部分始终来自同一官方 source；量化器运行一次后缓存 codes/scales，
各 setting 复用相同模块的同一份量化结果。训练 checkpoint 与 optimizer 格式不属于
本交接的前置条件。

### 1.3 独立执行所需依赖与数值基准

- 本文 Python 参考函数仅依赖公开 PyTorch；INT32 matmul 和 FP8 cast 示例在 CPU
  使用，选择具备 `torch.float8_e4m3fn` 的版本。下载依赖公开 `huggingface_hub`/CLI，
  整模型使用能加载该官方 config 的公开 Transformers 或其他公开 backend。
- 参考函数按文内出现顺序执行，共享 `import torch`，**不 import 私有 Megatron/vLLM
  扩展，不调用本机脚本**。它们是小张量正确性基准，不是高性能整模型实现。
- 公共轨迹固定 CPU/PyTorch 参考环境及其代码 hash，先生成 golden；优化 kernel 对
  这份 reference 验收。文中 CUDA 片段只解释历史舍入，无法访问原 kernel 时不要求
  对不可取得的 CUDA golden 逐位相等。
- 严格保留参数来源与运算次序，但不同 backend 的 reduction、除法或 FP8 cast 仍可能
  存在边界差异；记录并验收，不能无证据地宣称跨平台逐位相同。

## 2. w1.58 权重量化：必须保留的数值语义

### 2.1 分组、阈值与有效 scale

逻辑权重为 `[N,K]`；expert bank 为 `[E,N,K]`，逐 expert、逐输出行独立处理。
每一组是同一行的连续 128 个 K 元素，不能跨输出行、跨 expert 或把分组改到 N 维。
当前候选矩阵的 K 都能被 128 整除。

对 BF16 原始权重先转 FP32，定义：

```text
beta32[g] = max(sum(abs(W32[g])) / 128, 1e-5)
theta     = 0.5
q[g,i]    = clamp(round_to_nearest_even(W32[g,i] / (2 * beta32[g] * theta)), -1, +1)
W_fake    = BF16(q * beta32)
```

这是 abs-mean 三值量化，**不是 absmax INT8、sign-only、GPTQ INT2 或通用四状态 INT2**。
`round_to_nearest_even` 的 ties-to-even 不可替换为截断或无条件 `sign`；例如精确的
归一化 `+0.5/-0.5` 会 round 到零。

本规范先按上述公式生成 `W_fake`，再恢复 codes 和 effective scale；这沿用了原 QAT
导出的数学约定，但不要求获取它的实际代码。
非零组的 effective scale 是 `W_fake` 中非零值的公共绝对值，已经经过 BF16 舍入，
不是重新保存一份未经 BF16 舍入的原始 `beta32`。全零组用 `1e-5` 作正 scale 占位。
新格式把这个 effective scale 存成 BF16。

**顺序不可反过来**：不能先将 `beta32` 转 BF16，再用它重新判定 codes，否则边界附近
的零/非零决策可能改变。官方权重首次量化产生 codes 后，在 S2–S9 复用它们；
FP32-scale 对照与 BF16-scale 候选之间只改变 metadata，不重新判定 codes。
非零组预期能无损收窄；全零组的占位 scale 可能改变，但恢复值仍为零，需分别检查。

以下是可运行的 PyTorch 数学参考，面向小张量和 CPU 接口开发，不是逐位等同 CUDA 的
生产 exporter。公开轨迹的批量导出需与这个固定环境下的参考输出逐元素核对；
这并不承诺与不可取得的私有 CUDA exporter 逐位一致。

```python
import torch


def encode_w158_reference(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return INT8 ternary codes and BF16 effective scales; packing is separate."""
    assert weight.dtype == torch.bfloat16
    assert weight.ndim >= 2 and weight.shape[-1] % 128 == 0
    assert torch.isfinite(weight).all()
    groups = weight.float().reshape(*weight.shape[:-1], -1, 128)
    beta32 = groups.abs().mean(dim=-1, keepdim=True).clamp_min(1e-5)
    codes = torch.round(groups / (2.0 * beta32 * 0.5)).clamp(-1, 1)
    folded = (codes * beta32).to(torch.bfloat16)

    # Match the exporter's "recover from BF16 fake-quant weight" convention.
    effective32 = folded.float().abs().amax(dim=-1).clamp_min(1e-5)
    recovered = torch.round(folded.float() / effective32.unsqueeze(-1)).clamp(-1, 1)
    codes_i8 = recovered.to(torch.int8).reshape_as(weight)
    scales_bf16 = effective32.to(torch.bfloat16)
    restored = (
        codes_i8.float().reshape_as(groups) * scales_bf16.float().unsqueeze(-1)
    ).to(torch.bfloat16).reshape_as(weight)
    assert torch.equal(restored, folded.reshape_as(weight))
    return codes_i8, scales_bf16
```

历史训练 Triton kernel 的关键数学代码如下，仅用于解释运算；其输出指针 dtype 是输入权重 dtype，因而这里
的 `tl.store` 对 BF16 源模型执行最终 BF16 舍入：

```text
values = tl.load(weight_ptr + offsets, mask, other=0).to(tl.float32)
scale = tl.maximum(tl.sum(tl.abs(values), 1) / GROUP_SIZE, 1e-5)
codes = libdevice.rint(values / (2.0 * scale[:, None] * fixed_threshold))
quantized = tl.maximum(tl.minimum(codes, 1.0), -1.0) * scale[:, None]
tl.store(output_ptr + offsets, quantized, mask)
```

量化后的浮点 `-0.0` 和 `+0.0` 都映射为整数 code 0。常规数值相等不意味着零的 sign bit
或整个浮点 byte hash 相等；验收应区分这两种含义，不宣称三值编码保存了 signed zero。

### 2.2 实际 packed2 布局：CPU agent 必须按此解码

布局 ID：`ternary_2bit_k16_byteplane_v1`。

- 逻辑 `[E,N,K]` codes 存为 `[E,N,K/16]` 的 INT32 bit-container；普通 Linear 可以
  使用等价的 `[N,K/16]` 逻辑表示，必须在自己的 manifest/loader 声明，不能假定
  官方模型 loader 已经支持这种自定义 packed2 权重。
- 每个 32-bit word 包含连续 16 个 K 元素。
- 两位编码：`00 -> 0`，`01 -> +1`，`11 -> -1`；**`10` 保留，遇到必须报错**。
- 第 k 个元素的 bit offset 是 `(k % 4) * 8 + (k // 4) * 2`，不是普通的 `2*k`。
  每个 byte 依次容纳 K lanes `b, b+4, b+8, b+12`，便于四路 INT8 点积解码。

下面的纯 Python 实现完整定义一个 word 的布局；INT32 存储可能为负数，但解包必须把
它当作无符号位串再移位。

```python
def pack16(codes: list[int]) -> int:
    """Pack 16 ternary lanes into the signed INT32 container used on disk."""
    if len(codes) != 16 or any(q not in (-1, 0, 1) for q in codes):
        raise ValueError("Expected exactly 16 ternary codes")
    word = 0
    for k, q in enumerate(codes):
        shift = (k % 4) * 8 + (k // 4) * 2
        word |= (q & 3) << shift
    return word if word < (1 << 31) else word - (1 << 32)


def unpack16(word: int) -> list[int]:
    """Reject the reserved code rather than silently interpreting it as -2."""
    unsigned = word & 0xFFFFFFFF
    result = []
    for k in range(16):
        shift = (k % 4) * 8 + (k // 4) * 2
        value = (unsigned >> shift) & 3
        if value == 2:
            raise ValueError("Reserved packed2 code 10")
        result.append((value ^ 2) - 2)
    return result
```

若 CPU kernel 需要另一种重排，可以在加载时转换，但要使用新的明确 layout ID，并记录
重排后的真实字节数。不能直接将这里的位串解释为其他库的 INT2 layout。

## 3. Experts 的 A128 与 MoE forward 约定

**w1.58 描述权重，不单独决定 activation dtype。** S2–S9 的 routed experts 按本 recipe
使用动态 INT8 activation group-128，即存储 W2、执行 W2A8；整数点积累加为 INT32，
跨 group 应用 scale 并累加到 FP32，层间输出为 BF16。**S1 不启用这套量化路径。**

A128 关键计算与 embedding/head 的量化不是同一套舍入细节：

```text
denom32 = max(max(abs(x32[group])), 1e-5)
inverse32 = div_rn(1.0, denom32) * 127.0
qa = clamp(nearbyint(x32 * inverse32), -128, 127)
activation_scale32 = div_rn(1.0, inverse32)
A_fake = BF16(div_rn(qa, inverse32))
```

其中 `div_rn` 表示 FP32 round-to-nearest 除法；不得随意把 `1/denom*127` 重新结合为
另一种运算序列。历史源 kernel 禁用相关浮点融合。A8 的下界是 `-128`，而本模型
embedding/head 的权重 INT8 clamp 下界是 `-127`。

W2A8 GEMM 的数学表示为：

```text
y[t,n] = sum_g (activation_scale32[t,g] * weight_scale_bf16[n,g].float()
               * sum_i_in_group(int32(qa[t,i]) * int32(qw[n,i])))
```

每组 scale 不同，不能先把所有 K 的 INT32 点积相加，最后只乘一个 scale。
CPU 内核可以融合解包、点积和 group scale，但不应持久展开整个 expert bank。

以下为不依赖 Triton、CUDA 或私有代码的 CPU 数值 reference。`qw` 是从 packed2
解出的 INT8 三值小矩阵；生产 kernel 应按 tile 解包，而不是常驻整张 INT8 bank。
代码把 group 内 INT32 点积、两次 FP32 scale 乘法和 group 顺序显式写出。

```python
def quantize_activation_reference(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU A128: return INT8 codes, FP32 multiplicative scales and BF16 fake input."""
    assert x.device.type == "cpu" and x.dtype == torch.bfloat16
    assert x.ndim == 2 and x.shape[1] % 128 == 0 and torch.isfinite(x).all()
    groups = x.float().reshape(x.shape[0], x.shape[1] // 128, 128)
    denom = groups.abs().amax(dim=-1).clamp_min(1e-5)
    inverse = torch.div(torch.ones_like(denom), denom) * 127.0
    qa = torch.round(groups * inverse.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
    scales = torch.div(torch.ones_like(inverse), inverse)
    fake = torch.div(qa.float(), inverse.unsqueeze(-1)).to(torch.bfloat16)
    return qa.reshape_as(x), scales, fake.reshape_as(x)


def ternary_int8_linear_reference(
    qa: torch.Tensor, activation_scales: torch.Tensor,
    qw: torch.Tensor, weight_scales: torch.Tensor,
) -> torch.Tensor:
    """One CPU W2A8-equivalent linear, using decoded ternary codes for clarity."""
    assert qa.device.type == qw.device.type == "cpu"
    assert qa.dtype == qw.dtype == torch.int8 and qa.ndim == qw.ndim == 2
    assert qa.shape[1] == qw.shape[1] and qa.shape[1] % 128 == 0
    assert torch.isin(qw, torch.tensor([-1, 0, 1], dtype=torch.int8)).all()
    m, k = qa.shape
    n = qw.shape[0]
    assert activation_scales.shape == (m, k // 128)
    assert weight_scales.shape == (n, k // 128)
    y = torch.zeros((m, n), dtype=torch.float32)
    for group in range(k // 128):
        sl = slice(group * 128, (group + 1) * 128)
        dot_i32 = qa[:, sl].to(torch.int32) @ qw[:, sl].to(torch.int32).t()
        scaled = dot_i32.float() * activation_scales[:, group].float().unsqueeze(1)
        y = y + scaled * weight_scales[:, group].float().unsqueeze(0)
    return y.to(torch.bfloat16)
```

### 3.1 路由概率与 SwiGLU 的顺序

```text
原始 BF16 hidden
  -> FP32 router logits -> top-8 -> 对选中 logits 做 softmax
  -> FC1 的 A128：每个原始 token 只算一次，供选中的 8 个 experts 复用
  -> 每个 expert 的 FC1(gate/up)，输出 BF16
  -> FP32(silu(gate) * up * routing_probability)，整体舍入成 BF16
  -> FC2 输入的 A128：每个 token-expert assignment 单独计算
  -> FC2 -> FP32 按 expert 顺序归并后输出 BF16；不能再次乘 routing_probability
  -> 加上独立 shared-expert 分支
```

量化 recipe 的固定要求（不用于重写 S1 的官方 forward）：

- Router 使用未被 FC1 A128 替换的 hidden；保持 `topk_then_softmax`，不能换成全专家
  softmax 后再 top-k，也不能为了量化便利改变路由。
- `probability_placement = before_fc2_activation_quant`：概率不能移到 FC2 之后。
- FC2 A128 前的 BF16 舍入即使在融合 kernel 中也必须保留。
- Shared expert 有自己的 shared gate；新增 shared 投影量化不意味着向其乘 top-8 概率。
- 对 shared/full-attention/GDN 扩展的是投影的量化器，不复制 routed branch 的整套
  SwiGLU/probability 运算到 attention 或 GDN 中。

下面补足可独立执行的 routed branch reference。实际模型 `topk=8`；小张量检查可用更小值。
Gate/up 的次序固定为 gate 在前、up 在后。Shared 分支仍由官方模型执行，再与结果相加。

```python
def routed_moe_reference(
    x: torch.Tensor, router_weight: torch.Tensor,
    fc1_codes: torch.Tensor, fc1_scales: torch.Tensor,
    fc2_codes: torch.Tensor, fc2_scales: torch.Tensor, topk: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Portable quantized routed branch, not S1's native official MoE forward."""
    assert x.device.type == "cpu" and x.dtype == torch.bfloat16
    experts = fc1_codes.shape[0]
    assert 1 <= topk <= experts and fc2_codes.shape[0] == experts
    logits = x.float() @ router_weight.float().t()
    selected_logits, selected_experts = torch.topk(logits, topk, dim=-1, sorted=True)
    probabilities = torch.softmax(selected_logits, dim=-1)
    # Once per original token, reused by every selected expert.
    qa1, as1, _ = quantize_activation_reference(x)
    output = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(experts):
        rows, slots = torch.where(selected_experts == expert)
        if rows.numel() == 0:
            continue
        fc1 = ternary_int8_linear_reference(
            qa1[rows], as1[rows], fc1_codes[expert], fc1_scales[expert])
        gate, up = fc1.float().chunk(2, dim=-1)
        weighted = (gate * torch.sigmoid(gate) * up
                    * probabilities[rows, slots].unsqueeze(-1)).to(torch.bfloat16)
        qa2, as2, _ = quantize_activation_reference(weighted)
        fc2 = ternary_int8_linear_reference(
            qa2, as2, fc2_codes[expert], fc2_scales[expert])
        output.index_add_(0, rows, fc2.float())
    return output.to(torch.bfloat16), selected_experts, probabilities
```

此 reference 定义了公开轨迹的 FP32 group/专家归并顺序，不承诺与任意 CUDA 策略相同。
Top-k ties 需记录选中 IDs；优化归并次序或分块时，用同输入/同路由单独检查误差。
官方 S1 若使用不同的路由概率应用位置或舍入，应保留其原实现；不能为了对齐本函数
把原始 BF16 baseline 也改造成量化 forward。必要时额外建立无量化的 backend 适配
对照，但不能将该对照取代 S1。

训练 fake-quant 路线会把量化后的 activation 和 weight 分别舍入成 BF16 再做 GEMM。
整数 W2A8 路线先做整数点积再应用 scales，两者一般**不逐位相同**。私有模型曾有
不同计算路径的生成分叉，因此这里更不能将公式相近当作全模型逐位相同。
公开 source 的所有路径必须重新验收，不能引用不可获得的私有测试结果作为通过证据。

## 4. Embedding 与 LM head：INT8 规则及关键代码

### 4.1 两张独立权重，都是逐行 group-128

两者均为 `[248320,2048]`，不共享权重。分组沿 hidden 维，scale shape 均为
`[248320,16]`。每个 token/词表行有 16 个独立 scale，不是一个 per-row scale，
也不是一个覆盖 128 个词表行的二维 block。

首先由官方浮点权重构建未收窄 scale 的量化表示，数学规则沿用原 recipe：

```text
amax32 = max(max(abs(W32[row, group])), 1e-5)
reciprocal32 = 127.0 / amax32
q8 = clamp(round_to_nearest_even(W32 * reciprocal32), -127, +127)
W_fake_old = BF16(q8.float() / reciprocal32)
```

新格式只把 metadata 存成 BF16：

```text
q8_new = q8_old                         # 保持 codes 不变
reciprocal_bf16 = BF16(reciprocal32)
W_fake_new = BF16(q8_new.float() / reciprocal_bf16.float())
```

**这里保存的是 reciprocal `127/amax`，不是反量化乘法 scale `amax/127`。**
不能把 `q / reciprocal` 改成 `q * reciprocal`；也不能改存 BF16 absmax 后再重新计算
reciprocal，并声称与本计划是同一种格式。全零组用 epsilon 限制后得到有限 reciprocal，
codes 仍全零。

公开轨迹使用下方函数从官方权重生成 codes，并保留同次计算的 FP32 reciprocal 作为
可选 metadata 对照。BF16 reciprocal 候选直接复制这份 codes；不要先用收窄后的
reciprocal 重新生成 codes，那将成为额外的量化实验。这里不需要任何已有私有导出。

```python
def encode_embedding_head_reference(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Math reference: generate codes with FP32 reciprocal, then narrow metadata."""
    assert weight.dtype == torch.bfloat16 and weight.ndim == 2
    assert weight.shape[1] % 128 == 0 and torch.isfinite(weight).all()
    groups = weight.float().reshape(weight.shape[0], -1, 128)
    amax32 = groups.abs().amax(dim=-1).clamp_min(1e-5)
    reciprocal32 = 127.0 / amax32
    codes = torch.round(groups * reciprocal32.unsqueeze(-1)).clamp(-127, 127)
    return codes.to(torch.int8).reshape_as(weight), reciprocal32.to(torch.bfloat16)


def restore_embedding_head_reference(
    codes: torch.Tensor, reciprocal: torch.Tensor,
) -> torch.Tensor:
    """For small tests or vocabulary tiles, NOT a full resident BF16 bank."""
    groups = codes.float().reshape(*codes.shape[:-1], codes.shape[-1] // 128, 128)
    restored = groups / reciprocal.float().unsqueeze(-1)
    return restored.to(torch.bfloat16).reshape_as(codes)


def embedding_lookup_reference(
    ids: torch.Tensor, codes: torch.Tensor, reciprocal: torch.Tensor,
) -> torch.Tensor:
    """Only decode the requested vocabulary rows; output shape is [*ids.shape,H]."""
    return restore_embedding_head_reference(codes[ids], reciprocal[ids])


def lm_head_reference(
    hidden: torch.Tensor, codes: torch.Tensor, reciprocal: torch.Tensor,
    tile_rows: int = 128,
) -> torch.Tensor:
    """BF16 operands, FP32 numerical accumulation reference, BF16 output store."""
    assert hidden.dtype == torch.bfloat16 and tile_rows > 0
    flat = hidden.reshape(-1, hidden.shape[-1])
    out = torch.empty((flat.shape[0], codes.shape[0]), dtype=torch.bfloat16,
                      device=hidden.device)
    for start in range(0, codes.shape[0], tile_rows):
        stop = min(start + tile_rows, codes.shape[0])
        weight = restore_embedding_head_reference(codes[start:stop], reciprocal[start:stop])
        out[:, start:stop] = (flat.float() @ weight.float().t()).to(torch.bfloat16)
    return out.reshape(*hidden.shape[:-1], codes.shape[0])
```

上面的 PyTorch 除法是公开轨迹的 reference，不是历史 Triton 除法的逐位替代品。
历史 kernel 使用 Triton 的 `/`，而非统一替换为 `tl.div_rn`；在 rounding boundary
上可能不同。以下摘录仅解释历史语义；公开实验不以不可获取的 CUDA 结果作门禁。
迁移 BF16 reciprocal 后必须先将其提升到 FP32 使用：

```text
# encode：旧生产路径计算和保存 FP32 reciprocal；新导出保留 codes 再 cast metadata。
scale = 127.0 / tl.maximum(tl.max(tl.abs(x), 0), 1e-5)
q = tl.extra.cuda.libdevice.rint(x * scale)
q = tl.maximum(tl.minimum(q, 127.0), -127.0)

# lookup / head restore：q 与 inverse 都以 FP32 参与除法。
q = tl.load(Codes + offsets).to(tl.float32)
inverse = tl.load(Scales + group_offsets).to(tl.float32)
weight_bf16 = (q / inverse).to(tl.bfloat16)
```

### 4.2 推理实现要求

- Embedding 是查表，仅解码请求涉及的行，输出 BF16 hidden。
- S2–S9 的 LM head 是 **W8A16 的存储/恢复语义**：输入 BF16，逐权重恢复成 BF16 后参与矩阵乘法。
  目标融合实现用 BF16 dot、FP32 accumulator，先存 BF16 输出，后续 logits processor
  的转换另算。参考代码的 FP32 matmul 累加顺序并不保证与该 kernel 逐位一致。
- 必须保留逐权重 BF16 舍入；把 scale 移到点积结果上，或将 hidden 量化成 A8，
  都是新的计算方案，不能当作无损实现替换。
- 生产推理不持久保留完整 BF16 embedding/head 副本。Head 可融合解码/GEMM或使用有界
  词表 tile；临时 tile、logits 和运行时库开销单列，不能算进静态权重 payload。
- BF16 reciprocal 可能引入新的恢复误差，需比较同次编码的 FP32 reciprocal、BF16 reciprocal
  及实际 CPU kernel 三者，不能以“codes 没变”宣称模型输出没变。

## 5. 量化位置、保护名单与完整实验计划

### 5.1 精确的模块集合

下面给出规范化的 HF text 命名，`L` 表示相应 decoder 层。官方多模态 snapshot 的
文本前缀为 `model.language_model.*`，在我们的 text-only manifest 中规范化为
`model.*`；这是**改 key 名，不改 tensor 内容或内部通道次序**。如果直接修改官方模型
对象，就保留其原 key，记录两者映射即可。不依赖任何 Megatron 转换脚本。

| 集合 | 要量化的投影 |
| --- | --- |
| Routed experts，S2–S9 共有 | `model.layers.L.mlp.experts.gate_up_proj`、`down_proj` |
| Shared expert | `model.layers.L.mlp.shared_expert.{gate_proj,up_proj,down_proj}.weight` |
| Full-attention | `model.layers.L.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight`，包括 Q 投影中的 attention gate 行 |
| GDN projections | `model.layers.L.linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}.weight` |
| Embedding，S2–S9 共有 | `model.embed_tokens.weight`，INT8 group-128 |
| LM head，S2–S9 共有 | `lm_head.weight`，INT8 group-128 |

以下代码使接手 agent 能从官方 safetensors keys 建立同样的模块集合，而无需查看本机
loader。若官方 revision 改了命名或 shape，先显式适配并审计，不能静默跳过目标。

```python
def canonical_text_key(name: str) -> str | None:
    """Map official snapshot keys; None means an explicitly excluded modality/MTP."""
    if name.startswith(("model.visual.", "mtp.")):
        return None
    prefix = "model.language_model."
    if name.startswith(prefix):
        return "model." + name[len(prefix):]
    if name == "lm_head.weight":
        return name
    raise ValueError(f"Unrecognized official snapshot key: {name}")


def quantization_role(key: str) -> str:
    """Classify canonical keys; norm/router/conv/etc. are preserved."""
    if key == "model.embed_tokens.weight":
        return "embedding"
    if key == "lm_head.weight":
        return "lm_head"
    if ".mlp.experts." in key and key.endswith(("gate_up_proj", "down_proj")):
        return "routed"
    if ".mlp.shared_expert." in key and key.endswith(
        ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
    ):
        return "shared"
    if ".self_attn." in key and key.endswith(
        ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight")
    ):
        return "attention"
    if ".linear_attn." in key and key.endswith(
        ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight",
         "in_proj_b.weight", "out_proj.weight")
    ):
        return "gdn"
    return "preserve"
```

对于本 revision，命中的 tensor 数应为 routed=80、shared=120、attention=40、
GDN projections=150、embedding=1、head=1；其余文本 tensor=301。总计 693 个
text tensors、34,660,610,688 个参数。先从 safetensors headers 审计全部 shape/dtype，
不要因为类名没有 `Linear` 就漏掉三维 expert bank。

S2–S9 保护名单：

- Router `mlp.gate.weight`：磁盘 BF16，recipe 运行时计算使用 FP32；不新增量化。
- `mlp.shared_expert_gate.weight`：BF16，不是 shared expert 的 `gate_proj`。
- GDN `A_log` 和输出 `linear_attn.norm.weight`：从官方 BF16 **无损提升**到 FP32
  存储/使用。这会额外增加 9,600 B，已计入 S2–S9 估算；不会恢复原始训练中未知的 FP32 值。
- GDN conv、`dt_bias`、其他 layer/final/Q/K norm：保持官方 BF16。
- KV cache、GDN recurrent/conv state：固定实验配置；GDN recurrent state 使用 FP32。

**官方 HF 权重已是该官方模型所需的 norm 和 Q/K/V/gate 布局。**
不执行 Megatron zero-centered gamma 的 `+1` 转换，不套用 Megatron grouped QKV
重排；只保留官方 forward 中本来存在的运算。对官方 GDN norm 做 FP32 widening
只需 `weight.float()`，不能额外加 1。S1 保持官方原始权重 dtype，不修改文件。

### 5.2 主矩阵

S1 的 routed experts、embedding、head 和主投影全部保持官方 BF16。
**只有 S2–S9** 共同使用 w1.58 routed experts、INT8 embedding/head 和 BF16 weight
scales；未列的小参数保持上述保护名单精度。所有行都来自同一个官方 source。

| Setting | Shared expert 投影 | Full-attention 投影 | GDN 投影 | 权重总字节 | GiB |
| --- | --- | --- | --- | ---: | ---: |
| S1：官方原始 BF16 baseline | BF16 | BF16 | BF16 | 69,321,221,376 | 64.5604 |
| S2：当前位置，BF16 scales | BF16 | BF16 | BF16 | 12,451,875,456 | 11.5967 |
| S3：S2 + shared | w1.58 | BF16 | BF16 | 12,233,640,576 | 11.3935 |
| S4：S2 + full-attention | BF16 | w1.58 | BF16 | 11,979,033,216 | 11.1563 |
| S5：S2 + GDN | BF16 | BF16 | w1.58 | 10,699,176,576 | 9.9644 |
| S6：S2 + full-attention + GDN | BF16 | w1.58 | w1.58 | 10,226,334,336 | 9.5240 |
| S7：S6 + shared | w1.58 | w1.58 | w1.58 | 10,008,099,456 | 9.3208 |
| S8：S2 其余大 Linear 变 FP8 block | FP8-B128 | FP8-B128 | FP8-B128 | 11,043,024,896 | 10.2846 |
| S9：S2 其余大 Linear 变 INT8 block | INT8-B128 | INT8-B128 | INT8-B128 | 11,043,024,896 | 10.2846 |

S3/S4/S5 均独立基于 S2，不是逐行累加。主路线为 **S1 → S2 → S6 → S7**，
S3/S4/S5 用于归因；S8/S9 也必须各自报告相对 S1 的质量、吞吐和内存变化。
S7 相比新 S2 节省 2.275943756 GiB，约 19.63%。

基准/辅助组：

- 不再另设 R0，官方原版基准统一命名 **S1**。
- **S2-scale32**：从同一官方权重生成的 FP32-scale 对照，12,971,084,416 B =
  12.0803 GiB。与 S2 共享 codes，只保留收窄前 metadata；自己按本文生成，
  **不需要下载私有旧导出**，也不使用其输出作为 golden。
- **C8**：新 S2 的同一组三值 codes 用 INT8 容器存储，scales 仍 BF16，
  36,611,066,496 B = 34.0967 GiB。与 S2 相比只改变包装，不增加权重精度；
  CPU 可用此组分离 packed2 解包成本与内存带宽收益。

### 5.3 Activation 消融不能被 setting 名字隐藏

矩阵的 ID 描述**权重量化位置**，每个运行记录另带 `activation_profile`：

1. **S1 使用 `official_bf16`**：官方原生 forward，不启用 A128、weight fake quant
   或量化 recipe 的概率前置；baseline 实现的 FP32 norm/softmax 等仍按官方规则执行。
2. S2–S9 第一轮 `new_linear_a16`：routed experts 使用本文 A128；新增 w1.58/FP8/INT8
   投影先使用 BF16 输入，解码权重进行数值对照，隔离新增权重量化误差。
3. S3–S7 再增加 `new_w158_linear_a128`：新增 w1.58 投影也采用专家的动态 A128，
   形成完整 W2A8 路径；只量化其输入，不改变该模块后续 attention/GDN 运算结构。
4. S8/S9 新增投影第一轮仍是 weight-only A16。以后增加 FP8 activation 或 INT8 activation
   时使用独立 profile 和验收结果，不与前面的运行配置混为一组。

S1→S2 同时引入专家权重/activation 量化、embedding/head INT8 和量化 recipe 的
数值路径；它是完整方案对原版的比较，不能把所有质量或速度变化都归因于 W2 包装。

这几个 profile 的静态权重空间相同，但误差、临时内存和吞吐未必相同。
BF16 operands reference、整数优化路径、原生训练 golden 必须分别命名。

## 6. S8/S9 的 block-scale 候选规范

两组共用 `[128,128]` 的 N×K 二维 block 分组，与 w1.58 的 `[1,128]` 不同。
每个逻辑投影独立分组；例如 `in_proj_a` 和 `in_proj_b` 不先拼接成一个新 block。

- S8 weight codes：FP8 `E4M3FN`，最大有限绝对值 448。
- S9 weight codes：对称 INT8，范围 `[-127,127]`，无 zero-point。
- Scale：每 block 一个 BF16 **乘法反量化 scale**，`W_restore = BF16(code.float()*scale.float())`。
  不复用 embedding/head 的 reciprocal 字段语义。
- 拟定编码次序：FP32 absmax 求 scale → BF16 保存 → 用保存值的 FP32 提升值生成 codes。
  对新 block recipe 这样明确约定，避免实现者各自选择不同的 scale 舍入时机；
  S2 的 experts/embedding/head 则执行前文“先生成 codes，metadata 收窄时不重算 codes”的规则。
- 全零 block 取 scale=1，codes 全零。对非有限输入或收窄后非正/非有限 scale 报错。
- 尾 block 只统计实际元素；scale 数量按 ceil 计算。若 CPU/GPU kernel 对 code 做实际
  padding 或进一步重排，额外空间另记。

```python
def encode_block8_reference(
    weight: torch.Tensor, kind: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Proposed S8/S9 recipe; not the current vLLM checkpoint schema."""
    assert weight.dtype == torch.bfloat16 and weight.ndim == 2
    assert torch.isfinite(weight).all() and kind in ("fp8", "int8")
    n, k = weight.shape
    dtype = torch.float8_e4m3fn if kind == "fp8" else torch.int8
    limit = 448.0 if kind == "fp8" else 127.0
    codes = torch.empty_like(weight, dtype=dtype)
    scales = torch.empty(((n + 127) // 128, (k + 127) // 128),
                         dtype=torch.bfloat16, device=weight.device)
    for row in range(0, n, 128):
        for col in range(0, k, 128):
            tile = weight[row:row + 128, col:col + 128].float()
            amax = tile.abs().amax()
            scale32 = torch.where(amax > 0, amax / limit, torch.ones_like(amax))
            scale = scale32.to(torch.bfloat16)
            if not bool(torch.isfinite(scale)) or not bool(scale > 0):
                raise ValueError("Invalid BF16 block scale")
            normalized = (tile / scale.float()).clamp(-limit, limit)
            q = normalized.to(dtype) if kind == "fp8" else normalized.round().to(dtype)
            codes[row:row + 128, col:col + 128] = q
            scales[row // 128, col // 128] = scale
    return codes, scales
```

这是借鉴 FP8 的分块 scale 思路，**不声称等同或直接兼容某个 NVIDIA ModelOpt 格式、
MXFP8/E8M0 scale 或现有 FP8 kernel ABI**。BF16 scale 与本模型的 w1.58/INT8 混合
loader 都需要显式适配。FP8 在目标 CPU 是否能高效执行，也需要实测，不能从 GPU
支持推断。

S8↔S9 才是同位置、同分组、同 scale dtype 的 FP8/INT8 格式对照。
S7↔S8/S9 同时改变位宽和分组，不能把质量差异单独解释为 FP8 与三值的差异。

可选后续 `S8-full/S9-full`：在上述 block 格式上进一步量化 router 和 shared gate
的权重，router 算术仍保持 FP32，预计均为 11,021,975,296 B = 10.2650 GiB。
只再省约 20.07 MiB，却可能改变 top-8，因此不纳入首轮主矩阵；必须单独审计路由。

## 7. 空间账本：可复算，不混入运行时内存

计入 text-only 推理权重及量化 metadata；不含 vision、MTP、训练状态、tokenizer、
safetensors header、KV/SSM cache、RoPE buffer、激活、logits、工作区或 allocator。
单位 `GiB = 2**30 bytes`。

对于可整除 128 的 P 个权重：

```text
BF16                       = 2*P bytes
w1.58 packed2 + BF16 scale  = P/4 + 2*(P/128) bytes
INT8 group-128 + BF16 scale = P   + 2*(P/128) bytes
FP8/INT8 block-128          = sum_matrices(N*K + 2*ceil(N/128)*ceil(K/128)) bytes
```

### 7.1 新 S2 的模块账本

| 模块 | 参数量 | 存储类型 | 大小 GiB |
| --- | ---: | --- | ---: |
| Routed FC1 | 21,474,836,480 | w1.58 + BF16 scale | 5.312500000 |
| Routed FC2 | 10,737,418,240 | w1.58 + BF16 scale | 2.656250000 |
| Embedding | 508,559,360 | INT8 + BF16 reciprocal scale | 0.481033325 |
| LM head | 508,559,360 | INT8 + BF16 reciprocal scale | 0.481033325 |
| Shared expert projections | 125,829,120 | BF16 | 0.234375000 |
| Full-attention projections | 272,629,760 | BF16 | 0.507812500 |
| GDN projections | 1,010,565,120 | BF16 | 1.882324219 |
| Router/shared gate/norm/GDN 小参数 | 22,213,248 | 保护名单规定的 BF16/FP32 混合 | 0.041384339 |
| **总计** | **34,660,610,688** | | **11.596712708** |

40 层共 256 experts/层，routed FC1 为 `[256,1024,2048]`，FC2 为
`[256,2048,512]`。即使每 token 仅选 top-8，仍需保存全部专家，不能只计 8 个。

| 新增量化模块 | BF16 大小 GiB | w1.58 + BF16 scale 大小 GiB | 节省 GiB |
| --- | ---: | ---: | ---: |
| Shared expert projections | 0.234375000 | 0.031127930 | 0.203247070 |
| Full-attention projections | 0.507812500 | 0.067443848 | 0.440368652 |
| GDN projections | 1.882324219 | 0.249996185 | 1.632328033 |

S2 比同源 S2-scale32 缩小 519,208,960 B = 495.15625 MiB，来自 expert scales 和
embedding/head reciprocal scales 减半。这些是指定格式的张量字节计算，不是新格式
加载/推理后的实测峰值。

Router 从文件 BF16 提升为运行时 FP32 会额外常驻 40 MiB。若保留全部 FP32 scale
副本、完整 BF16/INT8 展开权重或格式转换副本，必须把它们加回 RSS 账本。
S1 原始文件的 GDN 小参数也是 BF16；S2–S9 的 GDN FP32 widening 已额外计入
9,600 B。因此这里的 bytes 是精确格式估算，不是拿总参数量粗略乘以一个平均位宽。
若 backend 额外分配 RoPE 等 buffers 或加载了 vision/MTP，也必须单列。

## 8. 后续 agent 的执行步骤与验收

### 阶段 A0：先建立官方 S1 baseline

1. 下载并固定官方 BF16 revision，保存 config/tokenizer/shard hash，冻结公开 backend
   版本。用第 5.1 节规则审计 text tensor 数、参数量、dtype 和量化位置。
2. 按官方模型代码运行 text-only S1，保持官方权重/forward、不插入任何 fake quant。
   保存固定输入的 logits、任务分数、prefill/decode、B1/并发延迟和 RSS。
3. S1 与量化候选采用同一 CPU、线程/NUMA、输入/输出长度、tokenizer、chat template、
   cache、编译/预热和采样设置；记录实际 backend/ISA。不能只优化候选却用未经预热的
   baseline 作速度对比，也不能用 BF16 fake-quant 展开权重替代官方 S1。
4. 如果必须更换模型执行 backend，先用同一官方 BF16 权重验证其与原生参考的输出；
   无量化适配对照单列为 `S1-backend-check`，不覆盖 S1 的定义。
5. 若 S1 在目标机器上 OOM 或发生 paging，明确记录相应配置下无可比吞吐/内存峰值；
   不以 CPU offload、磁盘抖动或另一台机器的时间冒充同条件 baseline。

### 阶段 A：从官方权重构建 S2 与 metadata 对照

1. 实现一个只依赖公开 PyTorch/safetensors 的逐 tensor/逐行块 exporter。按本文的
   canonical key/role 识别模块：专家调用 `encode_w158_reference`，embedding/head
   调用 `encode_embedding_head_reference`；用第 2.2 节 bit layout 打包专家 codes。
   分块读取和写出，避免同时加载官方 BF16、INT8 展开和 packed2 全模型副本。
2. 未选中模块原值保留；仅对保护名单要求的 GDN 两类参数做无损 FP32 widening。
   无 DCP、无 optimizer、无 zero-centered gamma 转换。配置和 tokenizer 来自同一个
   官方 snapshot，不依赖私有目录的补权重或资产。
3. 生成显式 manifest：logical shape、tensor keys、codes/scale dtype、分组、scale
   方向和 packing ID。官方 `from_pretrained` 不会自动理解这种混合自定义格式，
   接手 agent 需实现 loader/量化模块，不是设置某个不存在的 CLI 开关。
4. 先用同一官方权重生成 `S2-scale32` 的 codes 和 FP32 metadata，再复用 codes 得到
   BF16-scale S2。逐 tensor 检查 dtype、bytes、scale round-trip 和恢复值差异。
   全零组占位 scale 与非零有效 scale 分开报告。
5. 公开 reference 由本文纯 CPU 函数和官方模型计算组成。可以先在小模型/分层测试
   中恢复 BF16 权重验证，或临时在完整官方架构上做 reference，但这样的展开版本不能
   按 packed2 的常驻内存报告性能。完整 S2 A128 路线需按第 3 节接入 routed branch，
   不能仅替换 BF16 weight 就声称运行了 W2A8。
6. S1、S2-scale32、S2 使用同一组 teacher-forced 输入，分别记录完整方案变化和仅
   metadata 收窄带来的差异。不依赖不可获得的私有输出/golden。

### 阶段 B：新增量化位置的 PTQ 筛选

1. 从与 S1/S2 相同的官方 BF16 source 构建 S3/S4/S5；复用本文 w1.58 算法，
   不换成其他 INT2 算法，相同模块在不同 setting 共享同一份 codes/scales。
2. 主路线执行 S1 → S2 → S6 → S7，记录单模块与组合影响；组合必须单测，不能相加推测。
3. 新增投影先运行 `new_linear_a16`，再运行 `new_w158_linear_a128`。
4. 从 S2 的 BF16 投影独立生成 S8/S9，固定同一 block/scale 规则。
5. 首轮不改保护名单、路由概率位置、SwiGLU/norm 精度或 KV/SSM 精度。

**公开轨迹没有三值 QAT，包括 S2 的 routed experts 也未经过本 recipe 的 QAT。**
直接 PTQ 只是候选筛选，尤其 GDN 的 `a/b` 控制投影可能对递归动态敏感。严重退化应
作为结果如实报告，不能换入未交付的私有权重、放宽门禁或只展示几条成功输出。
若需要 QAT 修复，另建 `source_track=official_qat`，固定公开数据/seed/步数预算及训练
实现，并从同一官方初始化公平比较；它不等于复现私有训练，也不隐含已获训练数据。

### 阶段 C：正确性与下游质量

- **格式层**：全量 codes/scale 校验，reserved packed2 code、signed INT32、尾 block、
  全零/极小值/舍入边界、非有限输入、实际 dtype/字节数。
- **算子层**：同一输入比较恢复 reference 与新 kernel，分别测 FC1/FC2、embedding、
  head、shared、attention/GDN 投影；固定路由比较 expert 输出，另报告自然路由变化。
- **模型层**：固定 token IDs 做 teacher-forced logits/NLL，记录 KL、top-1 一致率及
  中间层误差；greedy 输出分叉不能单独说明哪一层有 bug，也不能被高 cosine 掩盖。
  各 S2–S9 既报告相对 S1 的完整质量变化，也报告相对 S2 的新增量化位置影响。
- **任务层**：冻结数学/代码/中文任务样本、chat template、thinking 设置及解码参数，
  报告实际得分、样本数、差值和不确定性；校准/调参与最终评测数据分开。
- **长度层**：至少覆盖 1K/8K/32K，候选通过后扩至 64K/128K；长输入与输出长度分别
  记录。GDN 相关实验必须有长上下文检查，不能只测短文本首 token。
- **验收阈值**：在看结果前固定。量化变更允许的任务质量损失与 kernel 对同一量化
  reference 的数值容差是两类门禁，不能事后放宽门禁来制造“通过”。

公开轨迹的完整模型 reference 是官方 S1 以及本文定义的量化 reference，不需要
Megatron 环境。将独立片段、单算子、官方整模型和新量化模型验证分开报告。
私有原生训练对齐在只有公开权重时是**不可验证项**，不能将它写成必须从本机取得的
前置任务，也不能用公开 PTQ 通过来冒称私有 QAT 对齐。

### 阶段 D：交给 CPU agent 的吞吐与内存实验

- 每个候选提供精确 module allowlist、原始/量化 shape、codes dtype、scale dtype、
  scale 方向、block shape、packing layout、activation profile、source/code hash。
  在 schema 中分别声明专家乘法 scale 与 embedding/head reciprocal，不能只留一个
  含糊的 `scale_dtype=bf16` 字段。
- 接手者先用官方模型和本文代码在自己的环境生成 reference logits/关键层输出及
  质量报告，供同机 kernel 优化比较。不要要求另一个看不到本机的 agent 提供私有 golden。
- 固定 CPU 型号/ISA、线程数、绑核、NUMA、内存配置、缓存预算和后端版本；同一候选
  分开记录冷加载、warm prefill、B1 decode、并发 8/32 的延迟及吞吐，重复至少 3 次。
- 记录输入/输出 tokens、TTFT、decode tokens/s、总吞吐、稳态和峰值 RSS；mmap 的
  文件页、私有展开副本和 workspace 分开说明，不把磁盘大小当成峰值 RAM。
- 主结果表必须包含 **S1–S9**，每组列出相对 S1 的速度比、内存比和任务分数差；
  `speedup = throughput_setting / throughput_S1`，延迟比则用
  `latency_S1 / latency_setting`，注明指标方向，不能只列量化组之间的收益。
- 对专家重点用 S2↔C8 判断 W2 解包是否兑现带宽收益。不得把全部 W2 权重持久展开为
  INT8/BF16 后仍按压缩权重大小报告常驻内存。
- CPU 优化允许局部解包、SIMD/VNNI/AMX（按实际 ISA）点积、tiling 和融合，但必须复用
  此处规定的 group scale/舍入语义；不照搬 CUDA `dp4a`/Tensor Core 的速度结论。
- FP8/INT8 两者静态空间相同，不意味着 CPU 吞吐相同。若采用 BF16/FP32 fallback，
  在报告中明确说明实际计算路径及临时解码成本。

建议统一结果字段（交付格式要求，不是现有 CLI 参数）：

```text
setting_id, source_track, source_repo, source_revision, source_file_hashes,
activation_profile, quant_spec_version, exporter_sha256,
module_list, weight_format, weight_group_shape, weight_scale_semantics,
weight_scale_dtype, packed_layout, parameter_bytes, resident_weight_bytes,
reference_backend, validation_metrics, task_scores,
cpu_model, isa, threads, affinity, numa_policy, prompt_tokens, output_tokens,
batch_or_concurrency, prefill_tokens_per_s, decode_tokens_per_s,
ttft_ms, peak_rss_bytes, repeats, throughput_ratio_vs_S1,
memory_ratio_vs_S1, task_score_delta_vs_S1, notes
```

## 9. 独立交付协议、内置自检与验证边界

### 9.1 Manifest 的最小语义

接手 agent 自己生成模型文件和 manifest。以下是协议示例，不是宣称某个官方 loader
已经支持的 schema；实际实现还需列出全部 tensor、存储文件、hash、shape 和 dtype。
无需复制任何本机文件名或读取私有源码。

```json
{
  "quant_spec_version": "public_hf_w158_v2",
  "source_track": "official_ptq",
  "source_repo": "Qwen/Qwen3.6-35B-A3B",
  "source_revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
  "setting_id": "S2",
  "activation_profile": "new_linear_a16",
  "scope": "text_only_no_vision_no_mtp",
  "tensor_examples": [
    {
      "key": "model.layers.0.mlp.experts.gate_up_proj",
      "logical_shape": [256, 1024, 2048],
      "weight_format": "w1.58_packed2",
      "code_dtype": "int32",
      "stored_shape": [256, 1024, 128],
      "group_shape": [1, 128],
      "scale_dtype": "bfloat16",
      "scale_shape": [256, 1024, 16],
      "scale_semantics": "multiply_effective_bf16",
      "layout": "ternary_2bit_k16_byteplane_v1"
    },
    {
      "key": "model.embed_tokens.weight",
      "logical_shape": [248320, 2048],
      "weight_format": "symmetric_int8",
      "code_dtype": "int8",
      "group_shape": [1, 128],
      "scale_dtype": "bfloat16",
      "scale_shape": [248320, 16],
      "scale_semantics": "divide_reciprocal_127_over_absmax",
      "round_weight_to": "bfloat16"
    }
  ]
}
```

LM head 使用与 embedding 相同的量化字段，但 key 为 `lm_head.weight`，权重独立；
计算记录声明 `W8A16`、输出存储 dtype 和实际 logits dtype。S1 使用自己的原始官方
manifest/配置，不应被标记成 `w1.58_packed2` 或开启量化 wrapper。

### 9.2 无模型权重、无私有代码的内置自检

按顺序执行本文全部 Python 函数定义，然后调用 `reference_selftest()`。这只需要
CPU PyTorch，使用人造小张量，无下载、GPU、私有代码或私有 golden。

```python
def reference_selftest() -> dict[str, str]:
    """Small deterministic contract checks; not model quality or throughput tests."""
    assert pack16([0] * 16) == 0
    assert pack16([1] * 16) == 0x55555555
    assert pack16([-1] * 16) == -1
    assert pack16([0, 1] + [0] * 14) == 256
    assert unpack16(pack16([-1, 0, 1, 0] * 4)) == [-1, 0, 1, 0] * 4
    try:
        unpack16(2)
    except ValueError:
        pass
    else:
        raise AssertionError("Reserved code accepted")

    weight = torch.tensor([0.5, -0.5, 1.5, -1.5] * 32,
                          dtype=torch.bfloat16).reshape(1, 128)
    qw, sw = encode_w158_reference(weight)
    assert qw[0, :4].tolist() == [0, 0, 1, -1] and sw.item() == 1.0
    eh_q, eh_r = encode_embedding_head_reference(torch.ones((3, 128), dtype=torch.bfloat16))
    assert (eh_q == 127).all() and (eh_r == 127).all()
    ids = torch.tensor([2, 0, 2])
    assert (embedding_lookup_reference(ids, eh_q, eh_r) == 1).all()
    assert (lm_head_reference(torch.ones((1, 128), dtype=torch.bfloat16),
                              eh_q, eh_r, tile_rows=2) == 128).all()

    x = torch.ones((1, 128), dtype=torch.bfloat16)
    qa, sa, fake = quantize_activation_reference(x)
    assert (qa == 127).all() and torch.equal(fake, x)
    out = ternary_int8_linear_reference(
        qa, sa, torch.ones((1, 128), dtype=torch.int8),
        torch.ones((1, 1), dtype=torch.bfloat16))
    assert out.item() == 128.0
    moe_out, selected, probs = routed_moe_reference(
        x, torch.zeros((2, 128), dtype=torch.bfloat16),
        torch.ones((2, 256, 128), dtype=torch.int8),
        torch.ones((2, 256, 1), dtype=torch.bfloat16),
        torch.ones((2, 128, 128), dtype=torch.int8),
        torch.ones((2, 128, 1), dtype=torch.bfloat16), topk=2)
    assert selected.shape == (1, 2) and (probs == 0.5).all()
    assert (moe_out == 2097152.0).all()
    for kind in ("int8", "fp8"):
        qb, sb = encode_block8_reference(torch.zeros((32, 256), dtype=torch.bfloat16), kind)
        assert qb.shape == (32, 256) and sb.shape == (1, 2)
        assert (qb.float() == 0).all() and (sb == 1).all()
    assert canonical_text_key("model.language_model.embed_tokens.weight") == "model.embed_tokens.weight"
    assert canonical_text_key("model.visual.patch_embed.proj.weight") is None
    assert quantization_role("model.layers.0.mlp.shared_expert_gate.weight") == "preserve"
    assert quantization_role("model.layers.0.mlp.shared_expert.gate_proj.weight") == "shared"
    return {"packing": "passed", "weight_and_head": "passed",
            "a128_and_moe": "passed", "block8_and_keys": "passed"}
```

批量 exporter 还要增加坏输入、尾块、不同 shape、非法 code、极值和 round-trip 全量
校验；上述 smoke checks 不覆盖完整模型所有数值边界。正式 benchmark 不运行这种
Python 逐 expert/逐 group 的 reference 当作优化 kernel。

### 9.3 本文件的验证边界与持续跟踪

本文件是自包含 recipe 与执行计划，**不是完整 exporter、官方模型 loader 或高性能
CPU 后端**。接手者可依据官方架构和本文的全部量化定义自行实现，不能假设存在文档
之外的私有模块。此次只修改这个 Markdown，不新建脚本/报告、不转换权重、不启动
训练或整模型测试。后续计划修正和实验摘要继续维护在同一文件中。

v2 已重新执行以下检查（2026-09-17）：

- 从本 Markdown 在内存提取并执行全部 **8 段 Python**，仅 import 公开 PyTorch，
  没有私有 import、模型文件依赖或 GPU 调用；`reference_selftest()` 全部通过。
- 额外检查 packed2 的 4,147 组往返、16 个 reserved-code lane；w1.58 的 shape、
  全零/极小值和舍入；embedding 的重复/标量/空 IDs；FP8/INT8 的尾 block；以及
  两个 K group 使用不同 scales 时的 INT32 点积与 FP32 反量化。
- 使用已下载的官方 snapshot headers 核对 693 个 text tensors、34,660,610,688 个
  参数及第 5.1 节分类。由官方 BF16 payload 加明确的 dtype 转换重新计算，S1–S9
  九组 bytes/GiB 均与主矩阵一致；这不依赖私有量化导出。
- JSON manifest 示例可解析；文档不再以任何私有路径/源文件/golden 为运行前置条件；
  Markdown 格式检查通过，1 个文件、0 个错误。
- 本次小张量检查环境为 Python 3.12.3、PyTorch 2.13.0+cu130，计算全部在 CPU；
  CUDA build 后缀不是运行这些 reference 的要求。接手者仍需记录自己的版本并重跑自检。

这些是规范与参考代码验证，**没有运行官方 S1 整模型或 S2–S9 的质量/吞吐实验**。
未来的代码、PyTorch/backend 或 source revision 改动均须重新验收。

| 项目 | 状态 | 需要留下的证据 |
| --- | --- | --- |
| 本文 CPU reference、内置自检与语法检查 | v2 已通过 | 仅独立小张量检查，不冒充整模型验证 |
| 官方 headers 与 S1–S9 字节账本 | v2 已通过 | 官方 tensor 分类、数量、参数量、dtype |
| 单 Markdown 自包含性与格式 | v2 已检查 | 无私有 import/路径前置依赖、格式检查 |
| 官方 S1 BF16 质量/吞吐/RSS baseline | 待执行 | 官方 source、backend、配置、完整对照数据 |
| 公开 S2 与 S2-scale32 导出/loader | 待实施 | dtype、codes、恢复值、metadata 差异 |
| S3/S4/S5 单模块量化 | 待实施 | 每组实际 module 清单、质量与误差 |
| S6/S7 组合量化 | 待实施 | 组合效果、长上下文结果 |
| S8/S9 block FP8/INT8 | 待实施 | 同分组对照及实际格式支持 |
| 新量化投影 A128 路径 | 待实施 | 与 A16/整数 reference 分开的验收 |
| CPU S1–S9 吞吐/RSS 与 C8 对照 | 待交接 | 硬件、线程、NUMA、相对 S1 的重复测量 |
| 可选 full 版本与公开数据 QAT | 后置候选 | router 变化或等预算、可复现的训练实验 |

## 10. 2026-09-20 代码实施进度（真实权重尚未齐全）

**本节记录工具开发与人工小模型验证，不表示官方 Qwen3.6-35B-A3B 已量化或已跑分。**
第 9.3 节“此次只修改 Markdown”是此前规范整理的历史边界；本次经用户批准开始实现代码。
用户选择 **先测量、不判质量通过**，因此不自行设定允许掉分幅度，不把任何候选标成质量通过。

### 10.1 已实现的代码

实现位于 `BitNet` 与其 `3rdparty/llama.cpp` 的 `dev-yoco-u-moe-bitnet` 分支，尚未提交：

- `BitNet/utils/qwen36_w158/`：规范/CPU reference、source headers 审计、流式 canonical exporter、
  manifest、GGUF adapter、官方 S1/候选 reference、固定 token 的 NLL/KL/top-1 测量入口。
- llama.cpp 的 `src/llama-qwen36-w158.{h,cpp}`、recipe 专用 raw tensor loader 与
  `src/models/qwen35moe.cpp` 的 opt-in graph；不重定义现有 I2_S，也不修改未标记模型的路径。
- `build_script/build_qwen36_w158_cpu.sh`：独立 CPU 构建，不强开本机不支持的 BF16/AMX 指令。
- `benchmark_script/benchmark_qwen36_w158_cpu.sh`：native 单序列测量入口；不是官方 S1 对照表，
  尚不包含并发服务、TTFT、冷文件缓存或任务质量测量。
- 详细命令和限制见 `BitNet/utils/qwen36_w158/README.md`。

canonical 产物以内容寻址共享相同 codes/scales，并拒绝混合 source、导出中 source 改变、损坏缓存和
冲突覆盖。native GGUF 当前支持普通分片，但**分片仍复制 payload，未实现跨 setting 的文件去重**。
CPU kernel 当前是有界解包的标量正确性实现，尚无 SIMD 加速或真实吞吐收益结论。

### 10.2 已取得的验证证据

- Python 3.12.3、PyTorch 2.9.1+cu128、Transformers 5.16.1、safetensors 0.8.0；实际计算在 CPU。
  这不是第 9.3 节历史环境，新的产物记录自己的代码和环境 fingerprint。
- 启用真实 native 模型 harness 后，**127 项 Python 测试通过**，包括格式、source 审计、流式导出、
  源文件变化/并发发布、HF 小模型和 GGUF 加载测试；专用 C++ operator CTest 通过。
- BF16 CPY 回归 53/53 通过；普通 Qwen35MoE architecture/roundtrip 测试通过。
- **通用 `test-gguf 158` 仅 69/87 通过，18 项失败，原因未定。** 未做基线构建对比，不能声称是
  已证实的历史问题，也不能宣称全部回归通过。
- 人工 tiny hybrid 模型执行了 S1-backend-check、S2–S9、S2-scale32/C8，以及 S7-A128；固定
  token `[1,5,8,3]` 的 native chunk-1/chunk-4 一致，S2/C8 一致，非法 codes/scales/metadata 被拒绝。
- 24 组 native/HF 对比均仅覆盖这 4 个 token；top-1 均为 4/4，但 logits **不逐位相同**，最大绝对
  误差为 0.00390625–0.0078125。没有事前设定整模型数值门禁，因此只记录为 ungated 测量，
  不代表数值等价或质量通过。native harness 的 KV 为 FP32，HF 为 BF16；差异尚未完全定位。

证据根目录为 `/data3/huangxin/model_list/qwen36-w158-public-v2/`：source 审计在
`reports/source-audit.json`；人工模型与对比在 `fixtures/native-tiny-2p4rdq2t/`；native/HF
详细数据在其 `reference/native-comparison.json`；各项 C++ 回归日志在 `build-cpu/`。

### 10.3 尚未执行的真实模型任务

最近 source 审计仍只有 **21/26 个完整分片**，缺分片 22–26、safetensors index 和 tokenizer 文件。
下载 metadata 的 revision 匹配本规范，但 source 状态仍是 `incomplete`；完整导出与 S1 loader
会拒绝该目录。本次没有停止下载，也没有删除或修改下载缓存。

所以第 9.3 节状态表中的**官方 S1 质量/吞吐/RSS、真实 S2–S9 导出与评测、长上下文和 C8
性能对照仍未完成**。还需冻结任务样本/评测配置、定位或明确约束 backend 数值差异、处理上述
回归失败，再在 source 完整后逐阶段执行。64K/128K 不自动启动，不能以人工模型测试替代 35B 实验。

## 11. 2026-09-20 真实权重测试续接

本节更新第 10 节的历史状态。仅继续本文指定的真实模型实验，不重跑通用回归或启动 64K/128K。
继续采用 **先测量、不判质量通过**；短输入就绪性运行不能替代任务分数、长上下文或完整性能矩阵。

### 11.1 已固定的来源、环境与证据目录

- 官方 26/26 个权重文件已全量读取并计算 SHA256；与 Hugging Face 官方 API 返回的固定
  revision `995ad96eacd98c81ed38be0c5b274b04031597b0` 的 26 个 LFS SHA256、大小全部匹配。
  文件总大小 71,903,776,776 B；校验时未发现文件变化。
- 完整 source audit 已通过：693 个 text tensors、34,660,610,688 个参数，S1 text payload
  69,321,221,376 B；排除 vision/MTP，没有修改官方源文件或下载缓存。
- 真实官方 index 的 `metadata.total_size` 为 `71903645408.0`。旧审计器只接受 JSON 整数，
  因而首次拒绝完整官方源。现仅在内存中接受小于 `2**53` 的非负、整数值 float 并转为 int，
  保留 bool/负数/小数/非有限值拒绝及 header 字节总量核对；不重写 index。
- CPU：Intel Xeon Platinum 8358P；正式同配置运行固定 8 线程、CPU 0–7、NUMA 0。
  Python 3.12.3、PyTorch 2.9.1+cu128、Transformers 5.16.1、safetensors 0.8.0；GPU 禁用。
  本机不具备 AVX512_BF16；编译脚本关闭其显式开关。已有 `build-cpu` 的 `GGML_NATIVE=ON`
  构建会按本机 ISA 自动选择指令，可原地复用，不把 `NATIVE` 模式下单独的 AVX 开关当实际 ISA。
- 本轮独立证据目录：
  `/data3/huangxin/model_list/qwen36-w158-public-v2/results/real-20260920-5BJ71rCj/`。
  `run-config.json` 固定环境、官方 tokenizer/chat template 与 `enable_thinking=false`；
  `tokens-teacher-forced.json` 为固定 60-token 会话，首阶段用其前 16 个 token 检查真实执行入口。
  这组输入不是数学/代码/中文任务集，也不用于宣称任务得分。
- `source-weight-sha256.txt`、`source-audit.json`、`code-provenance.json` 分别保存权重校验、
  完整源审计及实际脚本/代码/二进制身份。仓库仍有未提交代码，因此 HEAD 不是完整代码身份。

### 11.2 官方 S1 首轮真实执行结果

通过现有 `qwen36_w158.evaluate --source ... --token-chunk 16 --vocab-chunk 128 --threads 8
--generate-tokens 4`，固定 CPU/NUMA 执行，退出码 0。

- 固定输入 16 tokens、15 个 next-token 预测位置：NLL `7.63551223387415`，next-token top-1
  accuracy `0.13333333333333333`。这里不是候选与 S1 的 top-1 一致率，也不是任务准确率。
- NLL 路径为官方 cached decoder 加 vocabulary-tiled BF16 head；分块可能改变底层 reduction，
  明确标记 `official_cached_decoder_plus_vocabulary_tiled_head_FP64_metric_reduction`。
- 额外经官方完整 CausalLM forward 生成 4 个 token，IDs 为 `[248046, 198, 248045, 74455]`。
  输入只是就绪性前缀，不据此判断回答质量；本轮没有保存完整 forward logits。
- 整条 CLI 耗时 `83.449007` 秒，进程峰值 RSS `21,582,954,496 B`。该时间包括 audit/hash/load、
  NLL 和 greedy，**不是 warm prefill/decode 吞吐、TTFT 或三次重复性能数据**。
  短输入按需触及的文件映射页不等于全部 BF16 参数常驻；不能以该 RSS 推断 S1 权重被压缩。
- 结果：`S1-readiness.json`、`S1-readiness.runtime.json`、`S1-readiness-attempt2.log`。
  首次外层采集因系统没有 `/usr/bin/time` 未启动模型；随后改用标准库 subprocess/resource，
  不安装依赖。后续 `run_cli.py` 额外按秒记录 RSS/file/anon/swap，应用内部时间另行读取。

### 11.3 进行中的工作与仍需补齐的边界

- 已启动真实 S1–S9、S2-scale32、C8 的同源 canonical 导出，输出为本轮目录的 `canonical/`。
  导出尚未完成时，不能仅凭缓存对象数量声称候选或全量格式校验通过；以最终 manifest 和退出码为准。
- 当前原生 logits harness 仍固定 2 线程、F32 attention KV、最多 256 IDs；现有 benchmark 可用
  8 线程和 BF16 attention KV。GDN recurrent state F32 与 attention KV BF16 不冲突。
  不把这些配置不同的运行直接作为匹配性能对照，也不以 `S1-backend-check` 替代官方 S1。
- 官方完整 S1 logits/性能基准、真实候选对比、1K/8K/32K、三次重复吞吐、并发 8/32、冻结任务
  样本与任务分数仍需后续实测。现有 evaluator 只报告固定 token 的 NLL/KL/top-1，不包含这些能力。
