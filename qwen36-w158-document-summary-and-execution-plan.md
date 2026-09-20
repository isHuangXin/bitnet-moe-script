# Qwen3.6 w1.58 文档解读与后续工作梳理

整理日期：2026-09-19。

原始规范：[qwen36-w158-quantization-settings-and-cpu-handoff.md](qwen36-w158-quantization-settings-and-cpu-handoff.md)。

本文整理原始规范的目标、实验矩阵、当前代码基础与后续执行顺序，不替代原始规范中的完整量化定义。

## Context：这次要回答什么

已阅读[交接文档](/home/huangxin/code_list/bitnet-moe-script/qwen36-w158-quantization-settings-and-cpu-handoff.md:1)，并只读核对相关脚本和源码。核对时 BitNet 与嵌套 llama.cpp 均在 `dev-yoco-u-moe-bitnet` 分支，HEAD 分别为 `4af06fe`、`c7a218c6e`。

本次仅做内容整理，不修改项目代码、不转换权重、不运行训练或整模型测试。下列执行路线用于解释文档中的待办，不代表这些实验已经完成，也不是本次要启动的实施任务。

## 1. 一句话概括

这是一份**量化数值规范 + 量化位置对照实验方案 + CPU 推理交接协议**：从同一份官方 Qwen3.6-35B-A3B BF16 权重出发，构建不同量化候选，先确认质量和数值正确性，再研究 CPU 上的速度与实际内存。

它不是已经完成的性能报告，也不是要求依赖私有 QAT checkpoint 的复现教程。

## 2. 文档已经规定的关键约束

- 唯一必需来源：官方 `Qwen/Qwen3.6-35B-A3B` BF16 snapshot，revision 固定为 `995ad96eacd98c81ed38be0c5b274b04031597b0`。只研究文本部分，排除 vision 和 MTP。
- S1 是官方原始 BF16 权重与官方 forward，不能用 fake-quant 模型替代。
- S2–S9 是从官方权重生成的公开 PTQ 候选，不能当作私有 iteration 2000 QAT 模型的副本。直接将专家权重三值化可能严重降低质量，这正是需要测量的内容。
- w1.58 指三值权重 `{-1, 0, +1}`；实际按 2 bit/weight 打包，另外保存 scale。沿每个输出行的 K 维 group-128，abs-mean、threshold=0.5、ties-to-even。
- 专家使用 W2A8：三值权重与动态 group-128 INT8 激活；组内 INT32 点积，按组应用 scale 后 FP32 累加，保留约定的 BF16 舍入边界。
- Embedding / LM head 是 INT8 权重 + BF16 reciprocal scale，按 `q / reciprocal` 恢复。Head 是 W8A16，不能因为专家使用 A8 就顺手改为 W8A8。
- BF16 weight scale 是落盘和常驻 metadata 的约束；不是把所有 reduction、激活 scale 和累加都改成 BF16。
- Router / shared gate / norm / GDN 状态等遵循保护名单，尤其不能随意改变 top-8 路由、概率乘入位置和 GDN 递归计算。

## 3. S1–S9 分别比较什么

S2–S9 共同使用三值 routed experts、INT8 embedding/head 和 BF16 weight scales；表中描述其余大投影。

| 组别 | 相对 S2 的新增变化 | 静态权重 GiB |
| --- | --- | ---: |
| S1 | 官方原始 BF16，对照基准 | 64.5604 |
| S2 | 基础量化方案，其余大投影 BF16 | 11.5967 |
| S3 | Shared expert 投影改 w1.58 | 11.3935 |
| S4 | Full-attention 投影改 w1.58 | 11.1563 |
| S5 | GDN 投影改 w1.58 | 9.9644 |
| S6 | Attention + GDN 投影改 w1.58 | 9.5240 |
| S7 | Shared + attention + GDN 投影全部改 w1.58 | 9.3208 |
| S8 | 三类剩余大投影从原 BF16 改 FP8，128×128 block | 10.2846 |
| S9 | 三类剩余大投影从原 BF16 改 INT8，128×128 block | 10.2846 |

主线是 S1 → S2 → S6 → S7；S3/S4/S5 用于定位单类模块的影响；S8/S9 是从 S2 独立出发的对照，不是从 S7 转回 8 bit。

辅助组：

- **S2-scale32**：与 S2 相同 codes，保留 FP32 metadata，用于隔离 scale 收窄的误差。
- **C8**：与 S2 相同三值 codes，但用 INT8 容器存储，用于分离 packed2 解包成本和带宽收益，不是更高精度的模型。

每个 setting 还要记录 activation profile：新增投影先做 A16 weight-only 对照，S3–S7 再测新增 w1.58 投影的 A128。矩阵名称本身不包含这些计算路径差异。

## 4. 已经完成什么，尚未完成什么

文档 v2 记录已经完成小张量 CPU reference 自检、693 个官方文本 tensor 的 headers 分类和空间账本核算；这些不是整模型质量或 CPU 性能验证。本次没有重跑这些检查。

[文末状态表](/home/huangxin/code_list/bitnet-moe-script/qwen36-w158-quantization-settings-and-cpu-handoff.md:941)明确仍待执行：官方 S1 整模型 baseline，公开 S2/S2-scale32 导出与 loader，S3–S9 的候选构建和质量实验，以及 CPU S1–S9/C8 的吞吐与 RSS 测量。

**因此，文档现在交付的是“规则、参考函数和验收计划”，不是已经可直接运行的全套 exporter、loader 和优化后端。**

## 5. 对照当前代码，哪些可以借用，哪些不能直接套用

### 5.1 已有 Qwen 架构支持，不需要从零重写模型

- [conversion/qwen.py:625](/home/huangxin/code_list/BitNet/3rdparty/llama.cpp/conversion/qwen.py:625) 已注册 `Qwen3_5MoeForConditionalGeneration` / `Qwen3_5MoeForCausalLM`。文档中的 Qwen3.6 snapshot 本来就使用 `qwen3_5_moe` 类型，名字不同不代表架构不支持。
- [src/models/qwen35moe.cpp:75](/home/huangxin/code_list/BitNet/3rdparty/llama.cpp/src/models/qwen35moe.cpp:75) 已有 full-attention、GDN、routed/shared experts 的加载和计算图，可作为 CPU 接入基础；但架构存在不等于新量化协议已经接通。

### 5.2 现有 I2_S 不等于文档的 w1.58 格式

[ggml-cpu/quants.c:1362](/home/huangxin/code_list/BitNet/3rdparty/llama.cpp/ggml/src/ggml-cpu/quants.c:1362) 的按行 I2_S 路线使用整行 absmax/TernarySEQ、FP32 per-row scale，编码为 `00=-1、01=0、10=+1`；文档要求 group-128 abs-mean、BF16 effective scale，编码为 `00=0、01=+1、11=-1`，且打包次序不同。

[quantize_row_i8_s:1311](/home/huangxin/code_list/BitNet/3rdparty/llama.cpp/ggml/src/ggml-cpu/quants.c:1311) 则对传入的整行激活求一个 scale，使用 `roundf`，不是文档规定的 A128 分组与 ties-to-even 流程。

**可以复用整数点积、SIMD 和调度思路，但必须明确适配编码、分组 scale、舍入与算子接口。不是换个量化类型名字就能完成。**

### 5.3 Qwen 当前 MoE forward 也不是候选 recipe

Qwen 调用通用 MoE 的 `SOFTMAX` 路径：[qwen35moe.cpp:500](/home/huangxin/code_list/BitNet/3rdparty/llama.cpp/src/models/qwen35moe.cpp:500)。当前 helper 先对全专家 softmax 再选 top-k、归一化，并在 down/FC2 之后乘路由权重：[llama-graph.cpp:1841](/home/huangxin/code_list/BitNet/3rdparty/llama.cpp/src/llama-graph.cpp:1841)、[llama-graph.cpp:2136](/home/huangxin/code_list/BitNet/3rdparty/llama.cpp/src/llama-graph.cpp:2136)。

文档的 S2–S9 要求 top-k logits 后 softmax，并在 FC2 输入 A128 之前乘概率、舍入 BF16。这些运算在加入量化后不能随意交换顺序；需要单独适配量化候选，不能为了对齐候选而修改 S1 的官方 forward。

### 5.4 现有脚本主要是 YOCO 性能实验工具，不是本次公开 Qwen exporter

- [generate_yoco_moe_30b_a3b_v3_models.py:441](/home/huangxin/code_list/bitnet-moe-script/generate_model_script/generate_yoco_moe_30b_a3b_v3_models.py:441) 用 `np.random.randn` 生成权重；[同文件:550](/home/huangxin/code_list/bitnet-moe-script/generate_model_script/generate_yoco_moe_30b_a3b_v3_models.py:550) 调用现有 I2_S 量化，并把 embedding 设为 Q8_0。这既不是官方 Qwen 权重来源，也不是文档的 INT8 group-128 reciprocal 格式。
- [benchmark_qwen3_30b_a3b_cpu.sh:12](/home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen3_30b_a3b_cpu.sh:12) 测的是 **Qwen3-30B-A3B 的 F16/Q4_0**，不是本文 **Qwen3.6-35B-A3B 的官方 BF16 S1**。
- [benchmark_yoco_moe_30b_a3b_v3_cpu.sh:20](/home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_yoco_moe_30b_a3b_v3_cpu.sh:20) 的 NUMA 绑定、线程参数、CPU-only 和重复测量组织方式可以复用，但要补齐本次模型、长度、并发与质量指标。
- [verify_precision_yoco_moe_30b_a3b_v3.py:75](/home/huangxin/code_list/bitnet-moe-script/verify_precision_script/verify_precision_yoco_moe_30b_a3b_v3.py:75) 实际比较的是两套 PyTorch 量化公式的输出，没有调用 C++ 引擎，不能当作新 CPU kernel 已通过验证的证据。

结论：**有可复用的模型与 CPU 工程基础，但没有从官方 source 到文档 S1–S9 的现成完整实验链路；直接运行现有脚本不能完成本文任务。**

## 6. 具体要做什么：推荐执行顺序

**优先完成 S1 + S2-scale32 + S2 的最小闭环，不要一开始就实现九套优化 kernel。** 先回答“官方权重直接三值 PTQ 后是否仍可用，以及 scale 收窄是否引入可接受误差”，再扩大量化范围。

1. **建立 baseline。** 固定官方 source / backend 版本 / tokenizer / 输入，审计文本 tensor，跑真正的 S1 BF16。若选 llama.cpp，先建立与官方原生实现对照的 `S1-backend-check`，不要把换 backend 的误差混成量化误差。
2. **实现导出与加载。** 从官方权重分块生成 S2-scale32 与 S2，共用 codes；输出显式 manifest，记录 shape、dtype、分组、scale 方向、packing ID 和 hash。补齐专家 packed2、embedding/head reciprocal 的 loader/量化算子，不把修改配置字符串当作格式支持。
3. **打通 reference 验证。** 按格式、单算子、固定路由 MoE、整模型 teacher-forced logits 分层核对。S1 对 S2 测完整方案变化；S2-scale32 对 S2 只测 metadata 收窄。仅收窄 metadata 预计节省约 495.16 MiB，但 codes 不变不代表恢复值或输出不变。
4. **跑位置实验。** 用 S3/S4/S5 定位各模块影响，实测组合 S6/S7，再比较独立的 S8/S9；新增投影先测 A16，再测规定的 A128 profile。组合结果不能由单模块结果相加推断。
5. **做质量筛选。** 固定数学/代码/中文样本及解码设置，记录任务分数、NLL/KL、top-1 一致率和中间层误差；首轮覆盖 1K/8K/32K，通过后扩展 64K/128K。尤其不能只凭几条生成文本判断 GDN 量化是否可用。
6. **交给 CPU 优化与 benchmark。** 候选在正确性和质量上通过后，优化局部解包、整数点积、tiling 和融合。在同一 CPU/线程/NUMA/输入输出长度下测冷加载、prefill、B1 decode、并发 8/32、TTFT、吞吐和 RSS；每组至少重复 3 次，报告相对 S1 的变化，并用 S2/C8 检查 packed2 的真实收益。

最终交付应包括：**可审计的量化权重与 manifest、可正确加载的 reference/CPU 路径、质量报告、同条件性能与内存结果表**，而不只是生成几个 GGUF 文件。

## 7. 验收边界与重点

- 表中 GiB 只包含文本权重和 metadata，不包含 KV/GDN state、激活、logits、workspace 等。权重文件 payload、常驻权重和峰值 RSS 必须分开；禁止完整展开 INT8/BF16 后仍按 packed2 大小报告常驻内存。
- GDN 是新增量化最大的空间收益来源，也是长上下文验证的重点。S7 比 S2 的权重 payload 减少约 2.276 GiB（19.63%），并不保证 CPU 吞吐按比例提升。
- S1 若在目标机器上 OOM 或 paging，要明确记录无法取得同条件可比结果，不能用 offload 或另一台机器的数据冒充 baseline。
- **量化质量损失与 kernel 对同一量化 reference 的实现误差，是两类验收。** 阈值应在看结果前固定。若官方三值 PTQ 严重退化，如实报告；如需 QAT，另建可复现训练轨迹，不能拿私有 QAT 结果替代公开 PTQ 实验。

**归根结底：这篇文档要把“量化哪些位置、使用什么精度、能否保持质量、CPU 上是否真的省内存并加速”变成一套可复现的对照实验；目前已有规范和部分工程基础，关键的官方权重导出、数值适配与整模型实验仍需落实。**
