# BitNet MoE Script

YOCO-MoE 模型的 GGUF 生成、构建与性能基准测试工具集。

## 模型架构

基于 **YOCO (You Only Cache Once)** 架构，结合 **Mixture-of-Experts (MoE)** 和 **BitNet 量化**，支持两种变体：

| 变体 | 层数 | 总参数量 | 激活参数量 | 说明 |
|------|------|---------|-----------|------|
| YOCO-MoE-30B-A3B-V3 | 20 (10 self + 10 cross) | ~31.77B | ~3.45B | 标准版本 |
| YOCO-U-MoE-30B-A3B-V3 | 40 (10 self × T=3 + 10 cross) | ~31.77B | ~6.47B | Universal Loop 版本，自解码器权重共享 |

**核心配置：**
- `d_model=3072`, `head=32`, `kv_head=8`, `head_dim=128`
- MoE: 128 experts, top-8 路由, `ffn_dim=1280`
- 量化: TernarySEQ 权重量化 + ADP 8-bit 激活量化
- YOCO window size: 512

## 项目结构

```
.
├── generate_model_script/          # GGUF 模型生成脚本
│   ├── generate_yoco_moe_30b_a3b_v3_models.py
│   └── generate_yoco_u_moe_30b_a3b_v3_models.py
├── build_script/                   # BitNet llama.cpp 编译脚本
│   └── build_yoco_moe_models.sh
├── benchmark_script/               # CPU 基准测试脚本
│   ├── benchmark_yoco_moe_30b_a3b_v3_cpu.sh
│   └── benchmark_yoco_u_moe_30b_a3b_v3_cpu.sh
├── simple_quant_moe_infer_with_mtp.py  # 独立 PyTorch 推理参考实现
└── model-config.md                 # 模型配置说明
```

## 使用流程

### 1. 生成 GGUF 模型

```bash
# 标准 YOCO-MoE
python generate_model_script/generate_yoco_moe_30b_a3b_v3_models.py --output-dir /path/to/output

# YOCO-U (Universal Loop) 变体
python generate_model_script/generate_yoco_u_moe_30b_a3b_v3_models.py --output-dir /path/to/output
```

生成 F16 和 I2_S 两种精度的 GGUF 模型文件。

### 2. 构建 llama.cpp

```bash
bash build_script/build_yoco_moe_models.sh
```

编译支持 YOCO 架构的 `llama-bench`、`llama-quantize`、`llama-embedding` 工具，启用 AVX-512 指令集优化。

### 3. 运行基准测试

```bash
# 标准 YOCO-MoE，指定线程数和可选 NUMA 节点
bash benchmark_script/benchmark_yoco_moe_30b_a3b_v3_cpu.sh 8 0

# YOCO-U 变体
bash benchmark_script/benchmark_yoco_u_moe_30b_a3b_v3_cpu.sh 8 0
```

测试 F16 与 I2_S 在不同 prompt/generation 长度下的 CPU 推理性能。

### 4. PyTorch 参考推理

`simple_quant_moe_infer_with_mtp.py` 提供独立的单 GPU PyTorch 推理实现，仅依赖标准库和 PyTorch，支持从合并检查点加载量化模型。

## 依赖

- [BitNet](https://github.com/microsoft/BitNet) 及其 llama.cpp 子模块
- Python 3, NumPy, PyTorch
- CMake, Clang (编译)
