# Qwen3.6-35B-A3B 的 GDN 在 llama.cpp 中的实现导航

本文用于让后续 agent 快速定位 Gated DeltaNet (GDN) 的模型构图、CPU 内核、SIMD、线程调度和状态缓存。

- 核对日期：2026-09-21。
- 本地仓库：`/fastdata/users/habao/llama.cpp`。
- 核对版本：`c550d2f60bde72df19fcef1fef627895095b8ba8`。
- 文件链接相对本文所在目录；行号对应上述版本，更新代码后应按函数名重新定位。
- 验证范围：读取本地源码，并核对官方模型 `config.json`。没有编译、运行算子测试或进行模型性能测试。

## 1. 先看这些文件

**本地已经有专用的 CPU GDN 算子实现。最重要的入口是 `qwen35moe.cpp` 的模型构图和 `ggml-cpu/ops.cpp` 的递归计算内核。**

| 目的 | 文件 | 关键位置 |
| --- | --- | --- |
| 看完整的 Qwen GDN block | [src/models/qwen35moe.cpp](llama.cpp/src/models/qwen35moe.cpp) | `build_layer_attn_linear()`，第 359 行；`build_qkvz()`，第 251 行 |
| 看 fused / 非 fused 路径和状态回写 | [src/models/delta-net-base.cpp](llama.cpp/src/models/delta-net-base.cpp) | `build_delta_net()`，第 425 行；`build_recurrent_attn()`，第 527 行 |
| 看算子接口、形状和输出布局 | [ggml/include/ggml.h](llama.cpp/ggml/include/ggml.h) | `ggml_gated_delta_net()`，第 2652 行，前面有布局说明 |
| 看算子节点创建、类型检查 | [ggml/src/ggml.c](llama.cpp/ggml/src/ggml.c) | `ggml_gated_delta_net()`，第 6366 行 |
| 看 CPU 算子分发和工作区规划 | [ggml/src/ggml-cpu/ggml-cpu.c](llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c) | `GGML_OP_GATED_DELTA_NET`，第 2092、2291、3033 行 |
| **看实际 CPU 递归计算** | [ggml/src/ggml-cpu/ops.cpp](llama.cpp/ggml/src/ggml-cpu/ops.cpp) | **`ggml_compute_forward_gated_delta_net_one_chunk()`，第 10895 行**；线程调度在第 11049 行 |
| 看 FP32 向量操作 | [vec.cpp](llama.cpp/ggml/src/ggml-cpu/vec.cpp)、[vec.h](llama.cpp/ggml/src/ggml-cpu/vec.h) | `ggml_vec_dot_f32()`，`vec.cpp:11`；`ggml_vec_mad_f32()`，`vec.h:319`；`ggml_vec_scale_f32()`，`vec.h:703` |
| 看向量操作映射到 CPU 指令 | [simd-mappings.h](llama.cpp/ggml/src/ggml-cpu/simd-mappings.h) | SVE：172；NEON：331；AVX512F：446；AVX：581 |
| 看跨调用保留的状态 | [src/llama-memory-recurrent.cpp](llama.cpp/src/llama-memory-recurrent.cpp) | 第 102 行附近分配 `cache_r_l*`、`cache_s_l*`；`get_r_l()` / `get_s_l()` 在 1294 / 1298 行 |
| 看已有测试和算子性能用例 | [tests/test-backend-ops.cpp](llama.cpp/tests/test-backend-ops.cpp) | `test_gated_delta_net`：4667；缓存测试：4737；性能用例：11510 附近 |

建议阅读顺序：模型 block -> `build_recurrent_attn` -> fused op 接口 -> CPU 内核 -> SIMD / 缓存 / 测试。

## 2. 为什么找 qwen35moe，而不是 qwen36

核对的[官方 Qwen3.6-35B-A3B 配置](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json)中：

```text
architectures = ["Qwen3_5MoeForConditionalGeneration"]
text_config.model_type = "qwen3_5_moe_text"
num_hidden_layers = 40
hidden_size = 2048
linear_conv_kernel_dim = 4
linear_key_head_dim = linear_value_head_dim = 128
linear_num_key_heads = 16
linear_num_value_heads = 32
full_attention_interval = 4
num_experts = 256
num_experts_per_tok = 8
```

`layer_types` 中有 30 个 `linear_attention` 和 10 个 `full_attention`。每四层通常是三个 GDN block 加一个 full-attention block，各层另有 MoE FFN。

转换器 [conversion/qwen.py](llama.cpp/conversion/qwen.py) 第 643 行注册 `Qwen3_5MoeForConditionalGeneration`，将其映射到 `MODEL_ARCH.QWEN35MOE`；GGUF 架构字符串为 `qwen35moe`。因此模型商品名中的 `3.6` 不要求存在 `qwen36.cpp`。

`qwen35moe.cpp:4` 的 `load_arch_hparams()` 读取 GDN 参数；优先读取显式 recurrent layer 列表，缺少时按 `full_attention_interval` 推导，第 29 行将 40 层识别为 `LLM_TYPE_35B_A3B`。具体运行时始终以该 GGUF 的元数据为准。

同系列 dense 模型的对应入口是 [src/models/qwen35.cpp](llama.cpp/src/models/qwen35.cpp) 的 `build_layer_attn_linear()`，第 335 行。查 35B-A3B 时应先读 MoE 文件。

**GDN 属于带递归状态的 linear attention，不是 sliding-window/local attention。** 其中 kernel size 为 4 的 depthwise causal convolution 具有局部感受野，但后面的矩阵状态会持续汇总更早的历史；它没有把模型的历史范围限定为 4 个 token。

## 3. 从模型到 CPU 内核的调用关系

下面表示构图和执行的衔接。`ggml_gated_delta_net()` 只创建节点，实际计算由后端执行阶段触发。

```text
src/models/qwen35moe.cpp
  graph constructor: hparams.is_recr(il)
    build_layer_attn_linear()
      QKV / z / alpha / beta projections
      build_conv_state() -> ggml_ssm_conv() -> SiLU
      Q/K L2 normalization
      build_recurrent_attn()
        src/models/delta-net-base.cpp
          build_delta_net()
            build_delta_net_fused()
              ggml_gated_delta_net()
                ggml/src/ggml.c: GGML_OP_GATED_DELTA_NET node

CPU graph execution
  ggml/src/ggml-cpu/ggml-cpu.c: op dispatch
    ggml_compute_forward_gated_delta_net()
      ggml_compute_forward_gated_delta_net_f32()
        ggml_compute_forward_gated_delta_net_one_chunk()
          ggml_vec_scale_f32 / ggml_vec_dot_f32 / ggml_vec_mad_f32

Graph also contains
  updated state -> ggml_cpy() -> persistent recurrent cache
  GDN output -> gated RMSNorm -> output projection
```

`qwen35moe.cpp:189` 选择 GDN 或 full attention，后者由同文件的 `build_layer_attn()` 实现。MoE 则在后续的 `build_layer_ffn()` 中构图，不属于 GDN 递归内核。

### 完整 GDN block 做了什么

在 `build_layer_attn_linear()` 中依次可以找到：

1. `build_qkvz()` 投影出 QKV 和输出门控 `z`。
2. `ssm_beta` 投影后接 sigmoid，得到 delta update 的强度 `beta`。
3. `ssm_alpha` 投影加 `ssm_dt`，经过 softplus，再乘 `ssm_a`，得到 log decay `g`。转换时 `ssm_a = -exp(A_log)`；真正的衰减因子是内核中的 `exp(g)`。
4. 从 `get_r_l()`、`get_s_l()` 取得短卷积历史和递归矩阵状态。
5. `build_conv_state()` 拼入历史，`ggml_ssm_conv()` 做卷积，再接 SiLU。
6. 拆出 Q/K/V，对 Q/K 做 L2 normalization。
7. `build_recurrent_attn()` 计算 GDN 输出和新状态。
8. `build_norm_gated()` 执行 RMSNorm 和 `SiLU(z)` 门控，最后用 `ssm_out` 做输出投影。

**这里的 fused GDN 指第 7 步的递归核心。** 不要把整个 QKV 投影、卷积、SiLU、归一化和输出投影都理解为同一个 CPU kernel。

短卷积自己的 CPU 实现在 `ops.cpp:9703` 的 `ggml_compute_forward_ssm_conv_f32()`，按通道划分线程工作，循环处理较短的卷积核。

## 4. GDN 算子的张量契约和数学

下表维度采用 GGML `ne[]` 顺序，最左侧维度在内存中变化最快，与常见 PyTorch shape 的阅读习惯不同。设 `d = 128`、`Hk = 16`、`Hv = 32`、`T = 每序列本批 token 数`、`B = 序列数`。

| 输入 | GGML shape | 含义 |
| --- | --- | --- |
| `q`, `k` | `[d, Hk, T, B]` | 卷积和 L2 normalization 之后的 Q/K |
| `v` | `[d, Hv, T, B]` | 卷积之后的 V |
| `g` | `[1, Hv, T, B]` | 每个 head 的 log decay |
| `beta` | `[1, Hv, T, B]` | 每个 head 的更新强度 |
| `state` | `[d, d, Hv, B]` | 本批开始前的矩阵状态 |

当前 `ggml.c:6366` 的接口对这六个输入都断言 `GGML_TYPE_F32`。Q/K/V 要求行连续，`g`、`beta` 和状态要求连续。接口还支持 `g` 第一维为 `d` 的 KDA 分支，Qwen 此处使用的是第一维为 1 的 scalar gate。

为直接对应 CPU 内存布局，令 `M` 的行是 value 维度、列是 key 维度。每个 head、每个 token 执行：

```text
M_decay = exp(g_t) * M_previous
delta   = beta_t * (v_t - M_decay @ k_t)
M_new   = M_decay + delta @ transpose(k_t)
o_t     = (M_new @ q_t) / sqrt(d)
```

这里 `delta @ transpose(k_t)` 表示向量外积。代码注释采用 `S[key][value]`，所以写成 `s_out[j*d+i] = S[i][j]`；上述 `M` 就是该 `S` 的转置。修改内核或对照论文时必须先统一这个约定。

输出 tensor 把两块数据拼在一起：

```text
[ outputs: d * Hv * T * B floats | snapshots: d * d * Hv * B * K floats ]
```

`K` 是 op parameter 中的快照槽数，与 key tensor `k` 无关。`K=1` 仅返回最终状态；`K>1` 时，slot 0 最新、slot 1 为前一个 token 的状态，以此类推。只写最后 `min(T,K)` 个快照，不能读取其余未写槽位。

### Q/K 与 V 的 head 对应关系

CPU 通过 `value_head_index % q_or_k_head_count` 选择 Q/K head，采用 tiled broadcast。这依赖转换器提前重排 V 相关权重。

`conversion/qwen.py:453` 的 `_LinearAttentionVReorderBase` 将 HF 的 grouped V-head 顺序变为 tiled 顺序，涉及 V projection、`z`、alpha/beta、`A_log`、`dt_bias`、卷积 V 通道和输出投影。不要只修改 runtime head 索引而忽略转换器，否则张量尺寸正确也可能得到错误结果。

HF 到 GGUF 的张量名称映射还可查 [gguf-py/gguf/tensor_mapping.py](llama.cpp/gguf-py/gguf/tensor_mapping.py)：`in_proj_a` 对应 `SSM_ALPHA`，`in_proj_b` 对应 `SSM_BETA`。

## 5. CPU 加速实际发生在哪里

### SIMD 和状态内存访问

`ops.cpp:10895` 的内核把衰减、状态预测、delta update 和输出计算放在一个算子内。对每个 head 保留连续的状态矩阵，利用向量操作计算：

- `ggml_vec_scale_f32()`：衰减整个状态矩阵。
- `ggml_vec_dot_f32()`：状态行与 K/Q 的点积。
- `ggml_vec_mad_f32()`：状态行加上 `delta[j] * k`。

这些函数按编译配置使用 SIMD，也保留非 SIMD 路径；`scale` 在启用 Apple Accelerate 时另有 vDSP 路径。具体的 x86 AVX/AVX512、ARM NEON/SVE 映射见 `simd-mappings.h`，指令集编译选项见 [ggml/src/ggml-cpu/CMakeLists.txt](llama.cpp/ggml/src/ggml-cpu/CMakeLists.txt)。

因此，GDN 核心的 FP32 向量计算本身不要求 VNNI。VNNI、AMX 等与其他量化矩阵乘路径是否加速，是另一项检查。

### 线程按 head 和 sequence 并行

`ggml_compute_forward_gated_delta_net_f32()` 的总任务数为：

```text
nr = value_head_count * sequence_count
```

它通过 GGML threadpool 分配这些任务，通常使用动态取任务的方式，NUMA 模式有单独的分块处理。`one_chunk` 中，一个任务处理一个 `(head, sequence)`，然后执行 `for (t = 0; t < n_tokens; t++)`。

对单序列、32 个 V head 的配置，核心最多只有 32 个独立的 head 工作单元；当前没有进一步把同一 head 的状态行拆给多个线程。更多 CPU 线程不保证更快。这里的限制只描述 GDN 核心，投影矩阵乘和 MoE 有各自的并行策略。

### Prefill 的 chunk 名字不等于 token 并行

`delta-net-base.cpp` 中有三种构图函数：

| 函数 | 位置 | 用途 |
| --- | --- | --- |
| `build_delta_net_autoregressive()` | 第 289 行 | 非 fused 的单 token 图 |
| `build_delta_net_chunking()` | 第 16 行 | 非 fused 的多 token 分块图；GDN chunk size 为 64，包含矩阵操作和三角求解 |
| `build_delta_net_fused()` | 第 373 行 | 单个 `GGML_OP_GATED_DELTA_NET` 节点，支持单 token / 多 token |

[src/llama-context.cpp](llama.cpp/src/llama-context.cpp) 第 233 行将 `fused_gdn_ar`、`fused_gdn_ch` 初始化为 `true`，`auto_fgdn` 为 `false`。普通路径由 `build_delta_net()` 根据 token 数和这两个标志选择。

当前 CPU fused 实现中，decode 和多 token prefill 都进入上面的递归内核，**每个 head 内的时间维仍然串行**。`GDN_CH` 是多 token 路径标签；`one_chunk` 则是线程任务分块。它们都不能证明 CPU 正在使用与 GPU Flash Linear Attention 类似的时间分块并行算法。

### 量化线性层与 FP32 递归状态要分开看

GDN block 的投影经 [src/llama-graph.cpp](llama.cpp/src/llama-graph.cpp) 第 1514 行的 `build_lora_mm()` 构建 `ggml_mul_mat()`，普通 CPU 入口可查 `ggml-cpu.c:1255` 的 `ggml_compute_forward_mul_mat()`。MoE 的 routed expert 路径还涉及 `MUL_MAT_ID`，对应 CPU 入口在第 1550 行；repack / 专用 buffer 等优化可能接管具体执行。

Q4/Q8 等 GGUF 权重可以让这些矩阵乘使用相应的量化路径，但这不代表 GDN 的 `state`、Q/K/V 和递归计算也变成 Q4、INT8 或三值。当前这里的接口和 CPU 核心是 FP32。

## 6. 状态缓存在哪里，内存怎么估算

[src/llama-memory-hybrid.cpp](llama.cpp/src/llama-memory-hybrid.cpp) 同时管理普通 attention KV cache 和 recurrent memory。它们按 layer filter 分别服务于 full-attention 层和 GDN 层。

- [src/llama-model.cpp](llama.cpp/src/llama-model.cpp) 第 2634 行附近创建 hybrid memory，第 2643、2644 行将 recurrent 两类缓存设为 `GGML_TYPE_F32`。
- `llama-memory-recurrent.cpp:102` 分配卷积历史 `r` 和矩阵状态 `s`；行数是 `mem_size * (1 + n_rs_seq)`。
- [src/llama-hparams.cpp](llama.cpp/src/llama-hparams.cpp) 第 208 行 `n_embd_r()` 计算卷积历史大小；第 236 行 `n_embd_s()` 计算矩阵状态大小。
- `delta-net-base.cpp:449` 的 `build_conv_state()` 更新卷积历史。
- `delta-net-base.cpp:527` 的 `build_recurrent_attn()` 调用 GDN 并通过 `ggml_cpy()` 回写矩阵状态。

普通路径使用 `K=1`。启用 recurrent rollback 保留时，`build_recurrent_attn()` 直接创建 fused op，使用 `K=n_rs_seq+1` 并回写快照。改变返回布局时必须同步检查这条路径。

对 Qwen3.6-35B-A3B，单个序列、单份核心状态的理论大小为：

```text
每层：32 * 128 * 128 * 4 bytes = 2 MiB
30 个 GDN 层：30 * 2 MiB = 60 MiB
```

这只是核心矩阵状态，不包括短卷积历史、算子输出/工作区、rollback 快照、额外序列、权重和普通 attention KV cache。每份 GDN 矩阵状态大小不随已处理 token 数增长；整个混合模型仍有 full attention 的 KV cache，不能据此说整个模型的缓存恒定。

## 7. 后续验证可用的现成命令

以下命令已按当前源码中的选项核对，**本轮没有执行**。模型路径是占位符；不需要为文档另写测试。

在 Linux 上建立一个单独的 CPU 构建目录，避免覆盖已有 build：

```bash
cd /fastdata/users/habao/llama.cpp
cmake -S . -B build-gdn-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF \
  -DGGML_VULKAN=OFF -DGGML_SYCL=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build build-gdn-cpu --config Release \
  --target test-backend-ops llama-bench -j
```

`GGML_NATIVE=ON` 针对编译所在 CPU 优化，迁移二进制到不同 CPU 时应重新考虑指令集配置。

算子支持、已有后端测试和性能测试：

```bash
./build-gdn-cpu/bin/test-backend-ops support -b CPU -o GATED_DELTA_NET
./build-gdn-cpu/bin/test-backend-ops test -b CPU -o GATED_DELTA_NET
./build-gdn-cpu/bin/test-backend-ops perf -b CPU -o GATED_DELTA_NET
```

`tests/test-backend-ops.cpp:4667` 的用例支持不同 head 数、head dim、多 token、多序列、Q/K-V head 数不等、非全连续视图、KDA 和多快照。第 4737 行还有 GDN 加缓存复制的测试；第 11510 行附近有 `d=128` 的 decode 和 prefill 性能用例。

测试工具默认以 CPU 作为比较基准，并开启 reference 标志。但此版本的 GDN 内核没有独立的 `use_ref` 数学实现，因此 `test -b CPU` 不能单独证明改写后的 GDN 算法正确。改动 CPU 数学后，需要与未改动版本、非 fused 图或独立参考实现比较输出与最终状态；还要覆盖 head broadcast 和 `K>1` 快照。现有 perf 中的 “Qwen3.5-like” case 使用等 Q/K/V head 数，不是完整模型所有细节的复刻。

这个版本的 `test-backend-ops` 没有 `-t` 线程参数；`-j` 控制 test mode 并发测试数。第 12135 行附近设置后端线程数，避免误用 `-j` 推断 GDN 内核的线程伸缩性。

端到端 CPU 性能示例：

```bash
./build-gdn-cpu/bin/llama-bench \
  -m /path/to/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -ngl 0 -t 16 -p 512 -n 128
```

分别看 `pp512` 和 `tg128`，它们衡量 prompt processing 和 token generation，包含投影、GDN、full attention、MoE 等全模型成本。此命令不会把两项自动变成一次“先读 512 token 再生成 128 token”的联合测试。若要测已有上下文长度对速度的影响，另查本版本 `llama-bench --help` 的 `-d/--n-depth` 和组合测试参数。

做性能比较时记录 commit、编译选项、CPU/NUMA、线程数、模型量化格式、batch/ubatch、序列数和快照配置。单个 GDN 算子变快，并不自动意味着 35B-A3B 整体同比例变快。

## 8. 给后续 agent 的定位提示

| 下一步研究 | 优先看哪里 |
| --- | --- |
| 改 CPU GDN 核心计算或减少状态内存访问 | `ops.cpp` 的 `ggml_compute_forward_gated_delta_net_one_chunk()` |
| 改线程粒度或研究单序列多核利用率 | 同文件的 `ggml_compute_forward_gated_delta_net_f32()`；连同 `ggml-cpu.c` 的任务数/工作区规划检查 |
| 研究更快的 CPU prefill | 对照 `build_delta_net_chunking()` 与当前 fused token loop；先测不同 `T` 下的算子成本 |
| 改状态精度 | 同时检查 `ggml.c` 类型约束、CPU 内核、memory 分配、快照和误差累积 |
| 查模型输出错误但 tensor shape 正确 | Q/K normalization、log decay、矩阵转置、V-head 重排和快照顺序 |
| 研究 GGUF 权重量化对速度的影响 | 投影的 `MUL_MAT`、专家的 `MUL_MAT_ID` 及实际选用的量化/repack 内核 |

源码更新后可重新定位：

```bash
rg -n 'build_layer_attn_linear|build_qkvz' src/models/qwen35moe.cpp
rg -n 'build_delta_net|build_recurrent_attn|build_conv_state' src/models/delta-net-base.cpp
rg -n 'fused_gdn_ar|fused_gdn_ch|auto_fgdn' src/llama-context.cpp
rg -n 'GGML_OP_GATED_DELTA_NET|ggml_compute_forward_gated_delta_net' ggml/src/ggml-cpu
rg -n 'ggml_gated_delta_net' ggml/include/ggml.h ggml/src/ggml.c
rg -n 'test_gated_delta_net' tests/test-backend-ops.cpp
rg -n 'Qwen3_5Moe|_LinearAttentionVReorderBase' conversion/qwen.py
```

其他后端可以作为算法或布局的对照，但不能据其代码推断 CPU 的实际执行方式：

- CUDA：[gated_delta_net.cu](llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu)。
- Metal：[gated_delta_net.metal](llama.cpp/ggml/src/ggml-metal/kernels/gated_delta_net.metal)。
- Vulkan：[gated_delta_net.comp](llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/gated_delta_net.comp)。
- SYCL：[gated_delta_net.cpp](llama.cpp/ggml/src/ggml-sycl/gated_delta_net.cpp)。
- OpenCL：[gated_delta_net.cl](llama.cpp/ggml/src/ggml-opencl/kernels/gated_delta_net.cl)。

本文只追踪当前本地 llama.cpp 的 Qwen GDN 路径。此前 Bonsai 三值权重、Hadamard 变换和模型训练方式的讨论见 [doc.md](doc.md)；不能仅根据 GDN 架构相同就认定不同 GGUF 量化格式的执行路径也相同。
