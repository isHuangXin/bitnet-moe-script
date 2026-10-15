# Qwen3.6 prefix snapshots: BF16 / Q4_0 independent prefill runs

**本版将文件快照读取改为 `io_uring+O_DIRECT`（输出模式 `io-uring-direct`），保留推理/统计、buffered+fsync 写入路径和 v1 文件格式。不自动哈希/加载真实模型或启动 GPU 实验。** 无模型检查不代表运行过真实模型扫描，也不能证明 GPU parity、性能、256K 或缓存加速；历史 buffered 读取的实验是否成功应以对应目录中的原始记录为准，不能重新标注为 direct I/O 结果。

保留所有历史小样本与原 `benchmark_qwen36_cuda.sh`。例如历史 v0 Q4_0 `/data2/huangxin/qwen36_cache_experiments/runs/run.mXyP8m2O`（P127/S1/D8/context256，runtime-id `69ecfbe8abf64c63679cbb87160efd951458eb0c3a72f6366aa13821d5f081e3`）通过了当时版本的 DRAM/file parity，但固定顺序、单次、无匹配 warmup 的耗时不能证明稳态加速，也不能作为新版验证结论。旧文件不迁移、不改写、不补算成新版统计。

## BF16 与 Q4_0 独立 pp 测试

wrapper 按参考脚本分成 **BF16、Q4_0 两组标题与调用**：当前 BF16 组整段注释，默认只执行 Q4_0。每个启用的组调用一次 runner，单独生成结果目录和表格；两组都启用时先 BF16 后 Q4_0，不合并结果。测试长度以脚本的 `Prefill_PROMPTS` 为准，可直接调整；decode 前向仍全部去掉。例如仅测试两个长序列长度：

```bash
Prefill_PROMPTS="131077,262144"
```

固定模型为：

- `/data3/huangxin/model_list/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-bf16.gguf`
- `/data3/huangxin/model_list/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-Q4_0.gguf`

`ppN` 表示**完整输入 P+S=N**，不是只有缓存前缀。每项 P=N-1、S=1、D=0；只有 context 按 `((N+255)/256)*256` 向上对齐，输入长度和 pp 标签不变：

| pp | P | S | D | context |
| --- | --- | --- | --- | --- |
| 128 | 127 | 1 | 0 | 256 |
| 65535 | 65534 | 1 | 0 | 65536 |
| 131077 | 131076 | 1 | 0 | 131328 |
| 262144 | 262143 | 1 | 0 | 262144 |

测试长度不做静默取整，也不额外加入 decode case。`pp131077` 和 `pp262144` 是否执行由 `Prefill_PROMPTS` 决定；`pp262144` 的完整输入已占满窗口，不能再追加任何 token 前向。`P+S+D <= context <= 262144` 及实际模型/补齐窗口检查仍然生效。

### 参考脚本风格的 shell 入口（保持四参数）

`benchmark_qwen36_prefix_cache_cuda.sh` 保持简短直接的结构：模型/程序路径、参数、`Prefill_PROMPTS`、参数检查、配置横幅、公共工具检查、BF16 组、Q4_0 组和成功后的 `Done.`。每组仅保留标题和 `python3 -I "$RUNNER"` 调用，分别只传 `--bf16-model` 或 `--q4-model`；所选模型的文件检查、哈希以及 runtime 校验、循环、日志和汇总均由 `run_sweep.py` 承担。这里不能改用普通 `llama-bench` 的 `-m/-t/-p` 参数，否则会绕过前缀恢复和正确性检查。

当前已注释 BF16 组，Q4_0 可直接运行，BF16 文件无需存在。要启用 BF16，取消该组每行开头的 `#`；只跑 BF16 时，再将 Q4_0 的标题和 Python 调用整组注释。保留公共参数和工具检查。**仅调整 benchmark 启动脚本中的测试长度、线程数、GPU、重复/预热次数或模型组，不需要重新编译或更新 runtime 指纹。** 启动脚本不纳入构建指纹；实际传入参数写入每次 sweep 的 `metadata.tsv`。`set -e` 保持：两组都启用时，BF16 失败就停止，不继续 Q4_0。

直接调用 `run_sweep.py` 时，`--bf16-model` 和 `--q4-model` 至少提供一个；未选参数应省略，不能传空字符串。仍兼容同时提供两个模型的旧调用方式（该次调用生成一张合并表），但 shell 每个启用的组各调用一次。

```bash
# NOT RUN: [threads] [gpu_id] [repetitions] [warmup]，默认 8 0 3 1
bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_prefix_cache_cuda.sh 8 0 3 1
```

**这是用户明确选择空闲卡后才运行的独立模型 sweep 入口，不是本次执行命令，也不是单 case smoke。** repetitions/warmup 均为 `1..1000`。参数个数、正整数/溢出、GPU index 和重复/预热边界在模型哈希或创建运行产物前校验。旧的 `[threads] [gpu_id] [model_path] [prefix] [suffix] [decode] [ctx] [repetitions] [warmup]` shell 用法已不支持；多余参数或把模型路径放第三位会报错，不做旧接口自动识别，也没有 `--sweep`、自动跳过、重试、CPU fallback 或 `--dry-run`。

runner 为子进程设置 `CUDA_VISIBLE_DEVICES=$gpu_id`，只暴露一张卡，probe 内 main_gpu=0、ngl=99。每次调用只校验和哈希传入的模型，单模型调用只哈希该模型一次，不检查另一份模型；准备工作在本次表头打印前完成，模型加载/哈希不计入请求耗时。每次调用固定自己的 `runtime-id`，每项前后检查 manifest、构建指纹、源码版本和所选模型身份，拒绝中途更换模型或混用构建，不自动重建。单次调用的表头之后不再插入哈希提示、原生 probe 进度或各项小表。

扫描可能耗时**多小时**：每个 case 是独立 probe 进程，重新加载模型，建立 golden、准备快照，再进行 B 干扰、预热和重复比较。BF16 长上下文可能显存不足；DRAM、编码快照大小和 NVMe 容量也可能触发限制。不会为跑完所有项而静默缩短输入、放宽容量或回退 CPU。任何失败立即停止，显示模型、pp 和 case 目录；已经完成的输出保留，不继续下一项。

## 快照范围与每个 case 的协议

这是跨请求的**完整前缀状态快照**，不是活跃 KV 分层分页、CPU offload 调度器或 Strata 集成。文件 store 改名为 `DirectFileStore`，构造参数与 put/get 接口、v1 记录格式不变；不改 llama.cpp、不重编译 CUDA，不增加 LRU、服务或 trace replay。

- 单序列 ID 0；public `llama_state_seq_get_size/get_data/set_data` 使用完整 host 状态，flags=NONE，不用 PARTIAL_ONLY/ON_DEVICE。
- 包含 full-attention KV（F16）、GDN recurrent state 和 convolution history（F32），不只是 attention KV。
- 保存准确的已处理前缀 token 与位置。快照**不含 logits/sampler**；恢复后必须计算至少 1 个未处理 suffix token，再获得 logits 和首 token。
- 只复用完全相同的前缀，不支持 hybrid state 任意裁剪/rollback；较早分叉需恢复更早 checkpoint 后向前重算。
- 恢复前 clear；导入字节数错误/失败后再次 clear 并退出，不使用半恢复状态。

每个 case 都比较 recompute、DRAM、io-uring-direct，没有 mode selector。建立不参与统计的 golden recompute，DRAM/file **各准备一次** A 前缀快照，之后 warmup/measurement 只读，不重复 put/fsync。admission、prefix、export、store/fsync 成本单独记录在 metadata，不混入 hit trial。

每个请求前执行并同步不同的 B 干扰，放在请求计时之外。请求内 clear 后重算 A 或真正读取/恢复 A，计算同一 suffix，再采样首 token。A/B/A 排除残留 GPU 状态造成的假命中。先完成所有模式的等量 warmup（默认每模式 1 次），再 measurement（默认每模式 3 次）；两个阶段分别在第 r 轮按 `[recompute, dram, io-uring-direct][(r+s)%3]` 轮换次序。

每次 warmup/measurement 都校验身份、字节数、位置与 logits/token，和同一 golden 比较。后缀后的完整 logits 必须有限，绝对/相对误差阈值均为 `1e-4`，greedy 首 token 完全相等。D>0 的底层兼容路径还检查每步生成（含最后已评估 token）的有限性与 token 序列。失败不放宽阈值，不过滤失败 trial 再报告成功均值。

## Pure-prefill 的 TTFT 与吞吐

D 表示请求的额外 decode 前向数。**D=0 仍从 prefill logits 采样一个首 token，用于 TTFT/parity，但绝不把它送回模型，不额外调用 `llama_decode`。** 因而 `sampled_output_count=1`、`evaluated_decode_tokens=0`、`decode_ms=0`、`decode_forward_ms=0`，并且 `last_sample_ms=ttft_ms`。采样不需要新增 context 位置，这正是完整 pp262144 合法的原因。

所有时间以 ms 记录：T0 在已同步 B 之后、请求 clear/load 之前；Tready 在 suffix 前向同步后、host logits 捕获/检查和首次采样之前；Tfirst/Tlast 为首/末次 sample 返回；Tend 为请求结束。D=0 的首/末次采样相同，Tend 可与 Tlast 相等，没有最后一个输出 token 的前向更新。

| 指标 | 定义/纯 PP 口径 |
| --- | --- |
| `prompt_ready_ms` | `Tready-T0`，包含 clear、命中时的 load+CRC/兼容性校验、restore 与 prefill |
| `ttft_ms` | `Tfirst-T0`，包含实际发生的 host logits 捕获/检查和首次采样 |
| `actual_prefill_eval_tps` | recompute：`1000*(P+S)/(prefix_ms+tail_ms)`；命中：`1000*S/tail_ms` |
| `effective_prompt_tps` | `1000*(P+S)/prompt_ready_ms`，包含命中恢复开销，不是实际执行的 prefill 吞吐 |
| `conditioned_decode_forward_tps` | D=0 时为 NA，不写 0 或 infinity 冒充吞吐 |
| `generation_after_first_tps` | D=0 时为 NA，没有首 token 后生成 |
| `end_to_end_output_tps` | 若保留，D=0 时为 `1000*1/ttft_ms`，只代表首 token 就绪率，不是 decode 吞吐 |
| `request_ms` | `Tend-T0`；D=0 不包含任何额外生成前向 |

D=0 终端只显示三模式的 TTFT、实际 prefill 与有效输入 mean +/- sample stddev，不显示 decode 列；原 TSV 可保留 decode 指标列，但值为 NA/`not_applicable`。`ppN`、采样数和实际前向数在输出中明确记录，不把一次采样叫作 decode 前向。

成功请求只 clear 一次，计入 `reset_ms`；`load_validation_ms` 仍计入整个 store get，包括内存分配、direct 读取、bounce buffer/状态拷贝、CRC 与身份/兼容性检查，不是纯设备 I/O 时间；`restore_ms` 只计导入。`host_validation_ms` 单独记录 host logits 拷贝和有限性检查；请求结束后的 golden 比较记录为 `parity_ms`。墙钟 TTFT/request 是 **instrumented/token-ready** 延迟，不扣除实际校验耗时来伪造生产延迟，也不是服务端/网络 TTFT。这里的状态恢复请求不等同于原 `llama-bench` 空历史随机 token 的独立 tg；原脚本保持不变。

底层 probe 的命名参数仍支持单 case 和 D>0，不再由此 shell 接口选择。必需项是 `--model PATH --model-sha256 HEX64 --runtime-id HEX64 --output-dir DIR`，其余可通过单独 `--help` 查看；无参数也只显示帮助，不加载模型。D>0 保留 D 个采样输出、D 次前向（含最后一个 token 的状态更新），预算为 P+S+D，刻意忽略 EOG。其 decode-forward TPS 为 `1000*D/decode_forward_ms`（同步前向，不含采样/校验）；首 token 后生成 TPS 为 `1000*(D-1)/(Tlast-Tfirst)`，D=1 时 NA；输出端到端 TPS 为 `1000*D/(Tlast-T0)`。不能把含主机操作的 `decode_ms` 当作纯 `decode_forward_ms`。直接调用也必须使用真实模型/runtime 指纹及获准的单卡，不能编造摘要。

## 输出、索引与统计

所有产物位于 `/data2/huangxin/qwen36_cache_experiments`。每次 runner 调用创建新的私有 `runs/sweep.XXXXXXXX/`；默认仅创建 Q4_0 的运行目录，两组都启用时创建两个独立目录，不覆盖或合并已有结果：

- `runtime-id`：本次固定的构建指纹。
- `metadata.tsv`：本次调用的实际参数，包含完整有序的 `pp_lengths`、`threads`、`cuda_visible_devices`、`repetitions_per_mode`、`warmup_per_mode`、`model_groups` 和 `runtime_id`。在第一个 case 开始前写入，因此失败或中断的 sweep 也保留尚未执行的计划长度；不同调用独立记录，不修改历史 metadata。
- `index.tsv`：追加 `status/model/pp/ctx/run_dir` 的 started/pass/failed 事件，供查看长任务正在处理哪项。
- **每个 sweep 根目录的 `summary.tsv`：该次运行的宽格式总表**。一行代表一个 model + pp + cache-mode；每个输入长度生成 3 行数据（recompute、DRAM、SSD），整张表另有一个表头。若启用两组，各组的表格不合并。
- `cases/BF16-ppN/`、`cases/Q4_0-ppN/`：私有空目录交给原生 probe，内部原始 TSV/cache 保留作审计，不在终端逐项展开。
- `logs/BF16-ppN.log`、`logs/Q4_0-ppN.log`：该项完整的原生 stdout/stderr，包含加载信息、进度和原始小表。日志位于 case 目录之外，不破坏 probe 的空目录要求。

每次 runner 调用只打印一次表头；每个启用的模型组单独输出一张表，按输入长度顺序追加每项三个已验证的行：`recompute`、`dram`、`io-uring-direct`。列为 `model | test | cache | n | TTFT mean +/- std | actual-prefill mean +/- std | effective-prompt mean +/- std`，使用固定列宽对齐，终端数值保留两位小数；TSV 仍保留原生数值精度。长 case 完成之前不会伪造结果行，可以查看 index 和该项日志；整轮成功后只报告总表/索引位置。

总表 TSV 分开存放数值字段：`model,test,cache,n,ttft_mean_ms,ttft_stddev_ms,actual_prefill_mean_tps,actual_prefill_stddev_tps,effective_prefill_mean_tps,effective_prefill_stddev_tps,ctx,run_dir`（实际以 tab 分隔）。ctx 来自 probe 的实际上下文；不重新计算均值，只转置原生结果。每次校验完整 case 后，用私有临时文件原子发布包含此前行和本项三行的总表。错误项不发布任何行，之前的已验证行保留，错误信息给出模型、pp 和日志路径；旧 long-format 历史文件不迁移或覆盖。

发布前检查原生进程退出、完整唯一的 3x11 summary 项、单位、样本数、状态、有限数值及标准差；D=0 的两个 decode 指标必须为 not_applicable/NA。metadata 中的完成/parity 标记、模型与 runtime 身份、GPU、pp、采样1/前向0、重复/预热和实际 context 也必须一致，并要求 `io_mode=io_uring+O_DIRECT`、`storage_backend=ssd`；旧 buffered 或缺失 I/O 元数据的结果不能进入 direct 总表。任何失败都停止，不能过滤异常行拼出成功表格。

内部 case 的 `results.tsv`、`continuations.tsv`、`summary.tsv` 保留原有列，文件模式标签改为 `io-uring-direct`；每个 case 的 `metadata.tsv` 记录读取/写入方式、I/O 参数以及显式 `threads`、GPU、测试长度、重复/预热次数；汇总前与 runner 的参数逐项核对。统计仍只包含正式 measurement，排除 golden/warmup；按逐 trial 速率求均值，标准差使用 n-1，n=1 时 std=0。

## 构建（一次完成 CUDA 库和 probe，不自动运行测试）

`build_qwen36_prefix_cache_cuda.sh` 直接配置并编译 llama.cpp CUDA 库，再编译 probe、生成 runtime 指纹，**无需先运行 `build_qwen36_cuda.sh`**。CUDA 构建目录仍为 `/home/huangxin/code_list/bitnet-moe-script/build_script/build_bin_qwen36_cuda`；目录不存在时自动创建并完整构建，后续调用按同一配置增量编译，不删除已有构建。

需要 CMake、Clang、CUDA Toolkit/NVCC（当前环境为 CUDA 12.8）和 liburing 开发头文件/库（Debian/Ubuntu：`liburing-dev`）。沿用 A100 `sm_80`、Release、native/AVX、OpenMP 关闭等编译参数；显式启用本地 shared ggml 与 CPU/CUDA 后端。只构建 `llama` 目标及其 `ggml`、`ggml-base`、`ggml-cpu`、`ggml-cuda` 依赖，不额外编译 CLI/benchmark/perplexity/quantize 工具。`LLAMA_BUILD_COMMON=ON` 让 CMake 在配置阶段生成清单需要的 `common/build-info.cpp`，不需要编译 common 库。

```bash
# [jobs] 为编译并行数，默认 8，不是推理线程数
bash /home/huangxin/code_list/bitnet-moe-script/build_script/build_qwen36_prefix_cache_cuda.sh 8
```

库构建成功后，Clang C++17 使用 nested `include` 与 `ggml/include`、显式 rpath，链接 llama/ggml/ggml-base 和 `-luring`，生成 `/data2/huangxin/qwen36_cache_experiments/bin/qwen36-prefix-snapshot` 与 `bin/runtime.sha256`。清单保留 probe、核心 C++ 源码/头文件、`run_sweep.py`、`build_qwen36_prefix_cache_cuda.sh`、上游头文件、CUDA 构建信息和共享库，以及按运行入口库路径由 `ldd` 解析的 liburing 路径与其真实文件路径。**不包含用于调参的 `benchmark_qwen36_prefix_cache_cuda.sh`。** 构建脚本、runner、核心源码、probe 或库发生变化仍需重建；源码版本、manifest、模型身份及每个 case 前后的严格校验均保留。从旧规则升级时需运行一次构建脚本生成新清单，之后仅修改调参入口不再需要重建。

配置、库编译或 probe 链接失败都会立即停止，不继续生成成功清单。不同时运行两个脚本操作同一 CUDA 构建目录，并确保所在分区空间充足。脚本不编译或运行测试程序、不创建测试目录、不加载模型，也不自动启动 benchmark。运行端仍单独要求匹配的 llama.cpp revision `373971d12e8bf094fac91667e42d732b5051f3bd`、干净的 tracked 源码和有效 runtime/model 身份。实际执行缓存读取还需要 Linux 内核/安全策略允许 io_uring，且快照所在文件系统支持 `O_DIRECT` 和 4096 字节对齐读取。

无模型验证是独立步骤：可单独编译运行 `test_snapshot_store.cpp`（也须链接 `-luring`），或通过 `python3 -I -B test_run_sweep.py [fixture_parent]` 运行汇总器测试。前者覆盖存储/假状态恢复、统计和 D=0/D=1/D=8 回归；后者用合成 TSV 与小文件检查单模型 12 次调用/36 行、双模型兼容路径 24 次调用/72 行、分开调用的目录隔离、日志捕获、身份及失败行为，包括未选模型不存在和拒绝旧 buffered I/O 元数据。测试夹具含 `pp262144` 边界，不改变 shell 中的测试长度列表。测试产物应放在 `/data2/huangxin/qwen36_cache_experiments/tests`，它们不是 GPU benchmark 数据，也不属于默认构建流程。

```bash
# 独立 host 测试；不链接 llama/CUDA，不加载模型
mkdir -p /data2/huangxin/qwen36_cache_experiments/tests
clang++ -std=c++17 -O2 -pthread \
  /home/huangxin/code_list/bitnet-moe-script/benchmark_script/qwen36_prefix_cache_cuda/test_snapshot_store.cpp \
  -luring -o /data2/huangxin/qwen36_cache_experiments/tests/test_snapshot_store
/data2/huangxin/qwen36_cache_experiments/tests/test_snapshot_store
python3 -I -B /home/huangxin/code_list/bitnet-moe-script/benchmark_script/qwen36_prefix_cache_cuda/test_run_sweep.py \
  /data2/huangxin/qwen36_cache_experiments/tests
```

构建完成只表示 probe 编译和清单生成成功；模型推理由 benchmark 入口另行启动，历史实验记录保持不变。

## 兼容性、容量和 io-uring-direct 边界

快照身份绑定模型内容 SHA-256、runtime 指纹、后端/设备与推理配置（context、batch、线程、KV/recurrent 类型、attention/RoPE/sequence）、准确 token 和位置。不跨模型、BF16/Q4_0、构建或 CPU/GPU 配置复用。文件有版本、受限长度及 CRC64；CRC 用于偶发损坏检查，不防恶意篡改，不接受不可信快照。

默认 `--max-blob-mib 8192 --max-store-mib 16384 --disk-reserve-mib 16384`：单个编码记录最多 **8 GiB**（含 metadata/token），DRAM store 和文件 store **各最多 16 GiB**，文件写入后至少保留 **16 GiB** 空闲。计入 staging/临时文件开销，检查后才分配/发布；使用同目录独占临时文件、完整写入、fsync、原子发布及目录同步。不自动扩容、淘汰或删除用户数据。每个 case 的缓存目录保留，因此整个 sweep 的磁盘占用会累积，16 GiB 不是所有 case 合计的全局容量上限。

**io-uring-direct** 使用 liburing 提交 `O_DIRECT` 读取：每批最多 **8** 个并发 **1 MiB** 请求，offset、请求长度与 bounce buffer 按 **4096 字节**对齐；bounce buffer 总量有界（最多 8 MiB），完成后拷贝有效记录字节。v1 envelope、CRC64、身份/长度/容量检查及 buffered 写入 + 文件 fsync + 原子发布 + 目录同步均不变，磁盘记录不新增对齐 padding。metadata 中 `storage_backend=ssd` 仅描述文件快照的存储配置，读取为 `io_mode=io_uring+O_DIRECT`，并记录 `io_queue_depth=8`、`io_chunk_bytes=1048576`、`io_alignment_bytes=4096` 和 `file_write_mode=buffered+fsync`。

`O_DIRECT` 绕过 OS page cache；warmup 不会把后续快照读取变成 buffered page-cache hit，但设备/控制器缓存和文件系统开销仍存在，不能称为原始 SSD 或冷设备延迟。没有全局 `drop_caches` 或设备缓存清扫；io_uring/`O_DIRECT` 不支持或读取失败就报错，**没有 buffered fallback**。`load_validation_ms` 仍包含分配、拷贝和 CRC，export/admission/fsync 的一次性成本虽不进入 hit TPS，仍须结合 metadata 分析，不能忽略准备成本。

本次不启动 sweep。未来真正执行前应先对新版做获准的小规模模型 parity；长上下文还必须服从模型实际/补齐窗口、显存和存储额度，不能因既往小样本通过就宣称 256K 可运行。未来 CPU 实验需要独立配置/身份和 parity，不能宣称快照跨后端兼容。

数据集 symlink `/home/huangxin/code_list/cc-traces-weka-062126` 指向 `/data3/huangxin/cc-traces`。`hash_ids` 是会话本地 **64-token block hash**，没有文本或词表 token ID，不能当作 Qwen token；后续只可先做明确标注的 structural/synthetic replay，保留会话命名空间、有序前缀与嵌套 subagent 时序。代理长度 `in` 可能高估约 260K，大量请求超过 256K，需另定超长请求/输出策略，不能静默截断。**0.9004 structural hit/reuse 不是实测 cache hit** 或性能收益。本阶段不扫描全量数据，不做 trace replay。
