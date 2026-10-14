# Qwen3.6 prefix snapshots: BF16 + Q4_0 unified prefill table

**本版只调整入口结构、日志捕获和总表汇总，不改变 C++ 推理/统计/存储。此次只编译与无模型检查，不自动哈希/加载真实模型或启动 GPU 实验。** 新的统一表格测试不代表运行过真实 24-case 扫描，也不能证明 GPU parity、性能、256K 或缓存加速；历史实验是否成功应以对应目录中的原始记录为准。

保留所有历史小样本与原 `benchmark_qwen36_cuda.sh`。例如历史 v0 Q4_0 `/data2/huangxin/qwen36_cache_experiments/runs/run.mXyP8m2O`（P127/S1/D8/context256，runtime-id `69ecfbe8abf64c63679cbb87160efd951458eb0c3a72f6366aa13821d5f081e3`）通过了当时版本的 DRAM/file parity，但固定顺序、单次、无匹配 warmup 的耗时不能证明稳态加速，也不能作为新版验证结论。旧文件不迁移、不改写、不补算成新版统计。

## 固定的两组 pp 测试

用户确认的是 **BF16、Q4_0 两组；pp 全部保留，decode 前向全部去掉**。wrapper 严格先 BF16 后 Q4_0，各按下列顺序执行 12 项，共 24 cases：

```text
128 256 512 1024 2048 4096 8192 16384 32768 65535 131077 262144
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

不把 65535/131077 改成 65536/131072，不缩短最后一项，不额外加入 decode case。`pp262144` 的完整输入已占满窗口，不能再追加任何 token 前向。`P+S+D <= context <= 262144` 及实际模型/补齐窗口检查仍然生效。

### 参考脚本风格的 shell 入口（保持四参数）

`benchmark_qwen36_prefix_snapshot.sh` 按 `benchmark_qwen36_cuda.sh` 的结构排列：模型/程序路径、参数、`Prefill_PROMPTS`、参数检查、配置横幅、文件检查、模型大小、一次统一调用和成功后的 `Done.`。runtime 校验、循环、日志和汇总由专用标准库工具 `run_sweep.py` 承担；不是改回普通 llama-bench，也不丢弃前缀恢复和正确性检查。

```bash
# NOT RUN: [threads] [gpu_id] [repetitions] [warmup]，默认 8 0 3 1
bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_prefix_snapshot.sh 8 0 3 1
```

**这是未来用户明确选择空闲卡后才运行的完整双模型 sweep，不是本次执行命令，也不再是单 case smoke。** repetitions/warmup 均为 `1..1000`。参数个数、正整数/溢出、GPU index 和重复/预热边界在模型哈希或创建运行产物前校验。旧的 `[threads] [gpu_id] [model_path] [prefix] [suffix] [decode] [ctx] [repetitions] [warmup]` shell 用法已不支持；多余参数或把模型路径放第三位会报错，不做旧接口自动识别，也没有 `--sweep`、自动跳过、重试、CPU fallback 或 `--dry-run`。

runner 为子进程设置 `CUDA_VISIBLE_DEVICES=$gpu_id`，只暴露一张卡，probe 内 main_gpu=0、ngl=99。每个模型只哈希一次，两个模型的准备工作在总表表头打印前完成；模型加载/哈希不计入请求耗时。运行前固定 `runtime-id`，每项前后检查 manifest、构建指纹、源码版本和模型身份，拒绝中途更换模型或混用构建，不自动重建。表头之后不再插入哈希提示、原生 probe 进度或各项小表。

扫描可能耗时**多小时**：每个 case 是独立 probe 进程，重新加载模型，建立 golden、准备快照，再进行 B 干扰、预热和重复比较。BF16 长上下文可能显存不足；DRAM、编码快照大小和 NVMe 容量也可能触发限制。不会为跑完 24 项而静默缩短输入、放宽容量或回退 CPU。任何失败立即停止，显示模型、pp 和 case 目录；已经完成的输出保留，不继续下一项。

## 快照范围与每个 case 的协议

这是跨请求的**完整前缀状态快照**，不是活跃 KV 分层分页、CPU offload 调度器或 Strata 集成。不改 snapshot_store 的接口/格式，不改 llama.cpp、不重编译 CUDA，不增加 LRU、服务或 trace replay。

- 单序列 ID 0；public `llama_state_seq_get_size/get_data/set_data` 使用完整 host 状态，flags=NONE，不用 PARTIAL_ONLY/ON_DEVICE。
- 包含 full-attention KV（F16）、GDN recurrent state 和 convolution history（F32），不只是 attention KV。
- 保存准确的已处理前缀 token 与位置。快照**不含 logits/sampler**；恢复后必须计算至少 1 个未处理 suffix token，再获得 logits 和首 token。
- 只复用完全相同的前缀，不支持 hybrid state 任意裁剪/rollback；较早分叉需恢复更早 checkpoint 后向前重算。
- 恢复前 clear；导入字节数错误/失败后再次 clear 并退出，不使用半恢复状态。

每个 case 都比较 recompute、DRAM、buffered-file，没有 mode selector。建立不参与统计的 golden recompute，DRAM/file **各准备一次** A 前缀快照，之后 warmup/measurement 只读，不重复 put/fsync。admission、prefix、export、store/fsync 成本单独记录在 metadata，不混入 hit trial。

每个请求前执行并同步不同的 B 干扰，放在请求计时之外。请求内 clear 后重算 A 或真正读取/恢复 A，计算同一 suffix，再采样首 token。A/B/A 排除残留 GPU 状态造成的假命中。先完成所有模式的等量 warmup（默认每模式 1 次），再 measurement（默认每模式 3 次）；两个阶段分别在第 r 轮按 `[recompute, dram, buffered-file][(r+s)%3]` 轮换次序。

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

成功请求只 clear 一次，计入 `reset_ms`；`load_validation_ms` 是 store get 与 CRC/兼容性检查，`restore_ms` 只计导入。`host_validation_ms` 单独记录 host logits 拷贝和有限性检查；请求结束后的 golden 比较记录为 `parity_ms`。墙钟 TTFT/request 是 **instrumented/token-ready** 延迟，不扣除实际校验耗时来伪造生产延迟，也不是服务端/网络 TTFT。这里的状态恢复请求不等同于原 `llama-bench` 空历史随机 token 的独立 tg；原脚本保持不变。

底层 probe 的命名参数仍支持单 case 和 D>0，不再由此 shell 接口选择。必需项是 `--model PATH --model-sha256 HEX64 --runtime-id HEX64 --output-dir DIR`，其余可通过单独 `--help` 查看；无参数也只显示帮助，不加载模型。D>0 保留 D 个采样输出、D 次前向（含最后一个 token 的状态更新），预算为 P+S+D，刻意忽略 EOG。其 decode-forward TPS 为 `1000*D/decode_forward_ms`（同步前向，不含采样/校验）；首 token 后生成 TPS 为 `1000*(D-1)/(Tlast-Tfirst)`，D=1 时 NA；输出端到端 TPS 为 `1000*D/(Tlast-T0)`。不能把含主机操作的 `decode_ms` 当作纯 `decode_forward_ms`。直接调用也必须使用真实模型/runtime 指纹及获准的单卡，不能编造摘要。

## 输出、索引与统计

所有产物位于 `/data2/huangxin/qwen36_cache_experiments`。每次执行创建新的私有 `runs/sweep.XXXXXXXX/`：

- `runtime-id`：本次固定的构建指纹。
- `index.tsv`：追加 `status/model/pp/ctx/run_dir` 的 started/pass/failed 事件，供查看长任务正在处理哪项。
- **根目录 `summary.tsv`：唯一面向用户的宽格式总表**。一行代表一个 model + pp + cache-mode，完整扫描为 72 行数据加一个表头，不再是逐指标长表。
- `cases/BF16-ppN/`、`cases/Q4_0-ppN/`：私有空目录交给原生 probe，内部原始 TSV/cache 保留作审计，不在终端逐项展开。
- `logs/BF16-ppN.log`、`logs/Q4_0-ppN.log`：该项完整的原生 stdout/stderr，包含加载信息、进度和原始小表。日志位于 case 目录之外，不破坏 probe 的空目录要求。

终端只打印一次总表表头，按模型和输入长度顺序追加每项三个已验证的行：`recompute`、`dram`、`buffered-file`。列为 `model | test | cache | n | TTFT mean +/- std | actual-prefill mean +/- std | effective-prompt mean +/- std`，使用固定列宽对齐，终端数值保留两位小数；TSV 仍保留原生数值精度。长 case 完成之前不会伪造结果行，可以查看 index 和该项日志；整轮成功后只报告总表/索引位置。

总表 TSV 分开存放数值字段：`model,test,cache,n,ttft_mean_ms,ttft_stddev_ms,actual_prefill_mean_tps,actual_prefill_stddev_tps,effective_prefill_mean_tps,effective_prefill_stddev_tps,ctx,run_dir`（实际以 tab 分隔）。ctx 来自 probe 的实际上下文；不重新计算均值，只转置原生结果。每次校验完整 case 后，用私有临时文件原子发布包含此前行和本项三行的总表。错误项不发布任何行，之前的已验证行保留，错误信息给出模型、pp 和日志路径；旧 long-format 历史文件不迁移或覆盖。

发布前检查原生进程退出、完整唯一的 3x11 summary 项、单位、样本数、状态、有限数值及标准差；D=0 的两个 decode 指标必须为 not_applicable/NA。metadata 中的完成/parity 标记、模型与 runtime 身份、GPU、pp、采样1/前向0、重复/预热和实际 context 也必须一致。任何失败都停止，不能过滤异常行拼出成功表格。

内部 case 的 `results.tsv`、`continuations.tsv`、`metadata.tsv`、`summary.tsv` 内容不变。统计仍只包含正式 measurement，排除 golden/warmup；按逐 trial 速率求均值，标准差使用 n-1，n=1 时 std=0。

## 构建（只编译 probe，不自动运行测试）

复用 `/home/huangxin/code_list/bitnet-moe-script/build_script/build_bin_qwen36_cuda/bin` 中已经编译的 CUDA 共享库。Clang C++17 使用 nested `include` 与 `ggml/include`、显式 rpath，链接 llama/ggml/ggml-base 及现有传递后端依赖；不重新运行 nvcc/CMake 构建 llama.cpp。

```bash
# 只编译 probe 并生成运行入口必需的指纹清单
bash /home/huangxin/code_list/bitnet-moe-script/build_script/build_qwen36_prefix_snapshot.sh
```

当前构建脚本只生成 `/data2/huangxin/qwen36_cache_experiments/bin/qwen36-prefix-snapshot` 和 `bin/runtime.sha256`。清单记录 probe、运行源码/头文件、两个 snapshot shell 入口、`run_sweep.py`、上游头文件、CUDA 构建信息和共享库。入口、runner、probe 或这些依赖修改后需显式重建，否则运行端会拒绝旧指纹。

构建脚本不再执行 Git/CMake 配置检查，不编译或运行测试程序，不创建测试目录，也不加载模型。运行端仍单独要求匹配的 llama.cpp revision `373971d12e8bf094fac91667e42d732b5051f3bd`、干净的 tracked 源码和有效 runtime/model 身份；简化构建并不绕过这些运行时要求。

无模型验证是独立步骤：可单独编译运行 `test_snapshot_store.cpp`，或通过 `python3 -I -B test_run_sweep.py [fixture_parent]` 运行汇总器测试。前者覆盖存储/假状态恢复、统计和 D=0/D=1/D=8 回归；后者用合成 TSV 与小文件检查 24 次调用、72 行总表、日志捕获、身份及失败行为。测试产物应放在 `/data2/huangxin/qwen36_cache_experiments/tests`，它们不是 GPU benchmark 数据，也不属于默认构建流程。

构建完成只表示 probe 编译和清单生成成功；模型推理由 benchmark 入口另行启动，历史实验记录保持不变。

## 兼容性、容量和 buffered-file 限制

快照身份绑定模型内容 SHA-256、runtime 指纹、后端/设备与推理配置（context、batch、线程、KV/recurrent 类型、attention/RoPE/sequence）、准确 token 和位置。不跨模型、BF16/Q4_0、构建或 CPU/GPU 配置复用。文件有版本、受限长度及 CRC64；CRC 用于偶发损坏检查，不防恶意篡改，不接受不可信快照。

默认 `--max-blob-mib 8192 --max-store-mib 16384 --disk-reserve-mib 16384`：单个编码记录最多 **8 GiB**（含 metadata/token），DRAM store 和文件 store **各最多 16 GiB**，文件写入后至少保留 **16 GiB** 空闲。计入 staging/临时文件开销，检查后才分配/发布；使用同目录独占临时文件、完整写入、fsync、原子发布及目录同步。不自动扩容、淘汰或删除用户数据。每个 case 的缓存目录保留，因此整个 sweep 的磁盘占用会累积，16 GiB 不是所有 case 合计的全局容量上限。

**buffered-file** 在 NVMe 上的读取，特别是 warmup 后，可能命中 OS page cache，不能叫物理 SSD 延迟。没有全局 `drop_caches`、缓存清扫或 direct I/O；冷 I/O 需另立实验。export/admission/fsync 的一次性成本虽不进入 hit TPS，仍须结合 metadata 分析，不能忽略准备成本。

本次不启动 sweep。未来真正执行前应先对新版做获准的小规模模型 parity；长上下文还必须服从模型实际/补齐窗口、显存和存储额度，不能因既往小样本通过就宣称 256K 可运行。未来 CPU 实验需要独立配置/身份和 parity，不能宣称快照跨后端兼容。

数据集 symlink `/home/huangxin/code_list/cc-traces-weka-062126` 指向 `/data3/huangxin/cc-traces`。`hash_ids` 是会话本地 **64-token block hash**，没有文本或词表 token ID，不能当作 Qwen token；后续只可先做明确标注的 structural/synthetic replay，保留会话命名空间、有序前缀与嵌套 subagent 时序。代理长度 `in` 可能高估约 260K，大量请求超过 256K，需另定超长请求/输出策略，不能静默截断。**0.9004 structural hit/reuse 不是实测 cache hit** 或性能收益。本阶段不扫描全量数据，不做 trace replay。
