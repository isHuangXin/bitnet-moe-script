# Qwen3.6-35B-A3B native I2S CPU workflow

The benchmark label **Q4_0_with_w158** now means Q4_0 non-routed-experts plus native BitNet I2_S routed experts, stored in `Qwen3.6-35B-A3B-Q4_0-W158-NativeI2S.gguf`. It is not the old S2/Q8 hybrid. The archived `Qwen3.6-35B-A3B-Q4_0-W158.gguf`, its logs, and the BF16/S2/Q4_0 controls are retained unchanged.

**The old TG128 = 14.06 tokens/s result is NOT native I2S performance.** It came from the legacy S2 unpack/Q8 expert path. Neither those numbers nor the old four-model comparison belong in a native result table. Native arithmetic also differs from S2, so a future speed difference cannot be attributed solely to swapping kernels.

## Three commands

Run these scripts in order. The comparison uses eight physical cores on NUMA node 0; this host's `lscpu -e=CPU,CORE,SOCKET,NODE` identifies CPUs 0-7 as distinct physical cores. Recheck that mapping on another host. Do not pass the optional NUMA positional argument when using external binding. Before timing, unset `GGML_BITNET_I2S_TRACE` and `GGML_BITNET_I2S_VERIFY` (explicit `0` is also accepted); the benchmark rejects other values rather than mixing instrumented and normal timings.

```bash
bash /home/huangxin/code_list/bitnet-moe-script/build_script/build_qwen36_q4_0_w158_cpu.sh 8
bash /home/huangxin/code_list/bitnet-moe-script/generate_model_script/generate_qwen36_35b_a3b_q4_0_w158_model.sh 8
numactl --physcpubind=0-7 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_35b_a3b_q4_0_w158_cpu.sh 8 --quick --compare
```

The local native build, generation and measurements have now been executed; see the dated observations below. The generated native file already exists, so repeating the generation command will safely refuse to overwrite it. The benchmark command can be rerun directly. The sampled chat-quality check did not pass.

After the quick comparison and runtime/quality checks, the complete native-only length sweep is:

```bash
numactl --physcpubind=0-7 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_35b_a3b_q4_0_w158_cpu.sh 8 --full
```

## Exact native format contract

Every key below has the prefix `qwen35moe.s2.`. Sharing that namespace does not make the new format strict S2.

| Suffix | Required value |
| --- | --- |
| `version` | `public_hf_w158_v2` |
| `setting` | `Q4_0_W158_NATIVE_I2S` |
| `layout` | `bitnet_i2s_k128_v1` |
| `activation_profile` | `bitnet_i8_s_row_f32_v1` |
| `probability_placement` | `after_fc2` |
| `group_size` | `128` (UINT32) |
| `dense_profile` | `upstream_q4_0_v1` |
| `router_profile` | `upstream_f32` |
| `expert_kernel` | `bitnet_i2s_native` |
| `logical_parameters` | Recomputed logical weight count, excluding auxiliary scales (UINT64) |
| `parameter_bytes` | Recomputed stored tensor payload, including scales (UINT64) |

All other listed values are strings. Unknown or mixed native contract fields are rejected. This hybrid does not advertise an ordinary upstream quantization file type. Native loader support must be explicitly compiled; unsupported builds must fail, never silently fall back to Q8.

- Non-routed-expert tensors are copied byte-for-byte from standard Q4_0: shared experts, attention/GDN main projections and embedding retain Q4_0; output head retains Q6_K; router, norms, convolution and small parameters retain upstream F32 and conversion semantics. Normal CPU repacking remains available for these tensors.
- Routed expert values come only from the original BF16-derived strict S2 artifact, never by quantizing already-lossy Q4_0 experts again. Each ternary value and BF16 group-128 scale is preserved exactly.
- S2 packing is byteplane-reordered: lane `k` within a 16-value word uses shift `(k % 4) * 8 + (k // 4) * 2`. It is not adjacent-lane packed2. The offline converter decodes that layout before native packing.
- Native x86 I2S stores `u = q + 1` for logical `q` in `{-1, 0, 1}`. Within each 128-value group, logical index `j` goes to `byte[j % 32]` at shift `6 - 2 * (j // 32)`. Native code `3` is reserved; S2 code `2` was reserved.
- The I32 tensor is an opaque packed-byte container with unchanged stored shape, not a standard GGML I2_S tensor. BF16 scales stay in their existing `.s2_scale` tensors; there is no appended single-matrix scale. Native runtime kernels consume the packed expert bytes directly, without a decoded/Q8 weight buffer.

## Arithmetic is not S2-equivalent

The native contract uses full-row FP32 absmax, signed INT8 activation codes and `roundf`, with explicit zero-row handling. Gate/up projections, SiLU, their product and down output remain FP32. Routing probabilities are multiplied **after down/fc2**, with the standard selected-slot aggregation order. It does not retain S2's BF16 intermediate rounding, probability-before-down-activation-quantization, or expert-ID-sorted aggregation.

Weight scales still change every 128 elements. For each group, the native unsigned-code dot is corrected and scaled as `(native_dot - sum(activation_codes_in_group)) / activation_multiplier * weight_scale[row, group]`. The activation multiplier comes from the complete logical K row, not independently from each 128-element group.

Lossless expert weight repacking therefore does **not** imply arithmetic-equivalent outputs, accuracy equivalence, or pure kernel-only speed attribution versus S2. Quality and speed must be measured independently.

## Generation and provenance

- Generator CLI: `[threads]`, default 8. `MODEL_DIR` defaults to `/data3/huangxin/model_list/Qwen3.6-35B-A3B`; `PYTHON` selects one interpreter executable with NumPy/gguf dependencies. Existing `Qwen3.6-35B-A3B-bf16.gguf`, `Qwen3.6-35B-A3B-S2.gguf`, its `.manifest.json`, and `Qwen3.6-35B-A3B-Q4_0.gguf` are required. Missing controls cause failure, not regeneration.
- Only `/data3/huangxin/model_list/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-Q4_0-W158-NativeI2S.gguf` and its `.manifest.json` are published by default. Existing files or symlinks at either destination are rejected. Staging and non-overwriting hard-link publication retain the manifest-last completion marker.
- The composer `/home/huangxin/code_list/BitNet/3rdparty/llama.cpp/convert_qwen36_q4_0_w158.py` defaults to `--expert-kernel native-i2s`. Explicit `--expert-kernel s2-q8` retains the legacy `Q4_0_W158` contract and raw S2 code layout for archival work with a fresh `--outfile`; the native generator/benchmark never selects it. The strict S2 auditor uses its unchanged, separate old contract.
- Full prerequisite audits are retained, including the composer's independent source audits. They check complete tensor inventory, dimensions, tokenizer identity, metadata, scales, reserved codes, manifest and tensor/file SHA256 values, and unchanged input identities. Provenance remains **caller-declared BF16 ancestry**: identity checks and local hashes do not prove a common checkpoint or independently verify the source revision.
- Repacking is bounded to 512 KiB of packed source per chunk, using a single packed expert tensor on disk as staging. It never materializes the whole model as plain weights. Space preflight includes the output and largest staging tensor.
- Complete output readback verifies every tensor hash, raw Q4 non-expert/scale byte preservation, exact metadata, and decoded ternary equality for every native expert value. Expert raw source/output bytes differ, so the manifest records `source_sha256` separately from `sha256`/`output_sha256` and names the lossless repack operation. It also records the final whole-GGUF hash.
- Generation logs use new unique directories under `generate_model_script/logs/qwen36_q4_0_w158_native_i2s/`. No archived model or log is deleted or overwritten.

## Build and benchmark controls

- Build CLI: `[jobs]`, default `JOBS` or 8. Defaults are parent source `/home/huangxin/code_list/BitNet`, llama.cpp source `/home/huangxin/code_list/BitNet/3rdparty/llama.cpp`, and output `build_script/build_bin_qwen36_q4_0_w158_native_i2s`. `BITNET_DIR`, `LLAMA_DIR`, `BUILD_DIR`, and `BUILD_LOG_DIR` override the relevant locations.
- The native build explicitly compiles the parent's `src/ggml-bitnet-mad.cpp` through the kernels-only interface: x86 AVX2, QK=128, native parallel width/config checked by the runtime. Enabling TL2 or renaming the old Q8 kernel is not this implementation. Unsupported configurations must fail clearly.
- Build settings include Release, Clang/Clang++, CPU-only, native CPU detection and CPU repacking on, OpenMP off. Only `llama-cli`, `llama-bench`, `llama-quantize`, and `llama-debug` are requested; no test target or `ctest`. Use a separate native build directory, not the archived Q8 build.
- Benchmark CLI: `[threads] [numa_node] [--quick|--full] [--compare]`; default 8 threads, full native-only. Flags may precede or follow positionals. Both modes use three repetitions, normal warmup, CPU-only `-ngl 0`, batch 2048, microbatch 512, f16 K/V cache and `-fa auto`.
- Full lengths: PP 128,256,512,1024; TG 16,32,64,128,256. Quick lengths: PP512 and TG128. **PP and TG are separate tests**, not one combined request.
- `--compare` runs **exactly BF16, Q4_0, Q4_0_with_w158**, sequentially with the same native-enabled binary and configuration. No fourth S2 model is benchmarked. BF16 is not FP16. The third label maps exclusively to the NativeI2S filename/setting/kernel. Baseline models must have ordinary metadata and must not enter the native expert path.
- Preflight checks exact native metadata types/values, `expert_kernel=bitnet_i2s_native`, and recomputed header parameter counts. Renaming the old Q8 file to the native filename does not bypass the contract check. Missing models, runtime failures, duplicate/combined/incomplete rows, inconsistent builds/settings, non-CPU rows and invalid timings fail the run; nothing is silently skipped.
- Omitted `numa_node` preserves external CPU affinity/memory policy. A numeric node applies `numactl --cpunodebind=N --membind=N`; `all` requests `--interleave=all` without pinning CPUs. Node binding alone does not select eight physical cores.
- `BUILD_DIR`, `LLAMA_DIR`, `MODEL_DIR`, `RESULTS_DIR`, and `PYTHON` override benchmark locations/interpreter. Only the NativeI2S model is required without `--compare`. `--help` requires no model or build. Tracing and verification are diagnostic-only; the timing wrapper rejects enabled flags rather than silently changing them.

## Evidence, summaries and acceptance

Build logs use unique directories under `build_script/logs/qwen36_q4_0_w158_native_i2s/`. They retain parent/child Git commits, dirty status/diffs, actual native source/header/config hashes, compiler versions, CMake configuration and executable hashes. A checkout commit alone is not proof of the dirty sources compiled.

Benchmark results use unique directories under `benchmark_script/results/qwen36_q4_0_w158_native_i2s/`. `RESULTS_DIR` overrides that parent. Per-model files retain untouched `*.stdout.jsonl`, raw `*.stderr.log`, and shell-quoted `*.command.sh`. Failures retain partial evidence and mark `metadata.json` failed, not successful.

`metadata.json` records binary/library hashes, command arrays, current source Git state, CMake cache identity, effective affinity, NUMA policy, CPU topology, GGUF header hashes/metadata/type counts/sizes, and copied manifest references. It records all `GGML_*`, `BITNET_*`, `OMP_*` and relevant affinity/library environment flags, including explicit trace/verify states. Model SHA256 references are labeled **not reverified**: benchmark preflight does not rehash model payloads or claim a fresh full integrity audit. The generator performs those audits.

`summary_pp.csv`, `summary_tg.csv`, and `summary.md` contain only completed rows, with mean tokens/s, standard deviation, elapsed milliseconds and repetitions. Compare mode adds ratios against BF16 and Q4_0 only, at the same phase and length. No old results or estimated throughput are merged into these summaries.

The acceptance protocol below requires runtime observation, not just help/argument checks or finding a symbol. The dated record that follows states what was actually observed:

1. Run the native build/generator and complete its audits on the real model.
2. In separate diagnostic PP and TG CLI runs, prove nonzero actual native `ggml_vec_dot_i2_i8_s_1xN`, `ggml_vec_dot_i2_i8_s_Nx1`, and applicable `1x1` tail counts, with routed-expert Q8 calls zero. Record native source/config identity and any sampled integer-dot verification; do not claim unexercised boundary coverage.
3. Check mmap and non-mmap inference, finite logits, multi-token/slot/tail behavior, unsupported-kernel rejection and short generated text. Recheck the previous empty-answer behavior; a native kernel does not guarantee a quantization-quality fix. Finite-logit and short-generation observations do not replace formal perplexity/accuracy evaluation.
4. Disable tracing/verification before the three-model quick timing comparison and subsequent native full-length sweep. Report actual throughput and quality observations without assuming native is faster than Q4_0.

Measured native payload is **9.444590 GiB**; the complete GGUF file is **10,152,044,512 bytes**. This is not peak RSS. GGUF headers, loading, Q4 CPU repacking, f16 KV cache, recurrent state, activation scratch and scheduling add costs. Scope is text-only CPU inference, without vision, MTP or LoRA; smaller weights and short-text smoke checks are not an accuracy evaluation.

## Observed run: 2026-09-24

Eight physical cores (0-7), NUMA node 0, three repetitions, same native-enabled binary, TRACE=0 and VERIFY=0:

| Model | PP512 tokens/s | TG128 tokens/s |
| --- | ---: | ---: |
| BF16 | 48.18 | 9.34 |
| Q4_0 | 94.18 | 20.01 |
| Q4_0_with_w158, native I2S | 83.83 | 18.56 |

The native hybrid reached 0.8901 of Q4_0 PP throughput and 0.9275 of Q4_0 TG throughput. It did not outperform Q4_0 in this run. All nine native full-sweep lengths also completed; raw sample variability is retained.

Actual diagnostic PP/TG calls reached the parent 1xN/Nx1 functions and reported routed Q8 calls zero. Counts account for all routed matrix group dots. Integer-dot spotchecks on real inputs passed, both loading modes ran, and all 248,320 saved logits were finite. The 1x1 tail was not exercised by these model shapes.

**The sampled chat-quality check failed:** `What is the capital of France? Answer in one short sentence.` returned no answer for the native hybrid in both loading modes. BF16/Q4_0 controls answered Paris in the earlier control run. The cause remains unresolved; neither successful kernel execution nor speed measurements qualify the model's accuracy.

- [Three-model results and raw-evidence directory](benchmark_script/results/qwen36_q4_0_w158_native_i2s/20260923T234001Z-quick-6ecbx41m/summary.md)
- [Full native length sweep](benchmark_script/results/qwen36_q4_0_w158_native_i2s/20260923T234601Z-full-m27xcc69/summary.md)
- [Runtime verification report, call counters, commands and probes](benchmark_script/results/qwen36_q4_0_w158_native_i2s/runtime-20260924-NeDq90/VERIFICATION.md)
