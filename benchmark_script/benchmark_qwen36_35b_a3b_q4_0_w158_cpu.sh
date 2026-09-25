#!/bin/bash
# Qwen3.6-35B-A3B: Q4_0_with_w158 CPU Benchmark
# Routed experts: native BitNet I2_S; other projections: Q4_0; head: Q6_K.
#
# Usage: bash benchmark_qwen36_35b_a3b_q4_0_w158_cpu.sh [threads] [numa_node]
# Examples:
#   bash benchmark_qwen36_35b_a3b_q4_0_w158_cpu.sh 8
#   bash benchmark_qwen36_35b_a3b_q4_0_w158_cpu.sh 8 0

set -e

MODEL_DIR="/data3/huangxin/model_list/Qwen3.6-35B-A3B"
BF16_MODEL="${MODEL_DIR}/Qwen3.6-35B-A3B-bf16.gguf"
Q4_0_MODEL="${MODEL_DIR}/Qwen3.6-35B-A3B-Q4_0.gguf"
W158_MODEL="${MODEL_DIR}/Qwen3.6-35B-A3B-Q4_0-W158-NativeI2S.gguf"
BENCH="/home/huangxin/code_list/bitnet-moe-script/build_script/build_bin_qwen36_q4_0_w158_native_i2s/bin/llama-bench"
THREADS=${1:-4}
NUMA_NODE=${2:-""}

if [[ $# -gt 2 || ! $THREADS =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: Usage: $0 [positive threads] [numa_node]"
    exit 1
fi
if [[ -n $NUMA_NODE && $NUMA_NODE != all && ! $NUMA_NODE =~ ^[0-9]+$ ]]; then
    echo "ERROR: numa_node must be a node number or all."
    exit 1
fi
if [[ ${GGML_BITNET_I2S_TRACE:-0} != 0 || ${GGML_BITNET_I2S_VERIFY:-0} != 0 ]]; then
    echo "ERROR: Unset GGML_BITNET_I2S_TRACE and GGML_BITNET_I2S_VERIFY before benchmarking."
    exit 1
fi

# Build CPU affinity prefix
TASKSET=()
if [ "$NUMA_NODE" = "all" ]; then
    TASKSET=(numactl --interleave=0,1)
    echo "CPU Pinning: ALL NUMA nodes (interleaved memory)"
elif [ -n "$NUMA_NODE" ]; then
    TASKSET=(numactl "--cpunodebind=$NUMA_NODE" "--membind=$NUMA_NODE")
    echo "CPU Pinning: NUMA node $NUMA_NODE"
fi

echo "========================================================"
echo "  Qwen3.6-35B-A3B: Q4_0_with_w158 Benchmark (native I2_S)"
echo "  Architecture: MoE (~35B total, ~3B active)"
echo "  Threads: $THREADS"
[ -n "$NUMA_NODE" ] && echo "  CPU Affinity: NUMA node $NUMA_NODE"
echo "========================================================"
echo

# Check native model exists
if [ ! -f "$W158_MODEL" ]; then
    echo "ERROR: Model not found: $W158_MODEL"
    echo "Run generate_model_script/generate_qwen36_35b_a3b_q4_0_w158_model.sh first."
    exit 1
fi

# Check native benchmark binary
if [ ! -x "$BENCH" ]; then
    echo "ERROR: llama-bench not found or not executable: $BENCH"
    echo "Run build_script/build_qwen36_q4_0_w158_cpu.sh first."
    exit 1
fi

echo "--- Model Size ---"
echo "  Q4_0_with_w158: $(du -h "$W158_MODEL" | cut -f1)"
echo

# Optional baselines: uncomment to compare all three models with the same binary.
# echo "--- BF16 Benchmark ---"
# "${TASKSET[@]}" "$BENCH" -m "$BF16_MODEL" -t "$THREADS" -p 128,256,512,1024 -n 16,32,64,128,256 -r 3 -ngl 0 --progress
# echo "--- Q4_0 Benchmark ---"
# "${TASKSET[@]}" "$BENCH" -m "$Q4_0_MODEL" -t "$THREADS" -p 128,256,512,1024 -n 16,32,64,128,256 -r 3 -ngl 0 --progress

echo "--- Q4_0_with_w158 Benchmark ---"
echo "Each result is printed after warmup and 3 repetitions; progress is shown below."
"${TASKSET[@]}" "$BENCH" -m "$W158_MODEL" -t "$THREADS" -p 128,256,512,1024 -n 16,32,64,128,256 -r 3 -ngl 0 --progress

echo
echo "Done."

# 4 threads
# numactl --physcpubind=0-3 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_35b_a3b_q4_0_w158_cpu.sh 4

# 8 threads
# numactl --physcpubind=0-7 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_35b_a3b_q4_0_w158_cpu.sh 8

# 12 threads
# numactl --physcpubind=0-11 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_35b_a3b_q4_0_w158_cpu.sh 12

# 16 threads
# numactl --physcpubind=0-15 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_35b_a3b_q4_0_w158_cpu.sh 16

# 32 threads
# numactl --physcpubind=0-31 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_35b_a3b_q4_0_w158_cpu.sh 32
