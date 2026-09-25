#!/bin/bash
# Qwen3.6-35B-A3B: BF16 + Q4_0 CPU Benchmark
# MoE architecture: ~35B total params, ~3B active
#
# Usage: bash benchmark_qwen36_35b_a3b_cpu.sh [threads] [numa_node]
# Examples:
#   bash benchmark_qwen36_35b_a3b_cpu.sh 8
#   bash benchmark_qwen36_35b_a3b_cpu.sh 8 0

set -e

MODEL_DIR="/data3/huangxin/model_list/Qwen3.6-35B-A3B"
BF16_MODEL="${MODEL_DIR}/Qwen3.6-35B-A3B-bf16.gguf"
Q4_0_MODEL="${MODEL_DIR}/Qwen3.6-35B-A3B-Q4_0.gguf"
BENCH="/home/huangxin/code_list/bitnet-moe-script/build_script/build_bin_upstream/bin/llama-bench"
THREADS=${1:-4}
NUMA_NODE=${2:-""}

# Build CPU affinity prefix
TASKSET=""
if [ "$NUMA_NODE" = "all" ]; then
    TASKSET="numactl --interleave=0,1"
    echo "CPU Pinning: ALL NUMA nodes (interleaved memory)"
elif [ -n "$NUMA_NODE" ]; then
    TASKSET="numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE"
    CPUS=$(numactl --hardware 2>/dev/null | grep "node $NUMA_NODE cpus:" | sed 's/.*cpus: //')
    echo "CPU Pinning: NUMA node $NUMA_NODE (cores: $CPUS)"
fi

echo "========================================================"
echo "  Qwen3.6-35B-A3B: BF16 + Q4_0 Benchmark (upstream llama.cpp)"
echo "  Architecture: MoE (~35B total, ~3B active)"
echo "  Threads: $THREADS"
[ -n "$NUMA_NODE" ] && echo "  CPU Affinity: NUMA node $NUMA_NODE"
echo "========================================================"
echo

# Check BF16 model exists
if [ ! -f "$BF16_MODEL" ]; then
    echo "ERROR: Model not found: $BF16_MODEL"
    echo "Run upstream convert_hf_to_gguf.py with --outtype bf16 --no-mtp first."
    exit 1
fi

# Check Q4_0 model exists
if [ ! -f "$Q4_0_MODEL" ]; then
    echo "ERROR: Model not found: $Q4_0_MODEL"
    echo "Run llama-quantize to create the Q4_0 model first."
    exit 1
fi

# Check bench binary
if [ ! -x "$BENCH" ]; then
    echo "ERROR: llama-bench not found or not executable: $BENCH"
    echo "Run build_script/build_upstream_llama_cpp.sh first."
    exit 1
fi

echo "--- Model Sizes ---"
echo "  BF16: $(du -h "$BF16_MODEL" | cut -f1)"
echo "  Q4_0: $(du -h "$Q4_0_MODEL" | cut -f1)"
echo

echo "--- BF16 Benchmark ---"
$TASKSET "$BENCH" -m "$BF16_MODEL" -t "$THREADS" -p 128,256,512,1024 -n 16,32,64,128,256 -r 3 -ngl 0

echo
echo "--- Q4_0 Benchmark ---"
$TASKSET "$BENCH" -m "$Q4_0_MODEL" -t "$THREADS" -p 128,256,512,1024 -n 16,32,64,128,256 -r 3 -ngl 0

echo
echo "Done."

# 4线程
# numactl --physcpubind=0-3 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_35b_a3b_cpu.sh 4

# 8线程
# numactl --physcpubind=0-7 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_35b_a3b_cpu.sh 8

# 12线程
# numactl --physcpubind=0-11 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_35b_a3b_cpu.sh 12

# 16线程
# numactl --physcpubind=0-15 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_35b_a3b_cpu.sh 16

# 32线程
# numactl --physcpubind=0-31 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_35b_a3b_cpu.sh 32