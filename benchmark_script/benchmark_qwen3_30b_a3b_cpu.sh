#!/bin/bash
# Qwen3-30B-A3B: F16 + Q4_0 CPU Benchmark
# MoE architecture: ~31B total params, ~3B active
#
# Usage: bash benchmark_qwen3_30b_a3b_cpu.sh [threads] [numa_node]
# Examples:
#   bash benchmark_qwen3_30b_a3b_cpu.sh 8
#   bash benchmark_qwen3_30b_a3b_cpu.sh 8 0

set -e

MODEL_DIR="/data3/huangxin/model_list/Qwen3-30B-A3B"
F16_MODEL="${MODEL_DIR}/Qwen3-30B-A3B-f16.gguf"
Q4_0_MODEL="${MODEL_DIR}/Qwen3-30B-A3B-Q4_0.gguf"
BENCH="/home/huangxin/code_list/bitnet-moe-script/build_script/build_bin_upstream/bin/llama-bench"
QUANTIZE="/home/huangxin/code_list/bitnet-moe-script/build_script/build_bin_upstream/bin/llama-quantize"
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
echo "  Qwen3-30B-A3B: F16 + Q4_0 Benchmark (upstream llama.cpp)"
echo "  Architecture: MoE (~31B total, ~3B active)"
echo "  Threads: $THREADS"
[ -n "$NUMA_NODE" ] && echo "  CPU Affinity: NUMA node $NUMA_NODE"
echo "========================================================"
echo

# Check F16 model exists
if [ ! -f "$F16_MODEL" ]; then
    echo "ERROR: Model not found: $F16_MODEL"
    echo "Run convert_hf_to_gguf.py first."
    exit 1
fi

# Check bench binary
if [ ! -f "$BENCH" ]; then
    echo "ERROR: llama-bench not found: $BENCH"
    echo "Run build_script/build_upstream_llama_cpp.sh first."
    exit 1
fi

# Quantize to Q4_0 if not exists
if [ ! -f "$Q4_0_MODEL" ]; then
    echo "--- Quantizing F16 -> Q4_0 ---"
    if [ ! -f "$QUANTIZE" ]; then
        echo "ERROR: llama-quantize not found: $QUANTIZE"
        exit 1
    fi
    $QUANTIZE "$F16_MODEL" "$Q4_0_MODEL" Q4_0
    echo
fi

echo "--- Model Sizes ---"
echo "  F16:  $(du -h $F16_MODEL | cut -f1)"
echo "  Q4_0: $(du -h $Q4_0_MODEL | cut -f1)"
echo

echo "--- F16 Benchmark ---"
$TASKSET $BENCH -m $F16_MODEL -t $THREADS -p 128,256,512,1024 -n 16,32,64,128,256 -r 3 -ngl 0

echo
echo "--- Q4_0 Benchmark ---"
$TASKSET $BENCH -m $Q4_0_MODEL -t $THREADS -p 128,256,512,1024 -n 16,32,64,128,256 -r 3 -ngl 0

echo
echo "Done."
