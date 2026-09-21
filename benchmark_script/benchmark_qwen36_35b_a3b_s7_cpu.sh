#!/bin/bash
# Qwen3.6-35B-A3B: S7 CPU Benchmark
# Routed experts: w1.58 W2A8; embedding/head: INT8 + BF16 reciprocal scales.
# Full-attention Q/K/V/O (including Q gate rows) and GDN qkv/z/a/b/out: W2A16.
# Shared expert gate/up/down projections: W2A16; independent shared_expert_gate: BF16.
#
# Usage: bash benchmark_qwen36_35b_a3b_s7_cpu.sh [threads] [numa_node]
# Examples:
#   bash benchmark_qwen36_35b_a3b_s7_cpu.sh 8
#   bash benchmark_qwen36_35b_a3b_s7_cpu.sh 8 0

set -e

MODEL_DIR="/data3/huangxin/model_list/Qwen3.6-35B-A3B"
S7_MODEL="${MODEL_DIR}/Qwen3.6-35B-A3B-S7.gguf"
BENCH="/home/huangxin/code_list/bitnet-moe-script/build_script/build_bin_qwen36_s7/bin/llama-bench"
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
echo "  Qwen3.6-35B-A3B: S7 Benchmark (upstream + S7 CPU support)"
echo "  Architecture: MoE (~35B total, ~3B active)"
echo "  Threads: $THREADS"
[ -n "$NUMA_NODE" ] && echo "  CPU Affinity: NUMA node $NUMA_NODE"
echo "========================================================"
echo

# Check S7 model exists
if [ ! -f "$S7_MODEL" ]; then
    echo "ERROR: Model not found: $S7_MODEL"
    echo "Run convert_hf_to_gguf_s7.py first."
    exit 1
fi

# Check S7 benchmark binary
if [ ! -x "$BENCH" ]; then
    echo "ERROR: S7 llama-bench not found or not executable: $BENCH"
    echo "Run build_script/build_qwen36_s7_cpu.sh first."
    exit 1
fi

echo "--- Model Size ---"
echo "  S7: $(du -h "$S7_MODEL" | cut -f1)"
echo

echo "--- S7 Benchmark ---"
$TASKSET "$BENCH" -m "$S7_MODEL" -t "$THREADS" -p 128,256,512,1024 -n 16,32,64,128,256 -r 3 -ngl 0

echo
echo "Done."

# 4 physical cores on NUMA node 0
# numactl --physcpubind=0-3 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_35b_a3b_s7_cpu.sh 4

# 8 physical cores on NUMA node 0
# numactl --physcpubind=0-7 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_35b_a3b_s7_cpu.sh 8

# 12 physical cores on NUMA node 0
# numactl --physcpubind=0-11 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_35b_a3b_s7_cpu.sh 12

# 16 physical cores on NUMA node 0
# numactl --physcpubind=0-15 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_35b_a3b_s7_cpu.sh 16

# 32 physical cores on NUMA node 0
# numactl --physcpubind=0-31 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_35b_a3b_s7_cpu.sh 32
