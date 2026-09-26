#!/bin/bash
# Qwen3.6-35B-A3B: BF16 + Q4_0 + BitNet I2_S CPU Benchmark
# Same binary and parameters for all three models; 5 repetitions with warmup.
#
# Usage: bash benchmark_qwen36_bitnet_cpu.sh [threads] [numa_node]
# Examples:
#   bash benchmark_qwen36_bitnet_cpu.sh 8
#   bash benchmark_qwen36_bitnet_cpu.sh 8 0

set -e

MODEL_DIR="/data3/huangxin/model_list/Qwen3.6-35B-A3B"
BF16_MODEL="${MODEL_DIR}/Qwen3.6-35B-A3B-bf16.gguf"
Q4_0_MODEL="${MODEL_DIR}/Qwen3.6-35B-A3B-Q4_0.gguf"
I2_S_MODEL="${MODEL_DIR}/Qwen3.6-35B-A3B-1.58bit.gguf"
BENCH="/home/huangxin/code_list/bitnet-moe-script/build_script/build_bin_qwen36_bitnet/bin/llama-bench"
THREADS=${1:-4}
NUMA_NODE=${2:-""}

if [[ $# -gt 2 || ! $THREADS =~ ^[1-9][0-9]*$ ||
      ( -n $NUMA_NODE && $NUMA_NODE != all && ! $NUMA_NODE =~ ^[0-9]+$ ) ]]; then
    echo "ERROR: Usage: $0 [positive threads] [numa_node|all]"
    exit 1
fi

# Build CPU affinity prefix; omit numa_node to preserve external binding.
TASKSET=()
if [ "$NUMA_NODE" = "all" ]; then
    TASKSET=(numactl --interleave=all)
    echo "CPU Pinning: ALL NUMA nodes (interleaved memory)"
elif [ -n "$NUMA_NODE" ]; then
    TASKSET=(numactl "--cpunodebind=$NUMA_NODE" "--membind=$NUMA_NODE")
    echo "CPU Pinning: NUMA node $NUMA_NODE"
fi

echo "========================================================"
echo "  Qwen3.6-35B-A3B: BF16 + Q4_0 + BitNet I2_S Benchmark"
echo "  Threads: $THREADS"
echo "========================================================"
echo

# Check models and benchmark binary.
for MODEL in "$BF16_MODEL" "$Q4_0_MODEL" "$I2_S_MODEL"; do
    if [ ! -s "$MODEL" ]; then
        echo "ERROR: Model missing or empty: $MODEL"
        exit 1
    fi
done
if [ ! -x "$BENCH" ]; then
    echo "ERROR: llama-bench not found or not executable: $BENCH"
    echo "Run build_script/build_qwen36_bitnet_cpu.sh first."
    exit 1
fi

echo "--- Model Sizes ---"
echo "  BF16: $(du -h "$BF16_MODEL" | cut -f1)"
echo "  Q4_0: $(du -h "$Q4_0_MODEL" | cut -f1)"
echo "  I2_S: $(du -h "$I2_S_MODEL" | cut -f1)"
echo

# echo "--- BF16 Benchmark ---"
# "${TASKSET[@]}" "$BENCH" -m "$BF16_MODEL" -t "$THREADS" -p 128,256,512,1024 -n 16,32,64,128,256 -r 5 -ngl 0 -b 2048 -ub 512 -ctk f16 -ctv f16 -fa auto

# echo
# echo "--- Q4_0 Benchmark ---"
# "${TASKSET[@]}" "$BENCH" -m "$Q4_0_MODEL" -t "$THREADS" -p 128,256,512,1024 -n 16,32,64,128,256 -r 5 -ngl 0 -b 2048 -ub 512 -ctk f16 -ctv f16 -fa auto

echo
echo "--- BitNet I2_S Benchmark ---"
"${TASKSET[@]}" "$BENCH" -m "$I2_S_MODEL" -t "$THREADS" -p 128,256,512,1024 -n 16,32,64,128,256 -r 5 -ngl 0 -b 2048 -ub 512 -ctk f16 -ctv f16 -fa auto

echo
echo "Done."

# 4线程
# numactl --physcpubind=0-3 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_bitnet_cpu.sh 4

# 8线程
# numactl --physcpubind=0-7 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_bitnet_cpu.sh 8

# 12线程
# numactl --physcpubind=0-11 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_bitnet_cpu.sh 12

# 16线程
# numactl --physcpubind=0-15 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_bitnet_cpu.sh 16

# 32线程
# numactl --physcpubind=0-31 --membind=0 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_bitnet_cpu.sh 32
