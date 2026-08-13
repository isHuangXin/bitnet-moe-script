#!/bin/bash
# YOCO-MoE-30B-A3B-V3: F16 vs I2_S CPU Benchmark
# Standard YOCO (no universal loop): 20 layers (10 self + 10 cross)
# 31.77B total params, ~3.45B active (top-8/128 experts)
#
# Usage: bash benchmark_yoco_moe_30b_a3b_v3_cpu.sh [threads] [numa_node]
# Examples:
#   bash benchmark_yoco_moe_30b_a3b_v3_cpu.sh 8
#   bash benchmark_yoco_moe_30b_a3b_v3_cpu.sh 8 0

set -e

MODEL_DIR="/home/azureuser/models/yoco-moe-models/yoco-moe-30b-a3b-v3"
F16_MODEL="${MODEL_DIR}/yoco-moe-30b-a3b-v3-bitnet-f16/ggml-model-f16.gguf"
I2S_MODEL="${MODEL_DIR}/yoco-moe-30b-a3b-v3-bitnet-i2s/ggml-model-i2_s.gguf"
BENCH="/home/azureuser/huangxin/code_list/bitnet-moe-script/build_script/build_bin_yoco_moe/bin/llama-bench"
THREADS=${1:-4}
NUMA_NODE=${2:-""}

# Build CPU affinity prefix
TASKSET=""
if [ -n "$NUMA_NODE" ]; then
    TASKSET="numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE"
    CPUS=$(numactl --hardware 2>/dev/null | grep "node $NUMA_NODE cpus:" | sed 's/.*cpus: //')
    echo "CPU Pinning: NUMA node $NUMA_NODE (cores: $CPUS)"
fi

echo "========================================================"
echo "  YOCO-MoE-30B-A3B-V3: F16 vs I2_S Benchmark"
echo "  Config: 20 layers (10 self + 10 cross)"
echo "  d_model=3072, head=32, cross_head=32, kv_head=8"
echo "  head_dim=128, diff_v3=True, yoco_window=512"
echo "  MoE: 128 experts, top-8, ffn_dim=1280"
echo "  Total params: ~31.77B, Active: ~3.45B"
echo "  Threads: $THREADS"
[ -n "$NUMA_NODE" ] && echo "  CPU Affinity: NUMA node $NUMA_NODE"
echo "========================================================"
echo

# Check models exist
for m in "$F16_MODEL" "$I2S_MODEL"; do
    if [ ! -f "$m" ]; then
        echo "ERROR: Model not found: $m"
        echo "Run generate_yoco_moe_30b_a3b_v3_models.py first."
        exit 1
    fi
done

# Check bench binary
if [ ! -f "$BENCH" ]; then
    echo "ERROR: llama-bench not found: $BENCH"
    echo "Run build_script/build_yoco_u.sh first."
    exit 1
fi

echo "--- Model Sizes ---"
echo "  F16:  $(du -h $F16_MODEL | cut -f1)"
echo "  I2_S: $(du -h $I2S_MODEL | cut -f1)"
echo

echo "--- F16 Benchmark ---"
$TASKSET $BENCH -m $F16_MODEL -t $THREADS -p 128,256,512,1024 -n 16,32,64,128,256 -r 3 -ngl 0

echo
echo "--- I2_S Benchmark ---"
$TASKSET $BENCH -m $I2S_MODEL -t $THREADS -p 128,256,512,1024 -n 16,32,64,128,256 -r 3 -ngl 0 || echo "⚠ I2_S benchmark failed."

echo
echo "Done."
