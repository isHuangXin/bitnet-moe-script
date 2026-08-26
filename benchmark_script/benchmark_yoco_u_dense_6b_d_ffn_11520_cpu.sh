#!/bin/bash
# YOCO-U-Dense-6B (d_ffn=11520): F16 vs I2_S CPU Benchmark
# YOCO-U universal loop: 20 stored layers (10 self + 10 cross), self-decoder looped T=3
# ~3.5B stored params, ~7B effective compute, Dense SwiGLU FFN (no MoE)
#
# Usage: bash benchmark_yoco_u_dense_6b_d_ffn_11520_cpu.sh [threads] [numa_node]
# Examples:
#   bash benchmark_yoco_u_dense_6b_d_ffn_11520_cpu.sh 8
#   bash benchmark_yoco_u_dense_6b_d_ffn_11520_cpu.sh 8 0

set -e

MODEL_DIR="/data3/huangxin/model_list/yoco-u-dense-6b-d_ffn-11520"
F16_MODEL="${MODEL_DIR}/yoco-u-dense-6b-d_ffn-11520-bitnet-f16/ggml-model-f16.gguf"
I2S_MODEL="${MODEL_DIR}/yoco-u-dense-6b-d_ffn-11520-bitnet-i2s/ggml-model-i2_s.gguf"
BENCH="/home/huangxin/code_list/bitnet-moe-script/build_script/build_bin_yoco_moe/bin/llama-bench"
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
echo "  YOCO-U-Dense-6B (d_ffn=11520): F16 vs I2_S Benchmark"
echo "  Config: 20 stored layers (10 self + 10 cross)"
echo "  d_model=3072, head=32, cross_head=32, kv_head=8"
echo "  head_dim=128, diff_v3=True, yoco_window=512"
echo "  Dense FFN (SwiGLU): intermediate_size=11520"
echo "  yoco_u_iters=3 (self-decoder looped T=3)"
echo "  Stored params: ~3.5B, Effective compute: ~7B"
echo "  Threads: $THREADS"
[ -n "$NUMA_NODE" ] && echo "  CPU Affinity: NUMA node $NUMA_NODE"
echo "========================================================"
echo

# Check models exist
for m in "$F16_MODEL" "$I2S_MODEL"; do
    if [ ! -f "$m" ]; then
        echo "ERROR: Model not found: $m"
        echo "Run generate_yoco_u_dense_6b_d_ffn_11520_models.py first."
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
