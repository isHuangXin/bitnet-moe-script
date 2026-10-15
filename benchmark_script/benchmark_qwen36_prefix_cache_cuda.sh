#!/bin/bash
# Qwen3.6-35B-A3B: Prefix Cache CUDA Benchmark
# Enable the desired model groups below; independent tables, no decode forwards.
#
# Usage: bash benchmark_qwen36_prefix_cache_cuda.sh [threads] [gpu_id] [repetitions] [warmup]
# Examples:
#   bash benchmark_qwen36_prefix_cache_cuda.sh 8 1
#   bash benchmark_qwen36_prefix_cache_cuda.sh 8 1 3 1

set -e

MODEL_DIR="/data3/huangxin/model_list/Qwen3.6-35B-A3B"
BF16_MODEL="${MODEL_DIR}/Qwen3.6-35B-A3B-bf16.gguf"
Q4_0_MODEL="${MODEL_DIR}/Qwen3.6-35B-A3B-Q4_0.gguf"
BENCH="/data2/huangxin/qwen36_cache_experiments/bin/qwen36-prefix-snapshot"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER="${SCRIPT_DIR}/qwen36_prefix_cache_cuda/run_sweep.py"
THREADS=${1-8}
GPU_ID=${2-0}
REPETITIONS=${3-3}
WARMUP=${4-1}
# Prefill_PROMPTS="128,256,512,1024,2048,4096,8192,16384,32768,65535"
Prefill_PROMPTS="131077,262144"

if [[ $# -gt 4 || ! $THREADS =~ ^[1-9][0-9]*$ || ! $GPU_ID =~ ^(0|[1-9][0-9]*)$ ||
      ! $REPETITIONS =~ ^[1-9][0-9]*$ || ! $WARMUP =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: Usage: $0 [positive threads] [nonnegative gpu_id] [repetitions] [warmup]"
    exit 1
fi
for COUNT in "$THREADS" "$GPU_ID" "$REPETITIONS" "$WARMUP"; do
    if [[ ${#COUNT} -gt 10 ]] || (( COUNT > 2147483647 )); then
        echo "ERROR: Argument exceeds the supported integer range."
        exit 1
    fi
done
if (( REPETITIONS > 1000 || WARMUP > 1000 )); then
    echo "ERROR: repetitions and warmup must be in [1, 1000]"
    exit 1
fi

echo "========================================================"
echo "  Qwen3.6-35B-A3B: Prefix Cache CUDA Benchmark"
echo "  GPU: $GPU_ID (CUDA_VISIBLE_DEVICES; probe uses GPU 0)"
echo "  Threads: $THREADS"
echo "  Prompt tokens: $Prefill_PROMPTS"
echo "  Repetitions: $REPETITIONS; warmup: $WARMUP"
echo "  Prefill only; separate result tables for enabled model groups"
echo "========================================================"
echo

# Check benchmark tools.
if [ ! -x "$BENCH" ] || [ ! -f "$RUNNER" ]; then
    echo "ERROR: Prefix snapshot benchmark tools are missing."
    echo "Run build_script/build_qwen36_prefix_cache_cuda.sh first."
    exit 1
fi

# echo
# echo "--- BF16 Long Sequence Benchmark ---"
# python3 -I "$RUNNER" --probe "$BENCH" \
#     --bf16-model "$BF16_MODEL" \
#     --threads "$THREADS" --gpu-id "$GPU_ID" --pp-lengths "$Prefill_PROMPTS" \
#     --repetitions "$REPETITIONS" --warmup "$WARMUP"

echo
echo "--- Q4_0 Long Sequence Benchmark ---"
python3 -I "$RUNNER" --probe "$BENCH" \
    --q4-model "$Q4_0_MODEL" \
    --threads "$THREADS" --gpu-id "$GPU_ID" --pp-lengths "$Prefill_PROMPTS" \
    --repetitions "$REPETITIONS" --warmup "$WARMUP"

echo
echo "Done."
