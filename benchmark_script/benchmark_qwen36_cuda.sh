#!/bin/bash
# Qwen3.6-35B-A3B: Q4_0 single-GPU CUDA Benchmark
# Three repetitions with warmup; the optional BF16 command is commented out.
# -p and -n run independent prompt-processing and generation tests.
# Long contexts, especially BF16, may run out of GPU memory.
#
# Usage: bash benchmark_qwen36_cuda.sh [threads] [gpu_id]
# Examples:
#   bash benchmark_qwen36_cuda.sh 4 0
#   bash benchmark_qwen36_cuda.sh 8 3

set -e

MODEL_DIR="/data3/huangxin/model_list/Qwen3.6-35B-A3B"
BF16_MODEL="${MODEL_DIR}/Qwen3.6-35B-A3B-bf16.gguf"
Q4_0_MODEL="${MODEL_DIR}/Qwen3.6-35B-A3B-Q4_0.gguf"
BENCH="/home/huangxin/code_list/bitnet-moe-script/build_script/build_bin_qwen36_cuda/bin/llama-bench"
THREADS=${1:-8}
GPU_ID=${2:-0}
Prefill_PROMPTS="128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144"
Decode_PROMPTS="128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144"

if [[ $# -gt 2 || ! $THREADS =~ ^[1-9][0-9]*$ || ! $GPU_ID =~ ^[0-9]+$ ]]; then
    echo "ERROR: Usage: $0 [positive threads] [nonnegative gpu_id]"
    exit 1
fi

# Select a visible CUDA device with -mg; keep inherited CUDA_VISIBLE_DEVICES.

echo "========================================================"
echo "  Qwen3.6-35B-A3B: Q4_0 CUDA Benchmark"
echo "  GPU: $GPU_ID (-mg, visible CUDA device)"
echo "  Threads: $THREADS"
echo "  Prompt tokens: $Prefill_PROMPTS"
echo "  Generation: independent $Decode_PROMPTS; repetitions: 3"
echo "========================================================"
echo

# Check models and benchmark binary.
for MODEL in "$Q4_0_MODEL" "$BF16_MODEL"; do
    if [ ! -s "$MODEL" ]; then
        echo "ERROR: Model missing or empty: $MODEL"
        exit 1
    fi
done
if [ ! -x "$BENCH" ]; then
    echo "ERROR: llama-bench not found or not executable: $BENCH"
    echo "Run build_script/build_qwen36_cuda.sh first."
    exit 1
fi

echo "--- Model Sizes ---"
echo "  BF16: $(du -h "$BF16_MODEL" | cut -f1)"
echo "  Q4_0: $(du -h "$Q4_0_MODEL" | cut -f1)"
echo

# echo
# echo "--- BF16 Long Sequence Benchmark ---"
# "$BENCH" -m "$BF16_MODEL" -t "$THREADS" -p "$Prefill_PROMPTS" -n "$Decode_PROMPTS" -r 3 -ngl 99 -sm none -mg "$GPU_ID" -b 2048 -ub 512 -ctk f16 -ctv f16 -fa auto

echo
echo "--- Q4_0 Long Sequence Benchmark ---"
"$BENCH" -m "$Q4_0_MODEL" -t "$THREADS" -p "$Prefill_PROMPTS" -n "$Decode_PROMPTS" -r 3 -ngl 99 -sm none -mg "$GPU_ID" -b 2048 -ub 512 -ctk f16 -ctv f16 -fa auto

echo
echo "Done."

# CUDA_VISIBLE_DEVICES=1 bash /home/huangxin/code_list/bitnet-moe-script/benchmark_script/benchmark_qwen36_cuda.sh 8 0