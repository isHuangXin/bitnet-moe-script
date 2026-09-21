#!/bin/bash
# Qwen3.6-35B-A3B: official BF16 HF weights -> S4 GGUF
# Text only, no MTP. Routed experts: w1.58 W2A8; embedding/head: INT8.
# Full-attention Q/K/V/O: W2A16 (including Q gate rows); shared experts/GDN: BF16.
#
# Usage: bash generate_qwen36_35b_a3b_s4_model.sh [threads]
# Example: bash generate_qwen36_35b_a3b_s4_model.sh 8

set -e

MODEL_DIR="/data3/huangxin/model_list/Qwen3.6-35B-A3B"
S4_MODEL="${MODEL_DIR}/Qwen3.6-35B-A3B-S4.gguf"
CONVERTER="/home/huangxin/code_list/BitNet/3rdparty/llama.cpp/convert_hf_to_gguf_s4.py"
SOURCE_REVISION="995ad96eacd98c81ed38be0c5b274b04031597b0"
THREADS=${1:-8}

echo "========================================================"
echo "  Qwen3.6-35B-A3B: S4 Model Conversion (no MTP)"
echo "  Source: $MODEL_DIR"
echo "  Output: $S4_MODEL"
echo "  Threads: $THREADS"
echo "========================================================"
echo

if [ ! -f "$CONVERTER" ]; then
    echo "ERROR: S4 converter not found: $CONVERTER"
    exit 1
fi

python3 -u "$CONVERTER" "$MODEL_DIR" \
    --outfile "$S4_MODEL" \
    --threads "$THREADS" \
    --source-revision "$SOURCE_REVISION"

echo
echo "Done."
echo "  Model: $S4_MODEL"
echo "  Manifest: ${S4_MODEL}.manifest.json"
