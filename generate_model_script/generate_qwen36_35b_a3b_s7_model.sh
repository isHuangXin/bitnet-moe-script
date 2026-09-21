#!/bin/bash
# Qwen3.6-35B-A3B: official BF16 HF weights -> S7 GGUF
# Text only, no MTP. Routed experts: W2A8; embedding/head: INT8.
# Full-attention Q/K/V/O (including Q gate rows) and GDN qkv/z/a/b/out: W2A16.
# Shared expert gate/up/down projections: W2A16; independent shared_expert_gate: BF16.
#
# Usage: bash generate_qwen36_35b_a3b_s7_model.sh [threads]
# Example: bash generate_qwen36_35b_a3b_s7_model.sh 8

set -e

MODEL_DIR="/data3/huangxin/model_list/Qwen3.6-35B-A3B"
S7_MODEL="${MODEL_DIR}/Qwen3.6-35B-A3B-S7.gguf"
CONVERTER="/home/huangxin/code_list/BitNet/3rdparty/llama.cpp/convert_hf_to_gguf_s7.py"
SOURCE_REVISION="995ad96eacd98c81ed38be0c5b274b04031597b0"
THREADS=${1:-8}

echo "========================================================"
echo "  Qwen3.6-35B-A3B: S7 Model Conversion (no MTP)"
echo "  Source: $MODEL_DIR"
echo "  Output: $S7_MODEL"
echo "  Threads: $THREADS"
echo "========================================================"
echo

if [ ! -f "$CONVERTER" ]; then
    echo "ERROR: S7 converter not found: $CONVERTER"
    exit 1
fi

python3 -u "$CONVERTER" "$MODEL_DIR" \
    --outfile "$S7_MODEL" \
    --threads "$THREADS" \
    --source-revision "$SOURCE_REVISION"

echo
echo "Done."
echo "  Model: $S7_MODEL"
echo "  Manifest: ${S7_MODEL}.manifest.json"
