#!/bin/bash
# Build upstream llama.cpp with optimized Qwen3.6 S7 CPU support.
# Full-attention Q/K/V/O (including Q gate rows) and GDN qkv/z/a/b/out: W2A16.
# Routed experts: W2A8; embedding/head: INT8.
# Shared expert gate/up/down projections: W2A16; independent shared_expert_gate: BF16.
# Build the S7 tools in an independent directory.

set -e

LLAMA_DIR="/home/huangxin/code_list/BitNet/3rdparty/llama.cpp"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_bin_qwen36_s7"

# 通用 cmake 参数
CMAKE_COMMON=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_COMPILER=clang
    -DCMAKE_CXX_COMPILER=clang++
    -DGGML_NATIVE=ON
    -DGGML_OPENMP=OFF
    -DBITNET_X86_TL2=OFF
    -DLLAMA_BUILD_COMMON=ON
    -DLLAMA_BUILD_TOOLS=ON
    -DLLAMA_BUILD_EXAMPLES=ON
    -DGGML_AVX=ON
    -DGGML_AVX2=ON
    -DGGML_AVX512=ON
    -DGGML_AVX512_VBMI=ON
    -DGGML_AVX512_VNNI=ON
    -DGGML_AVX512_BF16=ON
    -DGGML_FMA=ON
    -DGGML_F16C=ON
)

echo "========================================================"
echo "  Build Qwen3.6 S7 CPU tools (no TL2)"
echo "  Source: ${LLAMA_DIR}"
echo "  Output: ${BUILD_DIR}"
echo "========================================================"

if [ ! -f "${LLAMA_DIR}/src/llama-qwen36-s2.cpp" ] || \
   [ ! -f "${LLAMA_DIR}/convert_hf_to_gguf_s7.py" ]; then
    echo "ERROR: S7 source support not found in ${LLAMA_DIR}"
    exit 1
fi

cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" "${CMAKE_COMMON[@]}"
cmake --build "$BUILD_DIR" --target llama-bench llama-cli test-qwen36-s2 -j8
ctest --test-dir "$BUILD_DIR" -R '^test-qwen36-s2$' --output-on-failure --no-tests=error

echo
echo "Build complete: ${BUILD_DIR}/bin"
