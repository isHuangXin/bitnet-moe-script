#!/bin/bash
# Build upstream llama.cpp with optimized Qwen3.6 S2 CPU support.
# Rebuild the existing S2 tools in place.

set -e

LLAMA_DIR="/home/huangxin/code_list/BitNet/3rdparty/llama.cpp"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_bin_qwen36_s2"

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
echo "  Build Qwen3.6 S2 CPU tools (no TL2)"
echo "  Source: ${LLAMA_DIR}"
echo "  Output: ${BUILD_DIR}"
echo "========================================================"

if [ ! -f "${LLAMA_DIR}/src/llama-qwen36-s2.cpp" ]; then
    echo "ERROR: S2 source support not found in ${LLAMA_DIR}"
    exit 1
fi

cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" "${CMAKE_COMMON[@]}"
cmake --build "$BUILD_DIR" --target llama-bench llama-cli test-qwen36-s2 -j8
ctest --test-dir "$BUILD_DIR" -R '^test-qwen36-s2$' --output-on-failure --no-tests=error

echo
echo "Build complete: ${BUILD_DIR}/bin"
