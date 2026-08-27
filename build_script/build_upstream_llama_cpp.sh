#!/bin/bash
# ==========================================================
# Build upstream llama.cpp (ggml-org/llama.cpp master branch)
#
# Source: /home/huangxin/code_list/BitNet/3rdparty/llama.cpp
# Branch: upstream-master
#
# Purpose: Build llama-bench for benchmarking Qwen3-30B-A3B
#          on official upstream llama.cpp
# ==========================================================
set -e

LLAMA_DIR="/home/huangxin/code_list/BitNet/3rdparty/llama.cpp"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BUILD_DIR="${SCRIPT_DIR}/build_bin_upstream"

# Common cmake parameters (same as BitNet build)
CMAKE_COMMON=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_COMPILER=clang
    -DCMAKE_CXX_COMPILER=clang++
    -DGGML_NATIVE=ON
    -DGGML_OPENMP=OFF
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

# Switch to upstream-master branch
CURRENT_BRANCH=$(git -C "${LLAMA_DIR}" branch --show-current)
echo "========================================================"
echo "  Build upstream llama.cpp (ggml-org master)"
echo "  Source:   ${LLAMA_DIR}"
echo "  Branch:   upstream-master"
echo "  Current:  ${CURRENT_BRANCH}"
echo "  Commit:   $(git -C "${LLAMA_DIR}" rev-parse --short upstream/master)"
echo "  Output:   ${BUILD_DIR}"
echo "========================================================"

if [ -f "${BUILD_DIR}/bin/llama-bench" ] && [ -f "${BUILD_DIR}/bin/llama-quantize" ]; then
    echo "Build already exists, skip. To rebuild, delete: ${BUILD_DIR}"
    exit 0
fi

# Checkout upstream-master
git -C "${LLAMA_DIR}" checkout upstream-master
echo "Switched to upstream-master branch."

cmake -S "${LLAMA_DIR}" -B "${BUILD_DIR}" "${CMAKE_COMMON[@]}"
cmake --build "${BUILD_DIR}" \
    --target llama-bench \
    --target llama-quantize \
    -j$(nproc)

# Switch back to original branch
git -C "${LLAMA_DIR}" checkout "${CURRENT_BRANCH}"
echo "Restored branch: ${CURRENT_BRANCH}"

echo ""
echo "========================================================"
echo "  Build complete!"
echo "  llama-bench:    ${BUILD_DIR}/bin/llama-bench"
echo "  llama-quantize: ${BUILD_DIR}/bin/llama-quantize"
echo "========================================================"
