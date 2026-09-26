#!/bin/bash
# ==========================================================
# 构建 Qwen3.6 BitNet CPU 工具
#
# 源码: /home/huangxin/code_list/BitNet
# 用法: bash build_qwen36_bitnet_cpu.sh [jobs]，默认 8
# ==========================================================
set -e

BITNET_DIR="/home/huangxin/code_list/BitNet"
LLAMA_DIR="${BITNET_DIR}/3rdparty/llama.cpp"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BUILD_DIR="${SCRIPT_DIR}/build_bin_qwen36_bitnet"
JOBS=${1:-8}
if [[ $# -gt 1 || ! $JOBS =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: Usage: $0 [positive jobs, default 8]"
    exit 1
fi

# 通用 cmake 参数，与 build_yoco_moe_models.sh 保持一致
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
echo "  构建 Qwen3.6 BitNet CPU 工具"
echo "  源码:     ${BITNET_DIR}"
echo "  llama.cpp HEAD: $(git -C "${LLAMA_DIR}" rev-parse --short HEAD)"
echo "  输出:     ${BUILD_DIR}"
echo "  并行数:   ${JOBS}"
echo "========================================================"

cmake -S "${BITNET_DIR}" -B "${BUILD_DIR}" "${CMAKE_COMMON[@]}"
cmake --build "${BUILD_DIR}" \
    --target llama-bench \
    --target llama-cli \
    --target llama-perplexity \
    --target llama-quantize \
    -j"${JOBS}"

echo ""
echo "========================================================"
echo "  构建完成！"
echo "  llama-bench:      ${BUILD_DIR}/bin/llama-bench"
echo "  llama-cli:        ${BUILD_DIR}/bin/llama-cli"
echo "  llama-perplexity: ${BUILD_DIR}/bin/llama-perplexity"
echo "  llama-quantize:   ${BUILD_DIR}/bin/llama-quantize"
echo "========================================================"
