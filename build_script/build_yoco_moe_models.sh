#!/bin/bash
# ==========================================================
# 构建 BitNet llama.cpp 二进制（用于 YOCO-U 模型 F16 vs I2_S benchmark）
#
# 源码: /home/huangxin/code_list/BitNet
# llama.cpp: 3rdparty/llama.cpp 当前 HEAD
#
# 目的: 编译支持 YOCO-U 架构推理的 llama-bench / llama-quantize 等工具
# ==========================================================
set -e

BITNET_DIR="/home/huangxin/code_list/BitNet"
LLAMA_DIR="${BITNET_DIR}/3rdparty/llama.cpp"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BUILD_DIR="${SCRIPT_DIR}/build_bin_yoco_u"

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
    -DGGML_AVX512=OFF
    -DGGML_AVX512_VBMI=OFF
    -DGGML_AVX512_VNNI=OFF
    -DGGML_AVX512_BF16=OFF
    -DGGML_FMA=ON
    -DGGML_F16C=ON
)

echo "========================================================"
echo "  构建 BitNet llama.cpp (bitnet-embeddings-0.6b)"
echo "  源码:     ${BITNET_DIR}"
echo "  llama.cpp HEAD: $(git -C "${LLAMA_DIR}" rev-parse --short HEAD)"
echo "  输出:     ${BUILD_DIR}"
echo "========================================================"

if [ -f "${BUILD_DIR}/bin/llama-bench" ] && [ -f "${BUILD_DIR}/bin/llama-quantize" ]; then
    echo "构建已存在，跳过。如需重新构建请删除: ${BUILD_DIR}"
    exit 0
fi

cmake -S "${BITNET_DIR}" -B "${BUILD_DIR}" "${CMAKE_COMMON[@]}"
cmake --build "${BUILD_DIR}" \
    --target llama-bench \
    --target llama-embedding \
    --target llama-quantize \
    -j$(nproc)

echo ""
echo "========================================================"
echo "  构建完成！"
echo "  llama-bench:    ${BUILD_DIR}/bin/llama-bench"
echo "  llama-quantize: ${BUILD_DIR}/bin/llama-quantize"
echo "  llama-embedding:${BUILD_DIR}/bin/llama-embedding"
echo "========================================================"
