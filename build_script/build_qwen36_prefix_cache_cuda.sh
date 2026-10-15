#!/bin/bash
# ==========================================================
# 构建 Qwen3.6 CUDA 库和前缀缓存测试工具
#
# 一次完成 llama.cpp CUDA 库、probe 和 runtime 指纹构建
# 用法: bash build_qwen36_prefix_cache_cuda.sh [jobs]，默认 8
# ==========================================================
set -e

BITNET_DIR="/home/huangxin/code_list/BitNet"
LLAMA_DIR="${BITNET_DIR}/3rdparty/llama.cpp"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/../benchmark_script/qwen36_prefix_cache_cuda"
CUDA_BUILD_DIR="${SCRIPT_DIR}/build_bin_qwen36_cuda"
LIB_DIR="${CUDA_BUILD_DIR}/bin"
BUILD_DIR="/data2/huangxin/qwen36_cache_experiments"
BIN_DIR="${BUILD_DIR}/bin"
CXX="/usr/bin/clang++"
JOBS=${1-8}
if [[ $# -gt 1 || ! $JOBS =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: Usage: $0 [positive jobs, default 8]"
    exit 1
fi

CMAKE_COMMON=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_COMPILER=clang
    -DCMAKE_CXX_COMPILER="$CXX"
    -DBUILD_SHARED_LIBS=ON
    -DLLAMA_USE_SYSTEM_GGML=OFF
    -DGGML_CPU=ON
    -DGGML_CUDA=ON
    -DGGML_BACKEND_DL=OFF
    -DGGML_CPU_ALL_VARIANTS=OFF
    -DCMAKE_CUDA_ARCHITECTURES=80
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
echo "  构建 Qwen3.6 CUDA 库和前缀缓存测试工具"
echo "  源码:     ${SOURCE_DIR}"
echo "  CUDA 库:  ${LIB_DIR}"
echo "  CUDA 架构: 80 (A100)"
echo "  编译并行数: ${JOBS}"
echo "  输出:     ${BIN_DIR}"
echo "========================================================"

cmake -S "$BITNET_DIR" -B "$CUDA_BUILD_DIR" "${CMAKE_COMMON[@]}"
cmake --build "$CUDA_BUILD_DIR" --target llama -j"$JOBS"

mkdir -p "$BIN_DIR"
"$CXX" -std=c++17 -O2 -pthread \
    -I "$LLAMA_DIR/include" -I "$LLAMA_DIR/ggml/include" \
    "$SOURCE_DIR/snapshot_probe.cpp" -L "$LIB_DIR" \
    -Wl,-rpath,"$LIB_DIR" \
    -Wl,-rpath-link,/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs \
    -lllama -lggml -lggml-base -luring -o "$BIN_DIR/qwen36-prefix-snapshot"

# 记录同运行入口库路径下解析的 liburing；生成指纹清单，不执行测试。
URING_LIB="$(LD_LIBRARY_PATH="$LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$BIN_DIR/qwen36-prefix-snapshot" | awk '$1 ~ /^liburing\.so/ && $2 == "=>" {print $3}')"
# 调参入口不计入构建指纹；实际运行参数由 runner 写入 metadata。
sha256sum \
    "$BIN_DIR/qwen36-prefix-snapshot" "$URING_LIB" "$(readlink -e "$URING_LIB")" \
    "$SOURCE_DIR/snapshot_probe.cpp" "$SOURCE_DIR/snapshot_store.hpp" \
    "$SOURCE_DIR/benchmark_stats.hpp" "$SOURCE_DIR/run_sweep.py" \
    "$SCRIPT_DIR/build_qwen36_prefix_cache_cuda.sh" \
    "$LLAMA_DIR"/include/*.h "$LLAMA_DIR"/ggml/include/*.h \
    "$CUDA_BUILD_DIR/CMakeCache.txt" \
    "$CUDA_BUILD_DIR/3rdparty/llama.cpp/common/build-info.cpp" \
    "$LIB_DIR"/lib{llama,ggml,ggml-base,ggml-cpu,ggml-cuda}.so* \
    > "$BIN_DIR/runtime.sha256.tmp"
mv "$BIN_DIR/runtime.sha256.tmp" "$BIN_DIR/runtime.sha256"

echo ""
echo "========================================================"
echo "  构建完成！"
echo "  probe:    ${BIN_DIR}/qwen36-prefix-snapshot"
echo "  manifest: ${BIN_DIR}/runtime.sha256"
echo "========================================================"
