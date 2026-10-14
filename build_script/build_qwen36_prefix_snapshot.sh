#!/bin/bash
# ==========================================================
# 构建 Qwen3.6 前缀缓存测试工具
#
# 复用 build_qwen36_cuda.sh 编译的 llama.cpp CUDA 库
# 用法: bash build_qwen36_prefix_snapshot.sh
# ==========================================================
set -e

BITNET_DIR="/home/huangxin/code_list/BitNet"
LLAMA_DIR="${BITNET_DIR}/3rdparty/llama.cpp"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/../benchmark_script/qwen36_prefix_snapshot"
CUDA_BUILD_DIR="${SCRIPT_DIR}/build_bin_qwen36_cuda"
LIB_DIR="${CUDA_BUILD_DIR}/bin"
BUILD_DIR="/data2/huangxin/qwen36_cache_experiments"
BIN_DIR="${BUILD_DIR}/bin"
CXX="/usr/bin/clang++"

echo "========================================================"
echo "  构建 Qwen3.6 前缀缓存测试工具"
echo "  源码:     ${SOURCE_DIR}"
echo "  CUDA 库:  ${LIB_DIR}"
echo "  输出:     ${BIN_DIR}"
echo "========================================================"

mkdir -p "$BIN_DIR"
"$CXX" -std=c++17 -O2 -pthread \
    -I "$LLAMA_DIR/include" -I "$LLAMA_DIR/ggml/include" \
    "$SOURCE_DIR/snapshot_probe.cpp" -L "$LIB_DIR" \
    -Wl,-rpath,"$LIB_DIR" \
    -Wl,-rpath-link,/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs \
    -lllama -lggml -lggml-base -o "$BIN_DIR/qwen36-prefix-snapshot"

# 生成运行入口必需的指纹清单，不执行测试或额外校验。
sha256sum \
    "$BIN_DIR/qwen36-prefix-snapshot" \
    "$SOURCE_DIR/snapshot_probe.cpp" "$SOURCE_DIR/snapshot_store.hpp" \
    "$SOURCE_DIR/benchmark_stats.hpp" "$SOURCE_DIR/run_sweep.py" \
    "$SCRIPT_DIR/build_qwen36_prefix_snapshot.sh" \
    "$SCRIPT_DIR/../benchmark_script/benchmark_qwen36_prefix_snapshot.sh" \
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
