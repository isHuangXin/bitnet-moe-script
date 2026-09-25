#!/usr/bin/env bash
# Build isolated Qwen3.6 Q4_0_W158 tools with native BitNet I2_S CPU kernels.
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: build_qwen36_q4_0_w158_cpu.sh [jobs]' \
        '  jobs: positive integer (default: JOBS or 8)' \
        'Environment: BITNET_DIR, LLAMA_DIR, BUILD_DIR, BUILD_LOG_DIR, JOBS' \
        '  BITNET_DIR: /home/huangxin/code_list/BitNet by default' \
        '  LLAMA_DIR: $BITNET_DIR/3rdparty/llama.cpp by default' \
        '  BUILD_DIR: build_script/build_bin_qwen36_q4_0_w158_native_i2s by default' \
        'Builds llama-cli, llama-bench, llama-quantize, llama-debug; no tests.' \
        'Requires x86 AVX2; compiles the actual parent src/ggml-bitnet-mad.cpp.'
}
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
(( $# <= 1 )) || fail 'Expected at most one jobs argument; use --help.'
JOBS=${1:-${JOBS:-8}}
[[ $JOBS =~ ^[1-9][0-9]*$ && ${#JOBS} -le 6 ]] || fail 'jobs must be a positive integer of at most 6 digits.'
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
BITNET_DIR=${BITNET_DIR:-/home/huangxin/code_list/BitNet}
LLAMA_DIR=${LLAMA_DIR:-${BITNET_DIR}/3rdparty/llama.cpp}
BUILD_DIR=${BUILD_DIR:-${SCRIPT_DIR}/build_bin_qwen36_q4_0_w158_native_i2s}
BUILD_LOG_DIR=${BUILD_LOG_DIR:-${SCRIPT_DIR}/logs/qwen36_q4_0_w158_native_i2s}
for tool in cmake clang clang++ git realpath mktemp tee sha256sum; do
    command -v "$tool" >/dev/null || fail "Required tool not found: $tool"
done
[[ -f $LLAMA_DIR/CMakeLists.txt && -f $LLAMA_DIR/src/llama-qwen36-s2.cpp ]] || fail "Missing llama.cpp/S2 sources: $LLAMA_DIR"
for source in src/ggml-bitnet-mad.cpp include/gemm-config.h include/ggml-bitnet-i2s.h; do
    [[ -f $BITNET_DIR/$source ]] || fail "Missing native BitNet I2_S source: $BITNET_DIR/$source"
done
BITNET_DIR=$(realpath -e -- "$BITNET_DIR")
LLAMA_DIR=$(realpath -e -- "$LLAMA_DIR")
BUILD_DIR=$(realpath -m -- "$BUILD_DIR")
[[ $BUILD_DIR != / && $BUILD_DIR != "$LLAMA_DIR" && $BUILD_DIR != "$BITNET_DIR" ]] || fail 'BUILD_DIR must be a separate build directory.'
[[ ! -e $BUILD_DIR || -d $BUILD_DIR ]] || fail "BUILD_DIR is not a directory: $BUILD_DIR"

mkdir -p -- "$BUILD_DIR" "$BUILD_LOG_DIR"
BUILD_LOG_DIR=$(realpath -e -- "$BUILD_LOG_DIR")
LOG_DIR=$(mktemp -d "${BUILD_LOG_DIR}/$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")
LOG_FILE=${LOG_DIR}/build.log
trap 'rc=$?; if (( rc != 0 )); then printf "Build failed (%s); logs: %s\n" "$rc" "$LOG_DIR" >&2; fi' EXIT

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_COMPILER=clang
    -DCMAKE_CXX_COMPILER=clang++
    -DGGML_CPU=ON
    -DGGML_NATIVE=ON
    -DGGML_CPU_REPACK=ON
    -DGGML_BITNET_I2S_NATIVE=ON
    "-DBITNET_SOURCE_DIR=$BITNET_DIR"
    -DGGML_OPENMP=OFF
    -DGGML_BACKEND_DL=OFF
    -DGGML_CPU_ALL_VARIANTS=OFF
    -DGGML_BLAS=OFF
    -DGGML_ACCELERATE=OFF
    -DGGML_CUDA=OFF
    -DGGML_HIP=OFF
    -DGGML_MUSA=OFF
    -DGGML_VULKAN=OFF
    -DGGML_METAL=OFF
    -DGGML_SYCL=OFF
    -DGGML_OPENCL=OFF
    -DGGML_WEBGPU=OFF
    -DGGML_RPC=OFF
    -DGGML_HEXAGON=OFF
    -DGGML_ZDNN=OFF
    -DGGML_ZENDNN=OFF
    -DGGML_OPENVINO=OFF
    -DGGML_VIRTGPU=OFF
    -DGGML_VIRTGPU_BACKEND=OFF
    -DGGML_ET=OFF
    -DGGML_ET_SYSEMU=OFF
    -DLLAMA_BUILD_COMMON=ON
    -DLLAMA_BUILD_TOOLS=ON
    -DLLAMA_BUILD_EXAMPLES=ON
    # This revision defines llama-cli only with the server components enabled.
    -DLLAMA_BUILD_SERVER=ON
    -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_UI=OFF
    -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_TESTS=OFF
    -DGGML_BUILD_TESTS=OFF
)
CONFIGURE=(cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" "${CMAKE_ARGS[@]}")
BUILD=(cmake --build "$BUILD_DIR" --config Release --parallel "$JOBS" --target llama-cli llama-bench llama-quantize llama-debug)

record_source_tree() {
    local label=$1 source_dir=$2
    printf '%s source: %s\n' "$label" "$source_dir"
    printf '%s branch: ' "$label"; git -C "$source_dir" rev-parse --abbrev-ref HEAD
    printf '%s HEAD: ' "$label"; git -C "$source_dir" rev-parse HEAD
    git -C "$source_dir" status --short | tee "$LOG_DIR/${label}.status.txt"
    git -C "$source_dir" diff --binary --full-index --no-ext-diff --no-textconv HEAD -- > "$LOG_DIR/${label}.diff"
    printf '%s status and tracked dirty-diff SHA256:\n' "$label"
    sha256sum -- "$LOG_DIR/${label}.status.txt" "$LOG_DIR/${label}.diff"
}

{
    printf 'UTC: %s\nSource: %s\nBitNet: %s\nBuild: %s\nJobs: %s\nLogs: %s\n' "$(date -u +%FT%TZ)" "$LLAMA_DIR" "$BITNET_DIR" "$BUILD_DIR" "$JOBS" "$LOG_DIR"
    record_source_tree parent "$BITNET_DIR"
    record_source_tree child "$LLAMA_DIR"
    printf 'Native sources and build wrapper SHA256:\n'
    sha256sum -- "$BITNET_DIR/src/ggml-bitnet-mad.cpp" "$BITNET_DIR/include/gemm-config.h" \
        "$BITNET_DIR/include/ggml-bitnet-i2s.h" "${BASH_SOURCE[0]}" | tee "$LOG_DIR/native-sources.sha256"
    clang --version
    clang++ --version
    cmake --version
    printf 'Configure:'; printf ' %q' "${CONFIGURE[@]}"; printf '\n'
    "${CONFIGURE[@]}"
    cp -- "$BUILD_DIR/CMakeCache.txt" "$LOG_DIR/CMakeCache.txt"
    printf 'Build:'; printf ' %q' "${BUILD[@]}"; printf '\n'
    "${BUILD[@]}"
    for target in llama-cli llama-bench llama-quantize llama-debug; do
        [[ -x $BUILD_DIR/bin/$target ]] || fail "Missing built executable: $BUILD_DIR/bin/$target"
        sha256sum -- "$BUILD_DIR/bin/$target"
    done
    printf 'Build complete: %s/bin\nPersistent log: %s\n' "$BUILD_DIR" "$LOG_FILE"
} 2>&1 | tee "$LOG_FILE"
