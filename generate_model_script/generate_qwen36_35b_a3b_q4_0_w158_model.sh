#!/bin/bash
# Generate only the native I2S hybrid from existing BF16, Q4_0 and strict S2 controls.
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: generate_qwen36_35b_a3b_q4_0_w158_model.sh [threads]' \
        '  threads: positive integer (default: 8)' \
        '  Requires existing BF16, Q4_0, S2 and S2 manifest; never regenerates controls.' \
        '  Publishes only Qwen3.6-35B-A3B-Q4_0-W158-NativeI2S.gguf and its manifest.' \
        'Environment: LLAMA_DIR, MODEL_DIR, BUILD_DIR, PYTHON'
}
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
if [[ $# == 1 && ( $1 == --help || $1 == -h ) ]]; then usage; exit 0; fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LLAMA_DIR=${LLAMA_DIR:-/home/huangxin/code_list/BitNet/3rdparty/llama.cpp}
MODEL_DIR=${MODEL_DIR:-/data3/huangxin/model_list/Qwen3.6-35B-A3B}
BUILD_DIR=${BUILD_DIR:-${SCRIPT_DIR}/../build_script/build_bin_qwen36_q4_0_w158_native_i2s}
PYTHON=${PYTHON:-python3}
THREADS=${1:-8}
if [[ $# -gt 1 || ! $THREADS =~ ^[1-9][0-9]*$ || ${#THREADS} -gt 9 ]]; then
    fail 'Usage: generate_qwen36_35b_a3b_q4_0_w158_model.sh [positive threads, default 8]'
fi
[[ -d $MODEL_DIR ]] || fail "Model directory not found: $MODEL_DIR"
command -v "$PYTHON" >/dev/null || fail "Python executable not found: $PYTHON"
MODEL_DIR=$(cd -- "$MODEL_DIR" && pwd -P)
COMPOSER="${LLAMA_DIR}/convert_qwen36_q4_0_w158.py"
BF16_MODEL="${MODEL_DIR}/Qwen3.6-35B-A3B-bf16.gguf"
S2_MODEL="${MODEL_DIR}/Qwen3.6-35B-A3B-S2.gguf"
Q4_MODEL="${MODEL_DIR}/Qwen3.6-35B-A3B-Q4_0.gguf"
OUTPUT="${MODEL_DIR}/Qwen3.6-35B-A3B-Q4_0-W158-NativeI2S.gguf"
[[ -f $COMPOSER ]] || fail "Composer not found: $COMPOSER"
for path in "$OUTPUT" "${OUTPUT}.manifest.json"; do
    [[ ! -e $path && ! -L $path ]] || fail "Refusing existing destination (including symlinks): $path"
done
for path in "$BF16_MODEL" "$S2_MODEL" "${S2_MODEL}.manifest.json" "$Q4_MODEL"; do
    [[ -f $path && -r $path && -s $path ]] || fail "Required existing input is missing, unreadable, or empty: $path"
done

LOG_DIR="${SCRIPT_DIR}/logs/qwen36_q4_0_w158_native_i2s"
mkdir -p -- "$LOG_DIR"
RUN_DIR=$(mktemp -d "${LOG_DIR}/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
cleanup() {
    status=$?
    trap - EXIT
    if [[ $status -ne 0 ]]; then
        printf 'Generation failed; existing models were not overwritten. Logs: %s\n' "$RUN_DIR" >&2
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run() {
    local step=$1
    shift
    printf '\n[%s]' "$step" | tee -a "${RUN_DIR}/commands.log"
    printf ' %q' "$@" | tee -a "${RUN_DIR}/commands.log"
    printf '\n' | tee -a "${RUN_DIR}/commands.log"
    "$@" 2>&1 | tee "${RUN_DIR}/${step}.log"
}

printf 'Q4_0_W158_NATIVE_I2S generation, threads=%s\n' "$THREADS" | tee "${RUN_DIR}/generation.log"
printf 'LLAMA_DIR=%s\nMODEL_DIR=%s\nBUILD_DIR=%s\nPYTHON=%s\n' "$LLAMA_DIR" "$MODEL_DIR" "$BUILD_DIR" "$PYTHON" | tee -a "${RUN_DIR}/generation.log"
printf 'BF16=%s\nS2=%s\nQ4_0=%s\nOUTPUT=%s\nLOGS=%s\n' "$BF16_MODEL" "$S2_MODEL" "$Q4_MODEL" "$OUTPUT" "$RUN_DIR" | tee -a "${RUN_DIR}/generation.log"
printf 'Existing controls and Qwen3.6-35B-A3B-Q4_0-W158.gguf are retained unchanged.\nAncestry is caller-declared; matching metadata does not prove a common checkpoint.\n' | tee -a "${RUN_DIR}/generation.log"

# Retain full prerequisite audits and the composer's independent input/output audits.
run validate-bf16 env OMP_NUM_THREADS="$THREADS" "$PYTHON" -u "$COMPOSER" validate --model "$BF16_MODEL" --kind bf16
run validate-s2 env OMP_NUM_THREADS="$THREADS" "$PYTHON" -u "$COMPOSER" validate --model "$S2_MODEL" --kind s2
run validate-q4 env OMP_NUM_THREADS="$THREADS" "$PYTHON" -u "$COMPOSER" validate --model "$Q4_MODEL" --kind q4
run compose env OMP_NUM_THREADS="$THREADS" "$PYTHON" -u "$COMPOSER" --expert-kernel native-i2s \
    --q4-model "$Q4_MODEL" --s2-model "$S2_MODEL" --bf16-model "$BF16_MODEL" --outfile "$OUTPUT"
printf '\nNative I2S model: %s\nCompletion manifest: %s.manifest.json\nLogs: %s\n' "$OUTPUT" "$OUTPUT" "$RUN_DIR" | tee -a "${RUN_DIR}/generation.log"
