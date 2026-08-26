#!/bin/bash
# Run memory monitoring for all 4 benchmark scripts (F16 + I2_S)
# Then plot the results with the Python script
#
# Usage: bash benchmark_all_memory.sh [threads] [numa_node]
# Example: bash benchmark_all_memory.sh 8 0

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="/home/huangxin/code_list/bitnet-moe-script/benchmark_script"
THREADS=${1:-8}
NUMA=${2:-0}
MONITOR="${SCRIPT_DIR}/benchmark_memory_monitor.sh"

echo "============================================================"
echo "  Running memory monitoring for all models"
echo "  Threads: $THREADS, NUMA: $NUMA"
echo "============================================================"
echo

SCRIPTS=(
    "${BENCH_DIR}/benchmark_yoco_dense_3b_d_ffn_9216_cpu.sh"
    "${BENCH_DIR}/benchmark_yoco_moe_30b_a3b_v3_cpu.sh"
    "${BENCH_DIR}/benchmark_yoco_u_dense_6b_d_ffn_9216_cpu.sh"
    "${BENCH_DIR}/benchmark_yoco_u_moe_30b_a3b_v3_cpu.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo ">>> Running: $(basename $script) with $THREADS threads"
        bash "$MONITOR" "$script" $THREADS $NUMA
        echo
    else
        echo ">>> SKIP: $script not found"
    fi
done

echo "============================================================"
echo "  All done. Run plot script to generate chart:"
echo "  python /home/huangxin/code_list/bitnet-moe-script/images/plot_memory_usage.py"
echo "============================================================"
