#!/bin/bash
# Monitor memory usage of a benchmark script over time
# Samples RSS of all llama-bench child processes every 0.1s
#
# Usage: bash benchmark_memory_monitor.sh <benchmark_script> [script_args...]
# Output: CSV file with timestamp and memory (MB), plus a summary
#
# Example:
#   bash benchmark_memory_monitor.sh /path/to/benchmark_yoco_dense_3b_d_ffn_9216_cpu.sh 8 0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$1"
shift
ARGS="$@"

if [ -z "$SCRIPT" ]; then
    echo "Usage: $0 <benchmark_script> [script_args...]"
    exit 1
fi

# Output file
BASENAME=$(basename "$SCRIPT" .sh)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_FILE="${SCRIPT_DIR}/memory_logs/${BASENAME}_${TIMESTAMP}.csv"
LOG_FILE="${SCRIPT_DIR}/memory_logs/${BASENAME}_${TIMESTAMP}.log"
mkdir -p "$(dirname "$CSV_FILE")"

echo "=========================================="
echo "  Memory Monitor"
echo "  Script: $SCRIPT $ARGS"
echo "  Output: $CSV_FILE"
echo "  Log:    $LOG_FILE"
echo "=========================================="

# Write CSV header
echo "time_sec,rss_mb" > "$CSV_FILE"
> "$LOG_FILE"

# Record start time
START_TIME=$(date +%s%N)

# Start the benchmark in background, capture stdout with timestamps
bash "$SCRIPT" $ARGS 2>&1 | while IFS= read -r line; do
    NOW=$(date +%s%N)
    ELAPSED=$(echo "scale=3; ($NOW - $START_TIME) / 1000000000" | bc)
    echo "[$ELAPSED] $line" >> "$LOG_FILE"
    echo "$line"
done &
BENCH_PID=$!

# Monitor memory usage
SAMPLE_INTERVAL=0.1

while kill -0 $BENCH_PID 2>/dev/null; do
    # Get current time offset in seconds
    NOW=$(date +%s%N)
    ELAPSED=$(echo "scale=2; ($NOW - $START_TIME) / 1000000000" | bc)

    # Sum RSS of all llama-bench processes (in KB from /proc/*/status)
    TOTAL_RSS=0
    for pid in $(pgrep -f "llama-bench" 2>/dev/null); do
        if [ -f "/proc/$pid/status" ]; then
            RSS_KB=$(grep -m1 "^VmRSS:" /proc/$pid/status 2>/dev/null | awk '{print $2}')
            if [ -n "$RSS_KB" ] && [ "$RSS_KB" -gt 0 ] 2>/dev/null; then
                TOTAL_RSS=$((TOTAL_RSS + RSS_KB))
            fi
        fi
    done

    RSS_MB=$(echo "scale=2; $TOTAL_RSS / 1024" | bc)

    # Only record if there's actual memory usage
    if [ "$TOTAL_RSS" -gt 0 ]; then
        echo "$ELAPSED,$RSS_MB" >> "$CSV_FILE"
    fi

    sleep $SAMPLE_INTERVAL
done

# Wait for benchmark to finish
wait $BENCH_PID 2>/dev/null || true

# Summary
LINES=$(wc -l < "$CSV_FILE")
if [ "$LINES" -gt 1 ]; then
    PEAK_MB=$(awk -F',' 'NR>1 {if($2>max) max=$2} END {printf "%.1f", max}' "$CSV_FILE")
    PEAK_GB=$(echo "scale=2; $PEAK_MB / 1024" | bc)
    echo ""
    echo "=========================================="
    echo "  Memory Monitor Complete"
    echo "  Samples: $((LINES - 1))"
    echo "  Peak RSS: ${PEAK_MB} MB (${PEAK_GB} GB)"
    echo "  CSV: $CSV_FILE"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "  WARNING: No memory samples collected!"
    echo "  CSV: $CSV_FILE"
    echo "=========================================="
fi
