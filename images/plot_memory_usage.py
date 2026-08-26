#!/usr/bin/env python3
"""
Plot memory usage over time from benchmark memory monitor CSV files.

Usage:
    # Plot all CSV files (latest per model):
    python plot_memory_usage.py

    # Plot specific CSV file(s):
    python plot_memory_usage.py /path/to/file1.csv /path/to/file2.csv

    # The .log file is auto-detected (same name, .log extension)
"""
import os
import re
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np

LOGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'benchmark_memory_monitor_script', 'memory_logs')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'memory_usage_comparison.png')

# Model name mapping from filename
NAME_MAP = {
    'benchmark_yoco_dense_3b_d_ffn_9216_cpu': 'YOCO-Dense-3B (d_ffn=9216)',
    'benchmark_yoco_moe_30b_a3b_v3_cpu': 'YOCO-MoE-30B.A3B',
    'benchmark_yoco_u_dense_6b_d_ffn_9216_cpu': 'YOCO-U-Dense-6B (d_ffn=9216)',
    'benchmark_yoco_u_moe_30b_a3b_v3_cpu': 'YOCO-U-MoE-30B.A6B',
}

COLORS = {
    'YOCO-Dense-3B (d_ffn=9216)': '#70AD47',
    'YOCO-MoE-30B.A3B': '#4472C4',
    'YOCO-U-Dense-6B (d_ffn=9216)': '#FFC000',
    'YOCO-U-MoE-30B.A6B': '#ED7D31',
}


def parse_csv(filepath):
    """Parse CSV and return (times, rss_mb) arrays."""
    times = []
    rss = []
    with open(filepath) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    t = float(parts[0])
                    m = float(parts[1])
                    times.append(t)
                    rss.append(m)
                except ValueError:
                    continue
    return np.array(times), np.array(rss)


def parse_log(logpath):
    """Parse log file to extract test phase timestamps.

    llama-bench output lines look like:
    [12.345] bitnet | ... | pp128 | ... | 123.45 |
    [12.345] bitnet | ... | tg32  | ... | 45.67  |

    We also detect model switch by looking for lines containing the model path.
    Returns list of (time_sec, label) tuples.
    """
    events = []
    if not os.path.exists(logpath):
        return events

    # Pattern to match llama-bench result lines
    # e.g. [12.345] bitnet | ... | pp128 | ... |
    result_pattern = re.compile(
        r'^\[([0-9.]+)\]\s+.*\|\s*(pp|tg)(\d+)\s*\|'
    )
    # Pattern to detect model loading (echo lines from benchmark scripts)
    model_pattern = re.compile(
        r'^\[([0-9.]+)\]\s+.*(?:Running|Model|model).*?(f16|i2_s|F16|I2_S)',
        re.IGNORECASE
    )
    # Pattern to detect llama-bench command start with model path
    bench_start_pattern = re.compile(
        r'^\[([0-9.]+)\]\s+.*llama-bench.*?-m\s+\S*?(f16|i2_s)',
        re.IGNORECASE
    )

    with open(logpath) as f:
        for line in f:
            # Check for benchmark result lines (pp/tg)
            m = result_pattern.match(line)
            if m:
                t = float(m.group(1))
                task = m.group(2) + m.group(3)
                events.append((t, task))
                continue

            # Check for model switch markers
            m = model_pattern.match(line)
            if m:
                t = float(m.group(1))
                model_type = m.group(2).upper()
                events.append((t, f'__MODEL_{model_type}__'))
                continue

            m = bench_start_pattern.match(line)
            if m:
                t = float(m.group(1))
                model_type = m.group(2).upper()
                events.append((t, f'__START_{model_type}__'))

    return events


def get_model_name(filename):
    """Extract model name from CSV filename."""
    basename = os.path.basename(filename)
    for key, name in NAME_MAP.items():
        if key in basename:
            return name
    return basename


def main():
    # Determine input files
    if len(sys.argv) > 1:
        # User specified CSV file(s)
        csv_files = [f for f in sys.argv[1:] if f.endswith('.csv') and os.path.exists(f)]
        if not csv_files:
            print(f"Error: no valid CSV files found in arguments: {sys.argv[1:]}")
            return
        model_data = {}
        for f in csv_files:
            name = get_model_name(f)
            model_data[name] = f
        print(f"Plotting specified files: {csv_files}")
    else:
        # Auto-discover: latest CSV per model
        csv_files = sorted(glob.glob(os.path.join(LOGS_DIR, '*.csv')))
        if not csv_files:
            print(f"No CSV files found in {LOGS_DIR}")
            print("Run benchmark_all_memory.sh first.")
            return
        model_data = {}
        for f in csv_files:
            name = get_model_name(f)
            model_data[name] = f  # latest file wins (sorted by timestamp)
        print(f"Auto-detected {len(model_data)} model(s) from {LOGS_DIR}")

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    # Build title with model names
    model_names = list(model_data.keys())
    if model_names:
        title = f'Memory Usage During Benchmark - {" / ".join(model_names)}'
    else:
        title = 'Memory Usage During Benchmark (RSS over time)'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Determine output filename suffix from input CSV files
    input_files = list(model_data.values())
    if len(input_files) == 1:
        # Use the CSV filename (without extension) as suffix
        suffix = os.path.basename(input_files[0]).replace('.csv', '')
        output_path = os.path.join(os.path.dirname(__file__), f'memory_usage_{suffix}.png')
    else:
        output_path = OUTPUT_PATH

    for name, filepath in sorted(model_data.items()):
        times, rss = parse_csv(filepath)
        if len(times) == 0:
            continue

        color = COLORS.get(name, None)
        peak = rss.max()
        label = f'{name} (peak: {peak:.0f} MB / {peak/1024:.1f} GB)'
        ax.plot(times, rss, linewidth=1.5, label=label, color=color)

        # Detect phase boundary (F16 -> I2_S) by finding the biggest drop
        phase_boundary = None
        for i in range(1, len(rss)):
            if rss[i-1] > peak * 0.5 and rss[i] < peak * 0.3:
                phase_boundary = i
                break

        if phase_boundary is not None:
            t_split = times[phase_boundary]
            ax.axvline(x=t_split, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
            # F16 label at top
            ax.annotate('F16', xy=(t_split / 2, peak * 1.02),
                        ha='center', va='bottom', fontsize=11, fontweight='bold',
                        color=color, alpha=0.8)
            # I2_S label at top
            t_i2s_mid = t_split + (times[-1] - t_split) / 2
            y_i2s = rss[phase_boundary:].max()
            ax.annotate('I2_S', xy=(t_i2s_mid, y_i2s * 1.02),
                        ha='center', va='bottom', fontsize=11, fontweight='bold',
                        color=color, alpha=0.8)

        # Parse log file for sub-test annotations
        logpath = filepath.replace('.csv', '.log')
        events = parse_log(logpath)
        if events:
            # Filter only pp/tg events (these are completion timestamps)
            test_events = [(t, lbl) for t, lbl in events
                           if not lbl.startswith('__')]

            # Use alternating colors for adjacent regions
            bg_colors_pp = ['#E3F2FD', '#BBDEFB']  # light blue shades for prefill
            bg_colors_tg = ['#FFF3E0', '#FFE0B2']  # light orange shades for decode

            # Build time intervals: from previous event to current event
            # First interval starts at time 0 (or phase start)
            all_boundaries = [times[0]] + [t for t, _ in test_events]
            if phase_boundary is not None:
                # Also add phase split as a boundary
                pass

            pp_idx = 0
            tg_idx = 0
            for i, (t, lbl) in enumerate(test_events):
                # Determine start of this test's interval
                if i == 0:
                    t_start = times[0]
                else:
                    t_start = test_events[i-1][0]

                # Choose background color
                if lbl.startswith('pp'):
                    bg = bg_colors_pp[pp_idx % 2]
                    pp_idx += 1
                else:
                    bg = bg_colors_tg[tg_idx % 2]
                    tg_idx += 1

                # Fill background between intervals
                ax.axvspan(t_start, t, alpha=0.3, color=bg, zorder=0)

                # Draw vertical dashed line (thicker)
                ax.axvline(x=t, color='#555555', linestyle='--', alpha=0.6, linewidth=1.2)

                # Label at the middle of the interval, below x-axis
                t_mid = (t_start + t) / 2
                ax.annotate(lbl, xy=(t_mid, 0), xytext=(t_mid, -peak * 0.06),
                            ha='center', va='top', fontsize=9, fontweight='bold',
                            color='#222222', rotation=90,
                            annotation_clip=False)

    ax.set_xlabel('Time (seconds)', fontsize=11, labelpad=35)
    ax.set_ylabel('Memory RSS (MB)', fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Add secondary y-axis in GB
    ax2 = ax.secondary_yaxis('right', functions=(lambda x: x/1024, lambda x: x*1024))
    ax2.set_ylabel('Memory RSS (GB)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {output_path}")


if __name__ == '__main__':
    main()
