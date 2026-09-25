from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

# Source: Qwen3.6-35B-A3B，BitNet加速比测试 (7).pdf, sections 12 and 13.
# Section 12 measurements: PDF pages 19-20; section 13: PDF pages 21-22.
# The hybrid label follows the user's setup description; its table says Q4_0.
# Thread order: 4, 8, 12, 16, 32. Threads 1 and 2 have no measurements.
# Keep the source's "pp32" label; it appears between tg16 and tg64.

# ========== Section 13 / S1: original BF16 baseline ==========
# The table heading says F16.GGUF, but the conversion uses --outtype bf16.
s1_bf16 = {
    'pp128':  [23.87, 44.93, 62.88, 77.38, 111.29],
    'pp256':  [24.70, 46.82, 66.63, 81.22, 110.33],
    'pp512':  [25.06, 47.24, 66.79, 84.89, 114.64],
    'pp1024': [24.92, 47.22, 66.85, 85.63, 111.32],
    'tg16':   [5.69, 9.19, 11.30, 12.82, 13.95],
    'pp32':   [5.68, 9.22, 11.29, 12.95, 14.07],
    'tg64':   [5.71, 9.14, 11.33, 12.95, 14.10],
    'tg128':  [5.71, 9.15, 11.29, 12.95, 14.38],
    'tg256':  [5.69, 9.15, 11.25, 12.88, 14.31],
}
s1_bf16_err = {
    'pp128':  [0.11, 0.10, 1.72, 0.67, 0.41],
    'pp256':  [0.02, 0.04, 0.06, 0.28, 21.24],
    'pp512':  [0.12, 0.12, 0.14, 1.27, 0.35],
    'pp1024': [0.06, 0.12, 0.30, 0.24, 4.18],
    'tg16':   [0.01, 0.03, 0.02, 0.01, 0.09],
    'pp32':   [0.01, 0.03, 0.02, 0.01, 0.03],
    'tg64':   [0.02, 0.08, 0.02, 0.01, 0.40],
    'tg128':  [0.00, 0.03, 0.00, 0.01, 0.03],
    'tg256':  [0.01, 0.04, 0.01, 0.02, 0.08],
}

# ========== Section 13 / S1: Q4_0 quantized comparison ==========
# tg16 / Thread=16 uses the measured 25.35, not the expected 43 in the note.
s1_q4 = {
    'pp128':  [51.15, 86.85, 114.46, 150.83, 192.39],
    'pp256':  [51.30, 94.88, 123.62, 163.67, 192.56],
    'pp512':  [51.30, 93.18, 123.15, 161.54, 164.45],
    'pp1024': [50.74, 92.51, 121.88, 159.61, 164.29],
    'tg16':   [13.65, 19.70, 23.44, 25.35, 25.57],
    'pp32':   [13.66, 19.70, 23.55, 25.61, 25.30],
    'tg64':   [13.61, 19.76, 23.49, 25.54, 27.49],
    'tg128':  [13.67, 19.86, 23.56, 25.71, 27.48],
    'tg256':  [13.49, 19.49, 23.24, 25.52, 25.70],
}
s1_q4_err = {
    'pp128':  [0.09, 6.89, 6.17, 9.19, 3.91],
    'pp256':  [0.13, 0.47, 1.27, 0.50, 0.37],
    'pp512':  [0.01, 0.40, 0.41, 0.74, 0.68],
    'pp1024': [0.16, 0.24, 0.19, 1.20, 0.36],
    'tg16':   [0.02, 0.08, 0.04, 0.01, 0.12],
    'pp32':   [0.04, 0.05, 0.01, 0.06, 0.17],
    'tg64':   [0.03, 0.06, 0.05, 0.05, 0.04],
    'tg128':  [0.00, 0.23, 0.01, 0.02, 0.01],
    'tg256':  [0.01, 0.02, 0.00, 0.02, 0.01],
}

# ========== Section 12: Q4_0 + Expert BitNet ternary hybrid ==========
# Retain the reported tg256 / Thread=12 measurement: 13.33 +/- 10.17.
expert_bitnet = {
    'pp128':  [44.55, 80.55, 104.14, 147.56, 162.53],
    'pp256':  [45.66, 84.89, 118.26, 151.73, 159.10],
    'pp512':  [45.44, 83.92, 114.31, 146.78, 150.48],
    'pp1024': [45.00, 82.56, 112.72, 145.63, 161.75],
    'tg16':   [12.56, 18.47, 22.36, 25.16, 25.01],
    'pp32':   [12.52, 18.52, 22.55, 25.52, 25.24],
    'tg64':   [12.60, 18.49, 21.19, 25.72, 25.62],
    'tg128':  [12.56, 18.43, 20.18, 25.61, 26.33],
    'tg256':  [12.48, 18.29, 13.33, 25.39, 27.01],
}
expert_bitnet_err = {
    'pp128':  [0.17, 4.92, 8.30, 0.37, 3.65],
    'pp256':  [0.05, 0.28, 0.22, 0.19, 0.17],
    'pp512':  [0.13, 0.00, 0.07, 0.27, 0.34],
    'pp1024': [0.05, 0.25, 0.42, 0.42, 0.13],
    'tg16':   [0.06, 0.06, 0.09, 0.13, 0.03],
    'pp32':   [0.03, 0.08, 0.07, 0.02, 0.03],
    'tg64':   [0.01, 0.06, 1.25, 0.02, 0.02],
    'tg128':  [0.01, 0.12, 0.14, 0.01, 0.98],
    'tg256':  [0.02, 0.11, 10.17, 0.02, 0.43],
}

threads = [4, 8, 12, 16, 32]
pp_tasks = ['pp128', 'pp256', 'pp512', 'pp1024']
tg_tasks = ['tg16', 'pp32', 'tg64', 'tg128', 'tg256']


def draw_speedup_bracket(ax, x_pos, baseline, compared, color, width=0.18):
    """Connect measured means; retain compared / baseline even for slowdowns."""
    y_base, y_top = sorted((baseline, compared))
    ax.plot([x_pos - width/2, x_pos + width/2], [y_base, y_base],
            linestyle='--', color=color, linewidth=1.0, alpha=0.8)
    ax.plot([x_pos - width/2, x_pos + width/2], [y_top, y_top],
            linestyle='--', color=color, linewidth=1.0, alpha=0.8)
    ax.plot([x_pos, x_pos], [y_base, y_top],
            linestyle='--', color=color, linewidth=0.8, alpha=0.6)
    ax.text(x_pos, (y_base + y_top) / 2, f'{compared / baseline:.2f}x',
            ha='center', va='center', fontsize=9, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                      edgecolor=color, alpha=0.95, linewidth=0.5))


def plot_qwen36_bf16_vs_q4_0_vs_expert_bitnet():
    fig, axes = plt.subplots(2, 5, figsize=(32, 13))
    fig.suptitle(
        'Qwen3.6-35B-A3B — BF16 vs Q4_0 vs Q4_0 + Expert BitNet\n'
        'Speedup ratios: Q4_0 / BF16; Section 12 hybrid / Q4_0 (not BF16)',
        fontsize=17, fontweight='bold', y=0.985)

    speedup_q4_color = '#C0392B'
    speedup_hybrid_color = '#007D83'
    series = [
        ('Sec. 13 / S1 BF16 (baseline)', s1_bf16, s1_bf16_err, '#4472C4'),
        ('Sec. 13 / S1 Q4_0', s1_q4, s1_q4_err, '#ED7D31'),
        ('Sec. 12 Q4_0 + Expert BitNet', expert_bitnet, expert_bitnet_err, '#70AD47'),
    ]

    for t_idx, t in enumerate(threads):
        for row, (tasks, task_type) in enumerate([(pp_tasks, 'Prefill'), (tg_tasks, 'Decode')]):
            ax = axes[row, t_idx]
            x = np.arange(len(tasks))
            width = 0.24
            upper_values = []

            for offset, (label, data, errors, color) in zip([-width, 0, width], series):
                values = [data[task][t_idx] for task in tasks]
                yerr = [errors[task][t_idx] for task in tasks]
                ax.bar(x + offset, values, width, label=label, color=color,
                       edgecolor='black', linewidth=0.5, yerr=yerr, capsize=2,
                       error_kw=dict(ecolor='#333333', elinewidth=0.8, capthick=0.8))
                upper_values.extend(v + e for v, e in zip(values, yerr))

            for i, task in enumerate(tasks):
                baseline = s1_bf16[task][t_idx]
                q4 = s1_q4[task][t_idx]
                hybrid = expert_bitnet[task][t_idx]
                draw_speedup_bracket(ax, x[i] - width, baseline, q4, speedup_q4_color)
                draw_speedup_bracket(ax, x[i] + width, q4, hybrid, speedup_hybrid_color)

            ax.set_title(f'Thread={t} ({task_type})', fontsize=12, pad=9)
            ax.set_xticks(x)
            ax.set_xticklabels(['pp32*' if task == 'pp32' else task for task in tasks],
                               fontsize=10)
            ax.tick_params(axis='y', labelsize=10)
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(upper_values) * 1.15)
            if t_idx == 0:
                ax.set_ylabel('Throughput (tokens/s)', fontsize=11)

    legend_elements = [
        Patch(facecolor=color, edgecolor='black', linewidth=0.5, label=label)
        for label, _, _, color in series
    ] + [
        Line2D([0], [0], linestyle='--', color=speedup_q4_color, linewidth=1.5,
               label='Q4_0 / BF16'),
        Line2D([0], [0], linestyle='--', color=speedup_hybrid_color, linewidth=1.5,
               label='Sec. 12 hybrid / Q4_0'),
    ]
    fig.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, 0.923), ncol=5, fontsize=11, framealpha=0.95)
    fig.text(
        0.015, 0.085,
        'Sec. 12 hybrid label follows the requested setup description; its PDF table heading says Q4_0. '
        'S1: no MTP, TL2 off. Identical benchmark conditions are not established by the PDF.',
        fontsize=10, color='#444444')
    fig.text(
        0.015, 0.067,
        'Source: benchmark PDF (7), Sec. 12 pp. 19-20; Sec. 13 pp. 21-22. BF16 is the original baseline '
        '(table heading: F16.GGUF; conversion: --outtype bf16).',
        fontsize=10, color='#444444')
    fig.text(
        0.015, 0.049,
        'Error bars: reported ± values. Ratios use measured means at the same task/thread count; '
        'hybrid ratios use Q4_0 as baseline. Values below 1.00x indicate lower throughput.',
        fontsize=10, color='#444444')
    fig.text(
        0.015, 0.031,
        'Threads 1/2: no data. pp32*: source label retained, grouped with Decode by table position. '
        'S1 Q4_0 tg16 / Thread=16: measured 25.35 tokens/s, not the expected 43.',
        fontsize=10, color='#444444')
    fig.text(
        0.015, 0.013,
        'High-variability measurement retained: Sec. 12 hybrid tg256 / Thread=12 = '
        '13.33 ± 10.17 tokens/s (0.57x Q4_0); do not interpret this mean ratio as a stable slowdown.',
        fontsize=10, color='#444444')

    plt.tight_layout(rect=[0, 0.115, 1, 0.90])
    save_path = Path(__file__).resolve().parent / 'throughput_bar_qwen36_bf16_vs_q4_0_vs_expert_bitnet'
    for extension in ['png', 'pdf']:
        path = save_path.with_suffix(f'.{extension}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved {path}')
    plt.close(fig)


if __name__ == '__main__':
    plot_qwen36_bf16_vs_q4_0_vs_expert_bitnet()
