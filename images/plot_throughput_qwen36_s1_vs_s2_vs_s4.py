from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

# Source: Qwen3.6-35B-A3B，BitNet加速比测试 (2).pdf, S1/S2/S4 tables.
# Thread order: 4, 8, 12, 16, 32. Threads 1 and 2 have no measurements.
# Keep the source's "pp32" label; it appears between tg16 and tg64.

# ========== S1: original BF16 baseline ==========
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

# ========== S1: Q4_0 quantized comparison ==========
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

# ========== S2: routed experts ternary, embedding/output head INT8 ==========
# Other main projections remain BF16.
s2 = {
    'pp128':  [27.29, 51.95, 70.83, 89.76, 99.56],
    'pp256':  [28.04, 53.80, 75.92, 95.75, 98.94],
    'pp512':  [28.39, 54.03, 76.21, 96.71, 97.98],
    'pp1024': [28.16, 53.88, 76.10, 97.17, 97.49],
    'tg16':   [5.34, 8.65, 11.33, 13.41, 15.42],
    'pp32':   [5.32, 8.69, 11.40, 13.38, 15.46],
    'tg64':   [5.35, 8.70, 11.40, 13.58, 15.21],
    'tg128':  [5.34, 8.65, 11.34, 13.43, 14.94],
    'tg256':  [5.31, 8.58, 11.14, 13.32, 15.21],
}
s2_err = {
    'pp128':  [0.03, 0.34, 3.93, 2.18, 0.90],
    'pp256':  [0.02, 0.12, 0.26, 0.58, 0.10],
    'pp512':  [0.06, 0.05, 0.08, 0.57, 0.62],
    'pp1024': [0.12, 0.05, 0.26, 0.58, 0.13],
    'tg16':   [0.01, 0.02, 0.03, 0.02, 0.02],
    'pp32':   [0.00, 0.03, 0.02, 0.10, 0.04],
    'tg64':   [0.00, 0.00, 0.02, 0.07, 0.31],
    'tg128':  [0.01, 0.00, 0.01, 0.01, 0.62],
    'tg256':  [0.00, 0.01, 0.00, 0.01, 0.05],
}

# ========== S4: S2 + ternary full-attention Q/K/V/O projections ==========
# PDF pages 10-11. Shared experts/GDN remain BF16; compare S4 against S2.
s4 = {
    'pp128':  [26.92, 51.25, 70.84, 87.63, 97.92],
    'pp256':  [27.58, 53.08, 75.44, 95.57, 98.75],
    'pp512':  [27.93, 52.47, 75.98, 97.20, 90.65],
    'pp1024': [27.83, 53.20, 74.90, 96.42, 96.30],
    'tg16':   [4.88, 7.88, 10.32, 12.12, 13.35],
    'pp32':   [4.88, 7.92, 10.39, 11.99, 13.40],
    'tg64':   [4.92, 7.91, 10.40, 12.15, 13.27],
    'tg128':  [4.87, 7.93, 10.39, 12.29, 13.30],
    'tg256':  [4.83, 7.83, 10.27, 12.04, 13.26],
}
s4_err = {
    'pp128':  [0.16, 1.06, 3.16, 5.16, 0.22],
    'pp256':  [0.10, 0.26, 0.26, 0.17, 0.22],
    'pp512':  [0.07, 1.01, 0.15, 0.43, 5.70],
    'pp1024': [0.09, 0.15, 0.02, 0.40, 0.40],
    'tg16':   [0.01, 0.02, 0.01, 0.03, 0.09],
    'pp32':   [0.01, 0.01, 0.04, 0.10, 0.01],
    'tg64':   [0.00, 0.01, 0.01, 0.04, 0.01],
    'tg128':  [0.02, 0.01, 0.00, 0.04, 0.06],
    'tg256':  [0.01, 0.02, 0.02, 0.17, 0.02],
}

threads = [4, 8, 12, 16, 32]
pp_tasks = ['pp128', 'pp256', 'pp512', 'pp1024']
tg_tasks = ['tg16', 'pp32', 'tg64', 'tg128', 'tg256']


def draw_speedup_bracket(ax, x_pos, y_base, y_top, speedup_text, color, width=0.18):
    """Draw dashed lines at top and bottom with speedup text in the middle."""
    ax.plot([x_pos - width/2, x_pos + width/2], [y_base, y_base],
            linestyle='--', color=color, linewidth=1.0, alpha=0.8)
    ax.plot([x_pos - width/2, x_pos + width/2], [y_top, y_top],
            linestyle='--', color=color, linewidth=1.0, alpha=0.8)
    ax.plot([x_pos, x_pos], [y_base, y_top],
            linestyle='--', color=color, linewidth=0.8, alpha=0.6)
    ax.text(x_pos, (y_base + y_top) / 2, speedup_text,
            ha='center', va='center', fontsize=8, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                      edgecolor=color, alpha=0.95, linewidth=0.5))


def plot_qwen36_s1_vs_s2_vs_s4():
    fig, axes = plt.subplots(2, 5, figsize=(32, 13))
    fig.suptitle(
        'Qwen3.6-35B-A3B — S1 vs S2 vs S4 Throughput (tokens/s)\n'
        'Speedup ratios: S1 Q4_0 / S1 BF16; S2 / S1 BF16; S4 / S2 (not S1)',
        fontsize=17, fontweight='bold', y=0.985)

    colors = {'bf16': '#4472C4', 'q4': '#ED7D31', 's2': '#70AD47', 's4': '#FFC000'}
    speedup_q4_color = '#E74C3C'
    speedup_s2_color = '#9B59B6'
    speedup_s4_color = '#008C95'
    series = [
        ('S1 BF16 (baseline)', s1_bf16, s1_bf16_err, colors['bf16']),
        ('S1 Q4_0', s1_q4, s1_q4_err, colors['q4']),
        ('S2 ternary + INT8 / BF16', s2, s2_err, colors['s2']),
        ('S4 S2 + ternary full attention', s4, s4_err, colors['s4']),
    ]

    for t_idx, t in enumerate(threads):
        for row, (tasks, task_type) in enumerate([(pp_tasks, 'Prefill'), (tg_tasks, 'Decode')]):
            ax = axes[row, t_idx]
            x = np.arange(len(tasks))
            width = 0.18
            upper_values = []

            offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
            for offset, (label, data, errors, color) in zip(offsets, series):
                values = [data[task][t_idx] for task in tasks]
                yerr = [errors[task][t_idx] for task in tasks]
                ax.bar(x + offset, values, width, label=label, color=color,
                       edgecolor='black', linewidth=0.5, yerr=yerr, capsize=2,
                       error_kw=dict(ecolor='#333333', elinewidth=0.8, capthick=0.8))
                upper_values.extend(v + e for v, e in zip(values, yerr))

            for i, task in enumerate(tasks):
                baseline = s1_bf16[task][t_idx]
                q4 = s1_q4[task][t_idx]
                s2_value = s2[task][t_idx]
                s4_value = s4[task][t_idx]
                draw_speedup_bracket(
                    ax, x[i] - width, min(baseline, q4), max(baseline, q4),
                    f'{q4 / baseline:.2f}x', speedup_q4_color)
                draw_speedup_bracket(
                    ax, x[i] - width, min(baseline, s2_value), max(baseline, s2_value),
                    f'{s2_value / baseline:.2f}x', speedup_s2_color)
                draw_speedup_bracket(
                    ax, x[i] + width, min(s2_value, s4_value), max(s2_value, s4_value),
                    f'{s4_value / s2_value:.2f}x', speedup_s4_color)

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
               label='S1 Q4_0 / S1 BF16'),
        Line2D([0], [0], linestyle='--', color=speedup_s2_color, linewidth=1.5,
               label='S2 / S1 BF16'),
        Line2D([0], [0], linestyle='--', color=speedup_s4_color, linewidth=1.5,
               label='S4 / S2'),
    ]
    fig.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, 0.923), ncol=7, fontsize=11, framealpha=0.95)
    fig.text(
        0.015, 0.050,
        'S2: routed experts ternary; embedding + output head INT8; other main projections BF16. '
        'S4: S2 + W2A16 full-attention Q/K/V/O (including Q gate rows); shared experts/GDN remain BF16.',
        fontsize=10, color='#444444')
    fig.text(
        0.015, 0.031,
        'S1: no MTP, TL2 off. Error bars: reported ± values; ratios use measured means. '
        'S4 ratios use S2 as the baseline, NOT S1.',
        fontsize=10, color='#444444')
    fig.text(
        0.015, 0.012,
        'Threads 1/2: no data. pp32*: source label retained, shown with Decode rows by its table position. '
        'S1 Q4_0 tg16 / Thread=16: measured 25.35 tokens/s (not the expected 43).',
        fontsize=10, color='#444444')

    plt.tight_layout(rect=[0, 0.080, 1, 0.945])
    save_path = Path(__file__).resolve().parent / 'throughput_bar_qwen36_s1_vs_s2_vs_s4'
    for extension in ['png', 'pdf']:
        path = save_path.with_suffix(f'.{extension}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved {path}')
    plt.close(fig)


if __name__ == '__main__':
    plot_qwen36_s1_vs_s2_vs_s4()
