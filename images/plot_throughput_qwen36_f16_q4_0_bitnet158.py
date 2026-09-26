from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Source: v10：Qwen3.6-35B-A3B，BitNet加速比测试.pdf
# Plot the reported means, as in plot_throughput_yoco_moe_vs_qwen3_moe.py.
# Thread order: 4, 8, 12, 16, 32; threads 1 and 2 have no measurements.
# The source labels the decode-32 rows "pp32"; benchmark -n confirms tg32.

# ========== S1 baseline: section 3, PDF pages 4-5 ==========
# The table says F16, but the conversion command uses --outtype bf16.
# Retain the table label with an explicit footnote; do not equate F16 and BF16.
qwen36_f16 = {
    'pp128':  [23.87, 44.93, 62.88, 77.38, 111.29],
    'pp256':  [24.70, 46.82, 66.63, 81.22, 110.33],
    'pp512':  [25.06, 47.24, 66.79, 84.89, 114.64],
    'pp1024': [24.92, 47.22, 66.85, 85.63, 111.32],
    'tg16':   [5.69, 9.19, 11.30, 12.82, 13.95],
    'tg32':   [5.68, 9.22, 11.29, 12.95, 14.07],
    'tg64':   [5.71, 9.14, 11.33, 12.95, 14.10],
    'tg128':  [5.71, 9.15, 11.29, 12.95, 14.38],
    'tg256':  [5.69, 9.15, 11.25, 12.88, 14.31],
}

# ========== Q4_0: section 3, PDF pages 5-6 ==========
# The output head is Q6_K, as noted on PDF pages 6-7.
qwen36_q4_0 = {
    'pp128':  [51.15, 86.85, 114.46, 150.83, 192.39],
    'pp256':  [51.30, 94.88, 123.62, 163.67, 192.56],
    'pp512':  [51.30, 93.18, 123.15, 161.54, 164.45],
    'pp1024': [50.74, 92.51, 121.88, 159.61, 164.29],
    'tg16':   [13.65, 19.70, 23.44, 25.35, 25.57],
    'tg32':   [13.66, 19.70, 23.55, 25.61, 25.30],
    'tg64':   [13.61, 19.76, 23.49, 25.54, 27.49],
    'tg128':  [13.67, 19.86, 23.56, 25.71, 27.48],
    'tg256':  [13.49, 19.49, 23.24, 25.52, 25.70],
}

# ========== All-layer BitNet 1.58: section 13, PDF pages 20-21 ==========
# Page 19: main weights (including embedding and head) are I2_S.
# Page 20: 0.075433% of parameters remain F32; not every tensor is I2_S.
qwen36_bitnet158 = {
    'pp128':  [67.88, 103.23, 141.09, 209.89, 289.48],
    'pp256':  [68.68, 117.77, 171.94, 217.54, 285.03],
    'pp512':  [63.62, 114.76, 157.41, 198.42, 234.99],
    'pp1024': [62.76, 113.24, 157.57, 197.56, 215.73],
    'tg16':   [18.36, 25.79, 28.28, 27.25, 29.46],
    'tg32':   [17.40, 25.85, 29.65, 30.10, 28.42],
    'tg64':   [18.18, 25.18, 29.43, 30.08, 29.62],
    'tg128':  [18.02, 25.07, 29.31, 30.36, 29.47],
    'tg256':  [17.69, 24.91, 29.08, 30.26, 29.22],
}

threads = [4, 8, 12, 16, 32]
pp_tasks = ['pp128', 'pp256', 'pp512', 'pp1024']
tg_tasks = ['tg16', 'tg32', 'tg64', 'tg128', 'tg256']


def draw_speedup_bracket(ax, x_pos, y_base, y_top, speedup_text, color, width=0.18):
    """Draw a bracket between two bars with speedup text."""
    ax.plot([x_pos - width/2, x_pos + width/2], [y_base, y_base],
            linestyle='--', color=color, linewidth=1.0, alpha=0.8)
    ax.plot([x_pos - width/2, x_pos + width/2], [y_top, y_top],
            linestyle='--', color=color, linewidth=1.0, alpha=0.8)
    ax.plot([x_pos, x_pos], [y_base, y_top],
            linestyle='--', color=color, linewidth=0.8, alpha=0.6)
    y_mid = (y_base + y_top) / 2
    ax.text(x_pos, y_mid, speedup_text, ha='center', va='center',
            fontsize=8, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                      edgecolor=color, alpha=0.85, linewidth=0.5))


def plot_three_way():
    """Compare PDF throughput means; ratios are Q4_0/F16 and BitNet/Q4_0."""
    fig, axes = plt.subplots(2, 5, figsize=(32, 13))
    fig.suptitle(
        'Qwen3.6-35B-A3B — F16* vs Q4_0 vs BitNet 1.58 (I2_S) — Throughput (tokens/s)\n'
        'Speedup: Q4_0 / F16* | BitNet 1.58 / Q4_0',
        fontsize=14, fontweight='bold')

    colors = {'f16': '#4472C4', 'q4_0': '#70AD47', 'bitnet158': '#ED7D31'}
    speedup_q4_color = '#9B59B6'      # purple
    speedup_bitnet_color = '#E74C3C'  # red

    for t_idx, t in enumerate(threads):
        for row, (tasks, phase) in enumerate([(pp_tasks, 'Prefill'), (tg_tasks, 'Decode')]):
            ax = axes[row, t_idx]
            f16 = [qwen36_f16[task][t_idx] for task in tasks]
            q4 = [qwen36_q4_0[task][t_idx] for task in tasks]
            bitnet = [qwen36_bitnet158[task][t_idx] for task in tasks]

            x = np.arange(len(tasks))
            width = 0.24
            ax.bar(x - width, f16, width, label='F16* (S1 baseline)',
                   color=colors['f16'], edgecolor='black', linewidth=0.5)
            ax.bar(x, q4, width, label='Q4_0',
                   color=colors['q4_0'], edgecolor='black', linewidth=0.5)
            ax.bar(x + width, bitnet, width, label='BitNet 1.58 (I2_S, Sec. 13)',
                   color=colors['bitnet158'], edgecolor='black', linewidth=0.5)

            for i in range(len(tasks)):
                # Q4_0 vs F16: F16 is the baseline.
                draw_speedup_bracket(
                    ax, x[i] - width/2, min(f16[i], q4[i]), max(f16[i], q4[i]),
                    f'{q4[i] / f16[i]:.2f}x', speedup_q4_color, width=width)
                # BitNet 1.58 vs Q4_0: Q4_0 is the baseline.
                draw_speedup_bracket(
                    ax, x[i] + width/2, min(q4[i], bitnet[i]), max(q4[i], bitnet[i]),
                    f'{bitnet[i] / q4[i]:.2f}x', speedup_bitnet_color, width=width)

            ax.set_title(f'Thread={t} ({phase})', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(tasks, fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(f16 + q4 + bitnet) * 1.35)
            if t_idx == 0:
                ax.set_ylabel('Throughput (tokens/s)', fontsize=9)
                ax.legend(fontsize=8, loc='upper left')

    # Global legend for speedup colors.
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], linestyle='--', color=speedup_q4_color, linewidth=1.5,
               label='Q4_0 / F16*'),
        Line2D([0], [0], linestyle='--', color=speedup_bitnet_color, linewidth=1.5,
               label='BitNet 1.58 / Q4_0'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    fig.text(
        0.5, 0.025,
        '* F16 follows the PDF table label; the conversion command uses BF16. '
        'Q4_0 retains a Q6_K head; BitNet retains 0.075433% F32 parameters.\n'
        'Source: v10 report, Sec. 3 (pp. 4-6) and Sec. 13 (pp. 20-21). '
        'Means only (no error bars); ratios compare reported runs. '
        'Decode "pp32" in the source is shown as tg32.',
        ha='center', va='center', fontsize=9)

    save_path = Path(__file__).with_name('throughput_bar_qwen36_f16_q4_0_bitnet158.png')
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')


if __name__ == '__main__':
    plot_three_way()
