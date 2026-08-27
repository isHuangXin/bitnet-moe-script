import matplotlib.pyplot as plt
import numpy as np

# ========== YOCO-Dense-3B (d_ffn=9216) ==========
yoco_dense_3b_f16 = {
    'pp128':  [49.19, 121.97, 153.73, 197.96, 345.80],
    'pp256':  [49.38, 126.22, 170.12, 224.73, 354.22],
    'pp512':  [49.41, 127.23, 156.74, 244.11, 343.39],
    'pp1024': [47.73, 126.09, 124.11, 236.48, 337.57],
    'tg16':   [3.59, 7.99, 9.10, 9.16, 9.33],
    'tg32':   [3.67, 7.99, 9.13, 8.97, 9.32],
    'tg64':   [4.20, 8.03, 9.19, 7.88, 9.31],
    'tg128':  [4.47, 8.00, 9.18, 8.31, 9.29],
    'tg256':  [4.64, 7.95, 9.03, 9.17, 9.30],
}

yoco_dense_3b_i2s = {
    'pp128':  [135.45, 265.50, 378.29, 485.06, 605.59],
    'pp256':  [133.96, 249.80, 362.74, 441.73, 654.43],
    'pp512':  [123.82, 249.73, 350.67, 421.75, 526.41],
    'pp1024': [118.10, 215.68, 322.92, 398.46, 494.58],
    'tg16':   [19.13, 31.05, 37.60, 39.17, 38.64],
    'tg32':   [19.34, 30.82, 37.44, 39.32, 38.69],
    'tg64':   [19.23, 31.13, 37.23, 39.34, 38.54],
    'tg128':  [19.19, 30.89, 37.16, 39.17, 38.64],
    'tg256':  [17.89, 30.48, 36.75, 38.59, 38.26],
}

# ========== YOCO-Dense-6B (d_ffn=9216) ==========
yoco_dense_6b_f16 = {
    'pp128':  [32.49, 61.75, 91.52, 116.61, 167.17],
    'pp256':  [31.59, 62.76, 95.45, 120.77, 169.49],
    'pp512':  [31.74, 62.62, 93.55, 119.86, 165.93],
    'pp1024': [30.17, 60.67, 88.75, 114.84, 161.52],
    'tg16':   [2.68, 4.31, 4.75, 4.75, 4.71],
    'tg32':   [2.68, 4.32, 4.74, 4.76, 4.71],
    'tg64':   [2.68, 4.32, 4.73, 4.75, 4.72],
    'tg128':  [2.66, 4.31, 4.72, 4.74, 4.71],
    'tg256':  [2.65, 4.28, 4.70, 4.71, 4.69],
}

yoco_dense_6b_i2s = {
    'pp128':  [84.92, 145.08, 215.65, 254.28, 371.18],
    'pp256':  [79.58, 141.98, 196.08, 265.13, 332.27],
    'pp512':  [61.67, 111.55, 154.79, 197.59, 252.14],
    'pp1024': [56.47, 102.74, 146.89, 196.53, 244.40],
    'tg16':   [11.50, 18.55, 22.34, 23.91, 23.27],
    'tg32':   [11.56, 18.64, 22.48, 23.94, 22.98],
    'tg64':   [11.60, 18.48, 22.45, 23.77, 22.97],
    'tg128':  [11.43, 18.36, 22.15, 23.41, 22.87],
    'tg256':  [11.23, 17.92, 21.41, 22.77, 22.49],
}

# ========== Qwen3.8-Dense-27B ==========
# Thread=4 data is NOT available; threads=[4, 8, 12, 16, 32], index 0 = None
# F16.GGUF
qwen_dense_f16 = {
    'pp128':  [None, 4.04, 6.91, 7.28, 3.85],
    'pp256':  [None, 4.27, 6.92, 8.19, 5.28],
    'pp512':  [None, 4.09, 7.29, 8.24, 6.34],
    'pp1024': [None, 2.98, 7.29, 8.18, 6.26],
    'tg16':   [None, 3.22, 4.90, 3.99, 0.06],
    'tg32':   [None, 3.16, 4.93, 3.34, 0.09],
    'tg64':   [None, 2.66, 4.85, 4.04, 0.08],
    'tg128':  [None, 3.03, 4.93, 3.90, 0.12],
    'tg256':  [None, 3.10, 4.81, 3.95, 0.11],
}

# Q4_0.GGUF (used as quantized counterpart, similar to I2_S for YOCO)
qwen_dense_q4 = {
    'pp128':  [None, 22.96, 42.46, 38.79, 6.66],
    'pp256':  [None, 24.55, 41.93, 41.33, 11.79],
    'pp512':  [None, 23.38, 40.83, 38.57, 15.90],
    'pp1024': [None, 23.98, 40.37, 39.07, 16.50],
    'tg16':   [None, 8.92, 14.29, 10.67, 0.06],
    'tg32':   [None, 9.33, 14.66, 10.93, 0.08],
    'tg64':   [None, 7.94, 14.73, 10.90, 0.09],
    'tg128':  [None, 8.82, 14.53, 10.51, 0.10],
    'tg256':  [None, 9.00, 14.95, 9.57, 0.09],
}

threads = [4, 8, 12, 16, 32]
pp_tasks = ['pp128', 'pp256', 'pp512', 'pp1024']
tg_tasks = ['tg16', 'tg32', 'tg64', 'tg128', 'tg256']


def draw_speedup_bracket(ax, x_pos, y_base, y_top, speedup_text, color, width=0.18):
    """Draw dashed lines at top and bottom with speedup text in the middle."""
    ax.plot([x_pos - width/2, x_pos + width/2], [y_base, y_base],
            linestyle='--', color=color, linewidth=1.0, alpha=0.8)
    ax.plot([x_pos - width/2, x_pos + width/2], [y_top, y_top],
            linestyle='--', color=color, linewidth=1.0, alpha=0.8)
    ax.plot([x_pos, x_pos], [y_base, y_top],
            linestyle='--', color=color, linewidth=0.8, alpha=0.6)
    y_mid = (y_base + y_top) / 2
    ax.text(x_pos, y_mid, speedup_text, ha='center', va='center',
            fontsize=6, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                      edgecolor=color, alpha=0.85, linewidth=0.5))


def safe_val(data, task, t_idx):
    """Return value or None if missing."""
    v = data[task][t_idx]
    return v if v is not None else None


def plot_qwen_vs_yoco_dense():
    """
    Compare Qwen3.8-Dense-27B vs YOCO-Dense-3B vs YOCO-Dense-6B.
    6 bars: YOCO-3B F16, YOCO-3B I2_S, YOCO-6B F16, YOCO-6B I2_S, Qwen F16, Qwen Q4_0
    Speedup brackets: Qwen / YOCO-Dense-3B (as baseline)
    """
    fig, axes = plt.subplots(2, 5, figsize=(32, 13))
    fig.suptitle(
        'Qwen3.8-Dense-27B vs YOCO-Dense-3B vs YOCO-Dense-6B — Throughput (tokens/s)\n'
        'Speedup ratio: Qwen3.8-Dense-27B / YOCO-Dense-3B (YOCO-Dense-3B as baseline)',
        fontsize=14, fontweight='bold')

    colors = {
        'yoco3b_f16':  '#4472C4',   # blue
        'yoco3b_i2s':  '#ED7D31',   # orange
        'yoco6b_f16':  '#70AD47',   # green
        'yoco6b_i2s':  '#FFC000',   # yellow
        'qwen_f16':    '#A5A5A5',   # gray
        'qwen_q4':     '#5B9BD5',   # light blue
    }
    speedup_f16_color = '#9B59B6'    # purple - F16 ratio
    speedup_quant_color = '#E74C3C'  # red - quantized ratio

    for t_idx, t in enumerate(threads):
        for row, (tasks, task_type) in enumerate([(pp_tasks, 'Prefill'), (tg_tasks, 'Decode')]):
            ax = axes[row, t_idx]

            y3b_f16 = [yoco_dense_3b_f16[task][t_idx] for task in tasks]
            y3b_i2s = [yoco_dense_3b_i2s[task][t_idx] for task in tasks]
            y6b_f16 = [yoco_dense_6b_f16[task][t_idx] for task in tasks]
            y6b_i2s = [yoco_dense_6b_i2s[task][t_idx] for task in tasks]
            qw_f16  = [safe_val(qwen_dense_f16, task, t_idx) for task in tasks]
            qw_q4   = [safe_val(qwen_dense_q4, task, t_idx) for task in tasks]

            x = np.arange(len(tasks))
            width = 0.13

            # Draw all 6 bars
            ax.bar(x - 2.5*width, y3b_f16, width, label='YOCO-Dense-3B F16',
                   color=colors['yoco3b_f16'], edgecolor='black', linewidth=0.5)
            ax.bar(x - 1.5*width, y3b_i2s, width, label='YOCO-Dense-3B I2_S',
                   color=colors['yoco3b_i2s'], edgecolor='black', linewidth=0.5)
            ax.bar(x - 0.5*width, y6b_f16, width, label='YOCO-Dense-6B F16',
                   color=colors['yoco6b_f16'], edgecolor='black', linewidth=0.5)
            ax.bar(x + 0.5*width, y6b_i2s, width, label='YOCO-Dense-6B I2_S',
                   color=colors['yoco6b_i2s'], edgecolor='black', linewidth=0.5)

            # Qwen bars: handle None values
            qw_f16_plot = [v if v is not None else 0 for v in qw_f16]
            qw_q4_plot  = [v if v is not None else 0 for v in qw_q4]
            ax.bar(x + 1.5*width, qw_f16_plot, width, label='Qwen3.8-27B F16',
                   color=colors['qwen_f16'], edgecolor='black', linewidth=0.5)
            ax.bar(x + 2.5*width, qw_q4_plot, width, label='Qwen3.8-27B Q4_0',
                   color=colors['qwen_q4'], edgecolor='black', linewidth=0.5)

            # Speedup brackets: Qwen / YOCO-Dense-3B
            for i in range(len(tasks)):
                # F16 ratio
                if qw_f16[i] is not None and y3b_f16[i] > 0:
                    spd = qw_f16[i] / y3b_f16[i]
                    base = min(qw_f16[i], y3b_f16[i])
                    top = max(qw_f16[i], y3b_f16[i])
                    mid_x = (x[i] - 2.5*width + x[i] + 1.5*width) / 2
                    draw_speedup_bracket(ax, mid_x, base, top,
                                         f'{spd:.2f}x', speedup_f16_color, width=0.28)

                # Quantized ratio: Qwen Q4_0 / YOCO-Dense-3B I2_S
                if qw_q4[i] is not None and y3b_i2s[i] > 0:
                    spd = qw_q4[i] / y3b_i2s[i]
                    base = min(qw_q4[i], y3b_i2s[i])
                    top = max(qw_q4[i], y3b_i2s[i])
                    mid_x = (x[i] - 1.5*width + x[i] + 2.5*width) / 2
                    draw_speedup_bracket(ax, mid_x, base, top,
                                         f'{spd:.2f}x', speedup_quant_color, width=0.28)

            ax.set_title(f'Thread={t} ({task_type})', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(tasks, fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            all_vals = y3b_f16 + y3b_i2s + y6b_f16 + y6b_i2s + \
                       [v for v in qw_f16 if v is not None] + \
                       [v for v in qw_q4 if v is not None]
            if all_vals:
                ax.set_ylim(0, max(all_vals) * 1.35)
            if t_idx == 0:
                ax.set_ylabel('Throughput (tokens/s)', fontsize=9)
                ax.legend(fontsize=5.5, loc='upper left')

    # Global legend for speedup colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], linestyle='--', color=speedup_f16_color, linewidth=1.5,
               label='F16 ratio (Qwen3.8-27B / YOCO-Dense-3B)'),
        Line2D([0], [0], linestyle='--', color=speedup_quant_color, linewidth=1.5,
               label='Quant ratio (Qwen Q4_0 / YOCO-Dense-3B I2_S)'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

    save_path = '/home/huangxin/code_list/bitnet-moe-script/images/throughput_bar_qwen_vs_yoco_dense.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")


if __name__ == '__main__':
    plot_qwen_vs_yoco_dense()
