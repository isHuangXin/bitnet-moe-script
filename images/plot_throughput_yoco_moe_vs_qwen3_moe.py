import matplotlib.pyplot as plt
import numpy as np

# ========== YOCO-MoE-30B.A3B (BitNet) ==========
yoco_moe_f16 = {
    'pp128':  [20.93, 41.63, 60.11, 82.89, 105.55],
    'pp256':  [22.53, 45.06, 65.63, 87.57, 122.81],
    'pp512':  [23.41, 46.26, 66.89, 89.70, 125.85],
    'pp1024': [24.09, 47.06, 69.95, 90.42, 128.33],
    'tg16':   [5.73, 9.59, 12.30, 14.23, 17.42],
    'tg32':   [5.38, 10.02, 12.86, 15.61, 16.43],
    'tg64':   [5.66, 9.58, 12.93, 15.17, 16.90],
    'tg128':  [5.75, 9.61, 12.87, 14.73, 17.93],
    'tg256':  [5.65, 9.47, 12.70, 14.77, 17.91],
}

yoco_moe_i2s = {
    'pp128':  [70.39, 132.31, 180.65, 260.51, 306.75],
    'pp256':  [68.58, 129.64, 180.26, 232.34, 353.81],
    'pp512':  [69.13, 133.48, 181.88, 224.57, 335.62],
    'pp1024': [65.72, 130.12, 173.25, 229.23, 336.66],
    'tg16':   [20.29, 32.21, 39.15, 43.85, 45.80],
    'tg32':   [17.79, 26.15, 39.44, 43.90, 48.12],
    'tg64':   [17.33, 27.56, 35.62, 39.39, 40.99],
    'tg128':  [17.91, 28.79, 34.47, 39.60, 42.47],
    'tg256':  [17.94, 29.09, 34.95, 38.63, 40.85],
}

# ========== YOCO-Dense-3B (d_ffn=9216, BitNet) ==========
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

# ========== Qwen3-MoE-30B.A3B (upstream llama.cpp) ==========
qwen3_f16 = {
    'pp128':  [25.28, 48.41, 71.78, 90.80, 122.83],
    'pp256':  [28.09, 54.41, 79.68, 102.09, 144.65],
    'pp512':  [26.99, 55.11, 81.14, 104.52, 150.41],
    'pp1024': [26.47, 53.66, 79.12, 102.67, 145.93],
    'tg16':   [6.10, 9.94, 12.03, 12.95, 13.95],
    'tg32':   [6.19, 9.96, 12.03, 12.98, 13.81],
    'tg64':   [6.06, 9.97, 12.00, 12.98, 13.86],
    'tg128':  [6.07, 9.95, 11.93, 12.93, 13.80],
    'tg256':  [6.02, 9.83, 11.81, 12.84, 13.83],
}

qwen3_q4_0 = {
    'pp128':  [49.00, 90.98, 125.99, 148.00, 178.48],
    'pp256':  [49.96, 97.91, 139.25, 181.48, 258.09],
    'pp512':  [49.01, 96.48, 134.91, 181.24, 251.21],
    'pp1024': [47.05, 92.12, 130.17, 174.25, 246.48],
    'tg16':   [17.94, 29.14, 37.17, 43.43, 48.72],
    'tg32':   [18.45, 29.51, 37.46, 43.49, 49.23],
    'tg64':   [18.65, 29.47, 37.77, 43.56, 48.98],
    'tg128':  [18.27, 28.79, 36.97, 42.54, 48.26],
    'tg256':  [17.90, 27.96, 35.95, 41.12, 47.26],
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
            fontsize=6, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                      edgecolor=color, alpha=0.85, linewidth=0.5))


def plot_three_way():
    """
    Compare YOCO-MoE-30B.A3B vs YOCO-Dense-3B vs Qwen3-MoE-30B.A3B.
    6 bars per group.
    Speedup brackets: YOCO-MoE I2_S / Qwen3 Q4_0 (red), YOCO-MoE F16 / Qwen3 F16 (purple)
    """
    fig, axes = plt.subplots(2, 5, figsize=(32, 13))
    fig.suptitle(
        'YOCO-MoE-30B.A3B vs YOCO-Dense-3B vs Qwen3-MoE-30B.A3B — Throughput (tokens/s)\n'
        'Speedup: YOCO-MoE / Qwen3-MoE (Qwen3 as baseline)',
        fontsize=14, fontweight='bold')

    colors = {
        'yoco_moe_f16':    '#4472C4',   # blue
        'yoco_moe_i2s':    '#ED7D31',   # orange
        'yoco_dense_f16':  '#A5A5A5',   # gray
        'yoco_dense_i2s':  '#FFC000',   # yellow
        'qwen3_f16':       '#70AD47',   # green
        'qwen3_q4':        '#5B9BD5',   # light blue
    }
    speedup_f16_color = '#9B59B6'    # purple
    speedup_quant_color = '#E74C3C'  # red

    for t_idx, t in enumerate(threads):
        # ===== Row 0: Prefill =====
        ax_pp = axes[0, t_idx]
        ym_f16  = [yoco_moe_f16[task][t_idx] for task in pp_tasks]
        ym_i2s  = [yoco_moe_i2s[task][t_idx] for task in pp_tasks]
        yd_f16  = [yoco_dense_3b_f16[task][t_idx] for task in pp_tasks]
        yd_i2s  = [yoco_dense_3b_i2s[task][t_idx] for task in pp_tasks]
        q3_f16  = [qwen3_f16[task][t_idx] for task in pp_tasks]
        q3_q4   = [qwen3_q4_0[task][t_idx] for task in pp_tasks]

        x = np.arange(len(pp_tasks))
        width = 0.13
        ax_pp.bar(x - 2.5*width, ym_f16, width, label='YOCO-MoE F16',
                  color=colors['yoco_moe_f16'], edgecolor='black', linewidth=0.5)
        ax_pp.bar(x - 1.5*width, ym_i2s, width, label='YOCO-MoE I2_S',
                  color=colors['yoco_moe_i2s'], edgecolor='black', linewidth=0.5)
        ax_pp.bar(x - 0.5*width, yd_f16, width, label='YOCO-Dense-3B F16',
                  color=colors['yoco_dense_f16'], edgecolor='black', linewidth=0.5)
        ax_pp.bar(x + 0.5*width, yd_i2s, width, label='YOCO-Dense-3B I2_S',
                  color=colors['yoco_dense_i2s'], edgecolor='black', linewidth=0.5)
        ax_pp.bar(x + 1.5*width, q3_f16, width, label='Qwen3-MoE F16',
                  color=colors['qwen3_f16'], edgecolor='black', linewidth=0.5)
        ax_pp.bar(x + 2.5*width, q3_q4, width, label='Qwen3-MoE Q4_0',
                  color=colors['qwen3_q4'], edgecolor='black', linewidth=0.5)

        # Speedup brackets: YOCO-MoE vs Qwen3
        for i in range(len(pp_tasks)):
            # F16: YOCO-MoE F16 vs Qwen3 F16
            base = min(ym_f16[i], q3_f16[i])
            top = max(ym_f16[i], q3_f16[i])
            spd = ym_f16[i] / q3_f16[i]
            mid_x = (x[i] - 2.5*width + x[i] + 1.5*width) / 2
            draw_speedup_bracket(ax_pp, mid_x, base, top,
                                 f'{spd:.2f}x', speedup_f16_color, width=0.28)

            # Quant: YOCO-MoE I2_S vs Qwen3 Q4_0
            base = min(ym_i2s[i], q3_q4[i])
            top = max(ym_i2s[i], q3_q4[i])
            spd = ym_i2s[i] / q3_q4[i]
            mid_x = (x[i] - 1.5*width + x[i] + 2.5*width) / 2
            draw_speedup_bracket(ax_pp, mid_x, base, top,
                                 f'{spd:.2f}x', speedup_quant_color, width=0.28)

        ax_pp.set_title(f'Thread={t} (Prefill)', fontsize=10)
        ax_pp.set_xticks(x)
        ax_pp.set_xticklabels(pp_tasks, fontsize=8)
        ax_pp.grid(True, alpha=0.3, axis='y')
        all_vals = ym_f16 + ym_i2s + yd_f16 + yd_i2s + q3_f16 + q3_q4
        ax_pp.set_ylim(0, max(all_vals) * 1.35)
        if t_idx == 0:
            ax_pp.set_ylabel('Throughput (tokens/s)', fontsize=9)
            ax_pp.legend(fontsize=5.5, loc='upper left')

        # ===== Row 1: Decode =====
        ax_tg = axes[1, t_idx]
        ym_f16  = [yoco_moe_f16[task][t_idx] for task in tg_tasks]
        ym_i2s  = [yoco_moe_i2s[task][t_idx] for task in tg_tasks]
        yd_f16  = [yoco_dense_3b_f16[task][t_idx] for task in tg_tasks]
        yd_i2s  = [yoco_dense_3b_i2s[task][t_idx] for task in tg_tasks]
        q3_f16  = [qwen3_f16[task][t_idx] for task in tg_tasks]
        q3_q4   = [qwen3_q4_0[task][t_idx] for task in tg_tasks]

        x = np.arange(len(tg_tasks))
        ax_tg.bar(x - 2.5*width, ym_f16, width, label='YOCO-MoE F16',
                  color=colors['yoco_moe_f16'], edgecolor='black', linewidth=0.5)
        ax_tg.bar(x - 1.5*width, ym_i2s, width, label='YOCO-MoE I2_S',
                  color=colors['yoco_moe_i2s'], edgecolor='black', linewidth=0.5)
        ax_tg.bar(x - 0.5*width, yd_f16, width, label='YOCO-Dense-3B F16',
                  color=colors['yoco_dense_f16'], edgecolor='black', linewidth=0.5)
        ax_tg.bar(x + 0.5*width, yd_i2s, width, label='YOCO-Dense-3B I2_S',
                  color=colors['yoco_dense_i2s'], edgecolor='black', linewidth=0.5)
        ax_tg.bar(x + 1.5*width, q3_f16, width, label='Qwen3-MoE F16',
                  color=colors['qwen3_f16'], edgecolor='black', linewidth=0.5)
        ax_tg.bar(x + 2.5*width, q3_q4, width, label='Qwen3-MoE Q4_0',
                  color=colors['qwen3_q4'], edgecolor='black', linewidth=0.5)

        # Speedup brackets: YOCO-MoE vs Qwen3
        for i in range(len(tg_tasks)):
            # F16
            base = min(ym_f16[i], q3_f16[i])
            top = max(ym_f16[i], q3_f16[i])
            spd = ym_f16[i] / q3_f16[i]
            mid_x = (x[i] - 2.5*width + x[i] + 1.5*width) / 2
            draw_speedup_bracket(ax_tg, mid_x, base, top,
                                 f'{spd:.2f}x', speedup_f16_color, width=0.28)

            # Quant
            base = min(ym_i2s[i], q3_q4[i])
            top = max(ym_i2s[i], q3_q4[i])
            spd = ym_i2s[i] / q3_q4[i]
            mid_x = (x[i] - 1.5*width + x[i] + 2.5*width) / 2
            draw_speedup_bracket(ax_tg, mid_x, base, top,
                                 f'{spd:.2f}x', speedup_quant_color, width=0.28)

        ax_tg.set_title(f'Thread={t} (Decode)', fontsize=10)
        ax_tg.set_xticks(x)
        ax_tg.set_xticklabels(tg_tasks, fontsize=8)
        ax_tg.grid(True, alpha=0.3, axis='y')
        all_vals = ym_f16 + ym_i2s + yd_f16 + yd_i2s + q3_f16 + q3_q4
        ax_tg.set_ylim(0, max(all_vals) * 1.35)
        if t_idx == 0:
            ax_tg.set_ylabel('Throughput (tokens/s)', fontsize=9)
            ax_tg.legend(fontsize=5.5, loc='upper left')

    # Global legend for speedup colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], linestyle='--', color=speedup_f16_color, linewidth=1.5,
               label='F16 speedup (YOCO-MoE / Qwen3)'),
        Line2D([0], [0], linestyle='--', color=speedup_quant_color, linewidth=1.5,
               label='Quant speedup (YOCO-MoE I2_S / Qwen3 Q4_0)'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

    save_path = '/home/huangxin/code_list/bitnet-moe-script/images/throughput_bar_yoco_moe_vs_qwen3_moe.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")


if __name__ == '__main__':
    plot_three_way()
