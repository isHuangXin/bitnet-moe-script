import matplotlib.pyplot as plt
import numpy as np

# ========== YOCO-MoE-30B.A3B (from existing data) ==========
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

# ========== YOCO-U-MoE-30B.A6B (from existing data) ==========
yoco_u_moe_f16 = {
    'pp128':  [10.92, 20.06, 29.02, 38.27, 59.27],
    'pp256':  [11.32, 21.76, 32.59, 40.98, 63.20],
    'pp512':  [11.43, 22.79, 33.38, 43.17, 63.16],
    'pp1024': [11.59, 22.92, 33.56, 41.86, 65.67],
    'tg16':   [2.88, 4.84, 6.38, 7.16, 8.04],
    'tg32':   [2.88, 5.05, 6.36, 7.20, 8.52],
    'tg64':   [2.85, 4.95, 6.33, 7.08, 8.64],
    'tg128':  [2.86, 4.97, 6.23, 7.13, 8.49],
    'tg256':  [2.85, 4.96, 6.21, 7.10, 8.58],
}

yoco_u_moe_i2s = {
    'pp128':  [35.74, 67.11, 92.92, 130.48, 187.45],
    'pp256':  [34.42, 65.58, 93.37, 125.94, 195.92],
    'pp512':  [33.15, 60.07, 85.51, 110.48, 164.68],
    'pp1024': [33.10, 61.43, 86.91, 109.82, 164.44],
    'tg16':   [11.08, 15.54, 23.42, 26.46, 28.08],
    'tg32':   [11.64, 19.94, 20.47, 26.53, 25.37],
    'tg64':   [10.98, 17.50, 21.55, 22.78, 26.15],
    'tg128':  [11.12, 17.56, 20.47, 23.65, 25.85],
    'tg256':  [10.99, 17.44, 20.76, 22.81, 25.74],
}

# ========== YOCO-U-Dense-6B (d_ffn=9216) ==========
yoco_u_dense_6b_f16 = {
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

yoco_u_dense_6b_i2s = {
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

threads = [4, 8, 12, 16, 32]
pp_tasks = ['pp128', 'pp256', 'pp512', 'pp1024']
tg_tasks = ['tg16', 'tg32', 'tg64', 'tg128', 'tg256']


def plot_moe_vs_dense(moe_f16, moe_i2s, dense_f16, dense_i2s,
                      moe_name, dense_name, title, save_path):
    """
    Compare MoE vs Dense: 4 bars per task (MoE-F16, MoE-I2S, Dense-F16, Dense-I2S).
    Each subplot = one thread count. Row 0 = Prefill, Row 1 = Decode.
    """
    fig, axes = plt.subplots(2, 5, figsize=(28, 11))
    fig.suptitle(title, fontsize=15, fontweight='bold')

    colors = {
        'moe_f16': '#4472C4',
        'moe_i2s': '#ED7D31',
        'dense_f16': '#70AD47',
        'dense_i2s': '#FFC000',
    }

    for t_idx, t in enumerate(threads):
        # Row 0: Prefill
        ax_pp = axes[0, t_idx]
        moe_f16_vals = [moe_f16[task][t_idx] for task in pp_tasks]
        moe_i2s_vals = [moe_i2s[task][t_idx] for task in pp_tasks]
        dense_f16_vals = [dense_f16[task][t_idx] for task in pp_tasks]
        dense_i2s_vals = [dense_i2s[task][t_idx] for task in pp_tasks]

        x = np.arange(len(pp_tasks))
        width = 0.2
        bars1 = ax_pp.bar(x - 1.5*width, moe_f16_vals, width, label=f'{moe_name} F16',
                          color=colors['moe_f16'], edgecolor='black', linewidth=0.5)
        bars2 = ax_pp.bar(x - 0.5*width, moe_i2s_vals, width, label=f'{moe_name} I2_S',
                          color=colors['moe_i2s'], edgecolor='black', linewidth=0.5)
        bars3 = ax_pp.bar(x + 0.5*width, dense_f16_vals, width, label=f'{dense_name} F16',
                          color=colors['dense_f16'], edgecolor='black', linewidth=0.5)
        bars4 = ax_pp.bar(x + 1.5*width, dense_i2s_vals, width, label=f'{dense_name} I2_S',
                          color=colors['dense_i2s'], edgecolor='black', linewidth=0.5)

        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                h = bar.get_height()
                ax_pp.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.0f}',
                           ha='center', va='bottom', fontsize=6, rotation=90)

        ax_pp.set_title(f'Thread={t} (Prefill)', fontsize=10)
        ax_pp.set_xticks(x)
        ax_pp.set_xticklabels(pp_tasks, fontsize=8)
        ax_pp.grid(True, alpha=0.3, axis='y')
        all_vals = moe_f16_vals + moe_i2s_vals + dense_f16_vals + dense_i2s_vals
        ax_pp.set_ylim(0, max(all_vals) * 1.3)
        if t_idx == 0:
            ax_pp.set_ylabel('Throughput (tokens/s)', fontsize=9)
            ax_pp.legend(fontsize=7, loc='upper left')

        # Row 1: Decode
        ax_tg = axes[1, t_idx]
        moe_f16_vals = [moe_f16[task][t_idx] for task in tg_tasks]
        moe_i2s_vals = [moe_i2s[task][t_idx] for task in tg_tasks]
        dense_f16_vals = [dense_f16[task][t_idx] for task in tg_tasks]
        dense_i2s_vals = [dense_i2s[task][t_idx] for task in tg_tasks]

        x = np.arange(len(tg_tasks))
        bars1 = ax_tg.bar(x - 1.5*width, moe_f16_vals, width, label=f'{moe_name} F16',
                          color=colors['moe_f16'], edgecolor='black', linewidth=0.5)
        bars2 = ax_tg.bar(x - 0.5*width, moe_i2s_vals, width, label=f'{moe_name} I2_S',
                          color=colors['moe_i2s'], edgecolor='black', linewidth=0.5)
        bars3 = ax_tg.bar(x + 0.5*width, dense_f16_vals, width, label=f'{dense_name} F16',
                          color=colors['dense_f16'], edgecolor='black', linewidth=0.5)
        bars4 = ax_tg.bar(x + 1.5*width, dense_i2s_vals, width, label=f'{dense_name} I2_S',
                          color=colors['dense_i2s'], edgecolor='black', linewidth=0.5)

        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                h = bar.get_height()
                ax_tg.text(bar.get_x() + bar.get_width()/2, h + 0.2, f'{h:.1f}',
                           ha='center', va='bottom', fontsize=6, rotation=90)

        ax_tg.set_title(f'Thread={t} (Decode)', fontsize=10)
        ax_tg.set_xticks(x)
        ax_tg.set_xticklabels(tg_tasks, fontsize=8)
        ax_tg.grid(True, alpha=0.3, axis='y')
        all_vals = moe_f16_vals + moe_i2s_vals + dense_f16_vals + dense_i2s_vals
        ax_tg.set_ylim(0, max(all_vals) * 1.3)
        if t_idx == 0:
            ax_tg.set_ylabel('Throughput (tokens/s)', fontsize=9)
            ax_tg.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")


# Plot 1: YOCO-MoE-30B vs YOCO-Dense-3B
plot_moe_vs_dense(
    yoco_moe_f16, yoco_moe_i2s,
    yoco_dense_3b_f16, yoco_dense_3b_i2s,
    moe_name='MoE-30B.A3B',
    dense_name='Dense-3B',
    title='YOCO-MoE-30B.A3B vs YOCO-Dense-3B (d_ffn=9216) - Throughput (tokens/s)',
    save_path='/home/huangxin/code_list/bitnet-moe-script/images/throughput_bar_yoco_moe_dense.png',
)

# Plot 2: YOCO-U-MoE-30B vs YOCO-U-Dense-6B
plot_moe_vs_dense(
    yoco_u_moe_f16, yoco_u_moe_i2s,
    yoco_u_dense_6b_f16, yoco_u_dense_6b_i2s,
    moe_name='U-MoE-30B.A6B',
    dense_name='U-Dense-6B',
    title='YOCO-U-MoE-30B.A6B vs YOCO-U-Dense-6B (d_ffn=9216) - Throughput (tokens/s)',
    save_path='/home/huangxin/code_list/bitnet-moe-script/images/throughput_bar_yoco_u_moe_dense.png',
)
