import matplotlib.pyplot as plt
import numpy as np

# ========== YOCO-MoE-30B.A3B ==========
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

# ========== YOCO-U-MoE-30B.A6B ==========
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

threads = [4, 8, 12, 16, 32]
pp_tasks = ['pp128', 'pp256', 'pp512', 'pp1024']
tg_tasks = ['tg16', 'tg32', 'tg64', 'tg128', 'tg256']
all_tasks = pp_tasks + tg_tasks

def plot_model_throughput(f16_data, i2s_data, model_name, save_path):
    """Each subplot = one thread count, x-axis = all sequence lengths, bars = F16 vs I2_S"""
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    fig.suptitle(f'{model_name} - Throughput (tokens/s)', fontsize=14, fontweight='bold')

    for t_idx, t in enumerate(threads):
        # Row 0: Prefill (pp)
        ax_pp = axes[0, t_idx]
        f16_vals = [f16_data[task][t_idx] for task in pp_tasks]
        i2s_vals = [i2s_data[task][t_idx] for task in pp_tasks]

        x = np.arange(len(pp_tasks))
        width = 0.35
        bars1 = ax_pp.bar(x - width/2, f16_vals, width, label='F16', color='#4472C4', edgecolor='black', linewidth=0.5)
        bars2 = ax_pp.bar(x + width/2, i2s_vals, width, label='I2_S', color='#ED7D31', edgecolor='black', linewidth=0.5)

        for bar in bars1:
            h = bar.get_height()
            ax_pp.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.1f}', ha='center', va='bottom', fontsize=7)
        for bar, f16_v, i2s_v in zip(bars2, f16_vals, i2s_vals):
            h = bar.get_height()
            speedup = i2s_v / f16_v
            ax_pp.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.1f}\n({speedup:.1f}x)', ha='center', va='bottom', fontsize=7, color='red')

        ax_pp.set_title(f'Thread={t} (Prefill)', fontsize=10)
        ax_pp.set_xticks(x)
        ax_pp.set_xticklabels(pp_tasks, fontsize=8)
        ax_pp.grid(True, alpha=0.3, axis='y')
        ax_pp.set_ylim(0, max(max(f16_vals), max(i2s_vals)) * 1.25)
        if t_idx == 0:
            ax_pp.set_ylabel('Throughput (tokens/s)', fontsize=9)
            ax_pp.legend(fontsize=8)

        # Row 1: Decode (tg)
        ax_tg = axes[1, t_idx]
        f16_vals = [f16_data[task][t_idx] for task in tg_tasks]
        i2s_vals = [i2s_data[task][t_idx] for task in tg_tasks]

        x = np.arange(len(tg_tasks))
        bars1 = ax_tg.bar(x - width/2, f16_vals, width, label='F16', color='#4472C4', edgecolor='black', linewidth=0.5)
        bars2 = ax_tg.bar(x + width/2, i2s_vals, width, label='I2_S', color='#ED7D31', edgecolor='black', linewidth=0.5)

        for bar in bars1:
            h = bar.get_height()
            ax_tg.text(bar.get_x() + bar.get_width()/2, h + 0.3, f'{h:.1f}', ha='center', va='bottom', fontsize=7)
        for bar, f16_v, i2s_v in zip(bars2, f16_vals, i2s_vals):
            h = bar.get_height()
            speedup = i2s_v / f16_v
            ax_tg.text(bar.get_x() + bar.get_width()/2, h + 0.3, f'{h:.1f}\n({speedup:.1f}x)', ha='center', va='bottom', fontsize=7, color='red')

        ax_tg.set_title(f'Thread={t} (Decode)', fontsize=10)
        ax_tg.set_xticks(x)
        ax_tg.set_xticklabels(tg_tasks, fontsize=8)
        ax_tg.grid(True, alpha=0.3, axis='y')
        ax_tg.set_ylim(0, max(max(f16_vals), max(i2s_vals)) * 1.25)
        if t_idx == 0:
            ax_tg.set_ylabel('Throughput (tokens/s)', fontsize=9)
            ax_tg.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")

plot_model_throughput(yoco_moe_f16, yoco_moe_i2s, 'YOCO-MoE-30B.A3B',
                      '/home/huangxin/code_list/bitnet-moe-script/throughput_bar_yoco_moe.png')
plot_model_throughput(yoco_u_moe_f16, yoco_u_moe_i2s, 'YOCO-U-MoE-30B.A6B',
                      '/home/huangxin/code_list/bitnet-moe-script/throughput_bar_yoco_u_moe.png')
