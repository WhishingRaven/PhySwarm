import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


WIDTH_CM = 12.0     # 目标宽度 (厘米)
HEIGHT_CM = 8.0    # 目标高度 (厘米)
FIG_WIDTH = WIDTH_CM / 2.54
FIG_HEIGHT = HEIGHT_CM / 2.54

# ==========================================
# 🌟 全局科研绘图规范设置
# ==========================================
plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'

# --- 1. 配置参数 ---
FAIL_COUNTS = [0, 1, 2, 3, 4]
TRIALS_PER_COUNT = 60  
FAILURE_STEP = 300
TOTAL_STEPS = 1000
COLORS = sns.color_palette("viridis", len(FAIL_COUNTS))

resilience_data = []  
degradation_list = []  

print("Processing Macroscopic Throughput metrics for Resilience...")

for fc in FAIL_COUNTS:
    all_trials = []
    
    for idx in range(TRIALS_PER_COUNT):
        file_path = f"data/fails_{fc}_{idx}.csv" 
        if not os.path.exists(file_path): continue
        
        df = pd.read_csv(file_path)
        
        delivery_per_step = df.groupby('step')['delivery_event'].sum()
        
        cum_delivery = delivery_per_step.reindex(range(TOTAL_STEPS)).fillna(0).cumsum()
        
        step_stats = pd.DataFrame({
            'step': np.arange(TOTAL_STEPS),
            'throughput': cum_delivery.values
        })
        
        all_trials.append(step_stats)
  
        final_throughput = cum_delivery.iloc[-1]
        degradation_list.append({'FailCount': fc, 'FinalThroughput': final_throughput})

        df['step'] = pd.to_numeric(df['step'], errors='coerce')
        df['delivery_event'] = pd.to_numeric(
            df['delivery_event'],
            errors='coerce',
        ).fillna(0)

        delivery_per_step = df.groupby('step')['delivery_event'].sum()

        raw_total = df['delivery_event'].sum()

        current_window_total = (
            delivery_per_step
            .reindex(range(TOTAL_STEPS))
            .fillna(0)
            .sum()
        )

        delivery_rows = df.loc[
            df['delivery_event'] > 0,
            'step',
        ]

        last_delivery_step = (
            delivery_rows.max()
            if not delivery_rows.empty
            else None
        )

        print(
            f"fail={fc}, trial={idx}, "
            f"step_range=[{df['step'].min()}, {df['step'].max()}], "
            f"raw_total={raw_total}, "
            f"in_current_window={current_window_total}, "
            f"last_delivery_step={last_delivery_step}"
        )
        
    if len(all_trials) == 0: continue
    
    combined = pd.concat(all_trials)
    stats_df = combined.groupby('step')['throughput'].agg(['mean', 'std']).reset_index()
    stats_df['ci'] = 1.96 * (stats_df['std'] / np.sqrt(len(all_trials)))
    
    resilience_data.append(stats_df)

df_box = pd.DataFrame(degradation_list)


def plot_temporal_throughput():
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)

    for i, fc in enumerate(FAIL_COUNTS):
        if i >= len(resilience_data): break
        df_p = resilience_data[i]
        
        ax.plot(df_p['step'], df_p['mean'], color=COLORS[i], lw=2.5, label=f'$N_{{fail}}={fc}$')
        ax.fill_between(df_p['step'], df_p['mean'] - df_p['ci'], df_p['mean'] + df_p['ci'], 
                         color=COLORS[i], alpha=0.15, edgecolor='none')

    ax.axvline(FAILURE_STEP, color='red', linestyle=':', lw=2)
    
    y_max = resilience_data[0]['mean'].max()
    ax.annotate('Failure Injection', xy=(FAILURE_STEP, y_max * 0.4), xytext=(FAILURE_STEP + 100, y_max * 0.3),
                 arrowprops=dict(arrowstyle="->", color='red', connectionstyle="arc3,rad=.2"), 
                 fontsize=8, color='red', fontweight='bold')

    ax.set_xlabel('Simulation Steps', fontsize=10, fontweight='bold')
    ax.set_ylabel("Cumulative Throughput", fontsize=10, fontweight='bold')
    
    ax.legend(loc='upper left', ncol=1, frameon=True, facecolor='white', framealpha=0.9, fontsize=8)

    ax.set_ylim(0, y_max * 1.1) 
    ax.set_xlim(0, TOTAL_STEPS) 
    ax.tick_params(axis='both', labelsize=8)

    plt.tight_layout()
    os.makedirs("Paper_Figures", exist_ok=True)
    plt.savefig("Paper_Figures/Fig_Resilience_Throughput_Temporal.pdf", format='pdf', bbox_inches='tight')
    # plt.show()

plot_temporal_throughput()