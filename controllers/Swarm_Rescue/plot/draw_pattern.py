import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  
import glob
import os
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform
import scipy.sparse.csgraph as csgraph


WIDTH_CM = 12.0     # 目标宽度 (厘米)
HEIGHT_CM = 8.0    # 目标高度 (厘米)
FIG_WIDTH = WIDTH_CM / 2.54
FIG_HEIGHT = HEIGHT_CM / 2.54

plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

METHODS = {
    'PhySwarm': {'pattern': 'data/rl_*.csv',       'color': '#0047AB'}, # 宝蓝色
    'FSM':   {'pattern': 'data/machine_*.csv',  'color': '#FF7F0E'}, # 活力橙
    'Ablation A':{'pattern': 'data/static_1_*.csv', 'color': '#2CA02C'}, # 翡翠绿
    'Ablation B':  {'pattern': 'data/static_2_*.csv', 'color': '#D62728'}  # 玫瑰红
}

MAX_STEPS = 600 
COMM_VIEW = 0.6  # 硬阈值 
IDEAL_DIST = 0.4 # 理想通信距离 

def calc_continuous_fiedler(agents_pos, base_pos, target_pos, comm_range=0.6, ideal_dist=0.4):
    if len(agents_pos) == 0:
        return 0.0
        
    all_pos = np.vstack([agents_pos, base_pos, target_pos])
    N_total = len(all_pos)
    base_idx = N_total - 2
    target_idx = N_total - 1
    
    # 提取有效子图
    dist_matrix = squareform(pdist(all_pos))
    adj_binary = (dist_matrix <= comm_range).astype(int)
    n_comp, labels = csgraph.connected_components(adj_binary, directed=False)
    
    if labels[base_idx] != labels[target_idx]:
        return 0.0
        
    valid_mask = (labels == labels[base_idx])
    sub_dist = dist_matrix[valid_mask][:, valid_mask]
    
    W = np.exp(- (sub_dist / ideal_dist)**4)
    np.fill_diagonal(W, 0.0)
    
    # 计算拉普拉斯矩阵
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    
    eigenvals = np.linalg.eigvalsh(L)
    eigenvals = np.sort(eigenvals)
    
    if len(eigenvals) >= 2:
        return max(eigenvals[1], 0.0)
    else:
        return 0.0

def process_rescue_data(file_pattern):
    files = glob.glob(file_pattern)
    if not files: return None
    
    metrics_list = []
    time_series_fiedler = np.zeros((len(files), MAX_STEPS))
    
    for trial_idx, f in enumerate(files):
        df = pd.read_csv(f)
        steps = sorted(df['step'].unique())
        
        success = 1.0 if ('chain_connected' in df.columns and df['chain_connected'].max() > 0) else 0.0
        
        if success > 0:
            nav_time = df[df['chain_connected'] == 1]['step'].min()
        else:
            nav_time = MAX_STEPS
            
        collisions = df['is_colliding'].sum() if 'is_colliding' in df.columns else 0
        
        df_sorted = df.sort_values(by=['agent_id', 'step'])
        df_sorted['ang_vel'] = df_sorted.groupby('agent_id')['angle'].diff()
        df_sorted.loc[df_sorted['ang_vel'] > np.pi, 'ang_vel'] -= 2 * np.pi
        df_sorted.loc[df_sorted['ang_vel'] < -np.pi, 'ang_vel'] += 2 * np.pi
        jerk = df_sorted.groupby('agent_id')['ang_vel'].diff().abs().mean()
                
        # 计算时序 Fiedler Value
        for s in range(MAX_STEPS):
            if s not in steps:
                time_series_fiedler[trial_idx, s] = time_series_fiedler[trial_idx, s-1] if s > 0 else 0.0
                continue
            
            step_df = df[df['step'] == s]
            base_pos = step_df[['base_x', 'base_y']].iloc[0].values
            target_pos = step_df[['target_0_x', 'target_0_y']].iloc[0].values
            agents_pos = step_df[['pos_x', 'pos_y']].values
            
            fiedler_val = calc_continuous_fiedler(agents_pos, base_pos, target_pos, COMM_VIEW, IDEAL_DIST)
            time_series_fiedler[trial_idx, s] = fiedler_val

        metrics_list.append({
            'Success': success, 
            'TTC': nav_time, 
            'Collisions': collisions, 
            'Jerk': jerk
        })
            
    return pd.DataFrame(metrics_list), time_series_fiedler

print("Processing Rescue Datasets with Continuous Topological Network Theory...")
results = {name: process_rescue_data(info['pattern']) for name, info in METHODS.items() if process_rescue_data(info['pattern']) is not None}

if not results:
    print("❌ 未找到数据文件，请检查 pattern 路径！")
    exit()

fig2, ax2 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
steps_arr = np.arange(MAX_STEPS)

max_fiedler_display = 0.0

for name in results:
    color = METHODS[name]['color']
    ts_fiedler = results[name][1]
    N_trials = ts_fiedler.shape[0]
    
    # 先对每个独立的 trial 进行平滑，消除个体内震荡
    ts_smoothed = gaussian_filter1d(ts_fiedler, sigma=15, axis=1)
    
    # 跨 trial 聚合
    mean_fiedler = np.mean(ts_smoothed, axis=0)
    std_fiedler = np.std(ts_smoothed, axis=0)
    
    # 使用标准误 (SEM) 替代标准差 (STD)
    ci = 0.5 * (std_fiedler / np.sqrt(max(N_trials, 1))) 
    
    mean_fiedler = gaussian_filter1d(mean_fiedler, sigma=5)
    ci = gaussian_filter1d(ci, sigma=5)
    
    max_fiedler_display = max(max_fiedler_display, np.max(mean_fiedler))
    
    ax2.plot(steps_arr, mean_fiedler, color=color, linewidth=2.5, label=name, zorder=3)
    ax2.fill_between(steps_arr, np.clip(mean_fiedler - ci, 0, None), mean_fiedler + ci, 
                     color=color, alpha=0.15, edgecolor='none', zorder=2)

ax2.set_xlabel('Simulation Steps', fontsize=10, fontweight='bold')
ax2.set_ylabel(r"Algebraic Connectivity", fontsize=10, fontweight='bold')
ax2.set_xlim(0, MAX_STEPS)

if max_fiedler_display == 0: max_fiedler_display = 1.0 

ax2.set_ylim(0.0, 0.03)
ax2.set_yticks([0.00, 0.01, 0.02, 0.03])

ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

ax2.tick_params(axis='both', which='major', labelsize=7)
ax2.legend(loc='upper right', fontsize=8, frameon=True, edgecolor='gray')

plt.tight_layout()
plt.savefig("Paper_Figures/Fig_Rescue_Fiedler_Evolution_Smooth.pdf", bbox_inches='tight')
# plt.show()

print("✅ Radar Chart & Smooth Algebraic Connectivity Line Plot saved successfully.")