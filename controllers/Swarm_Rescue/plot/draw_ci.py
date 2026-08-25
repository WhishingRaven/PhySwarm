import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
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

FAIL_COUNTS = [0,1,2,3,4]
TRIALS_PER_COUNT = 30  
FAILURE_STEP = 300
TOTAL_STEPS = 600

COMM_VIEW = 0.8       # 硬阈值：超过此距离直接断开
IDEAL_COMM_DIST = 0.6 # 软阈值：连续链路质量的理想衰减基准

COLORS = sns.color_palette("viridis", len(FAIL_COUNTS))

print("Processing Rescue Network Resilience metrics (Fiedler Value)...")

def calc_continuous_fiedler(agents_pos, base_pos, target_pos, comm_range, ideal_dist):
    if len(agents_pos) == 0:
        return 0.0
        
    all_pos = np.vstack([agents_pos, base_pos, target_pos])
    N_total = len(all_pos)
    base_idx = N_total - 2
    target_idx = N_total - 1
    
    # 构建二值拓扑图判断是否连通
    dist_matrix = squareform(pdist(all_pos))
    adj_binary = (dist_matrix <= comm_range).astype(int)
    n_comp, labels = csgraph.connected_components(adj_binary, directed=False)
    
    if labels[base_idx] != labels[target_idx]:
        return 0.0 
        
    # 提取连通子图
    valid_mask = (labels == labels[base_idx])
    sub_dist = dist_matrix[valid_mask][:, valid_mask]
    
    # 连续链路质量衰减模型
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

def calculate_step_fiedler(group):
    bx, by = group.iloc[0]['base_x'], group.iloc[0]['base_y']
    tx, ty = group.iloc[0]['target_0_x'], group.iloc[0]['target_0_y']
    
    base_pos = np.array([bx, by])
    target_pos = np.array([tx, ty])
    
    active_agents = group[group['is_alive'] == 1]
    
    if len(active_agents) == 0:
        return 0.0
        
    agents_pos = active_agents[['pos_x', 'pos_y']].values
    
    fiedler_val = calc_continuous_fiedler(agents_pos, base_pos, target_pos, COMM_VIEW, IDEAL_COMM_DIST)
    
    return fiedler_val


resilience_data = []  
degradation_list = []  

for fc in FAIL_COUNTS:
    all_trials = []
    
    for idx in range(TRIALS_PER_COUNT):
        file_path = f"data/fails_{fc}_{idx}.csv" 
        if not os.path.exists(file_path): continue

        df = pd.read_csv(file_path)
        
        # 1. 按照 step 计算 Fiedler Value
        step_stats = df.groupby('step').apply(calculate_step_fiedler).reset_index(name='fiedler_value')
        all_trials.append(step_stats[['step', 'fiedler_value']])
        
        # 2. 提取稳态残余连通度
        steady_val = step_stats[step_stats['step'] >= FAILURE_STEP + 200]['fiedler_value'].mean()
        degradation_list.append({'FailCount': fc, 'MeanPostFailureFiedler': steady_val})
        
    if len(all_trials) == 0: continue
    
    # 汇总并计算均值和置信区间
    combined = pd.concat(all_trials)
    stats_df = combined.groupby('step')['fiedler_value'].agg(['mean', 'std']).reset_index()
    stats_df['ci'] = 1.96 * (stats_df['std'] / np.sqrt(len(all_trials)))
    
    stats_df['mean'] = stats_df['mean'].ewm(alpha=0.3, adjust=False).mean()
    stats_df['ci'] = stats_df['ci'].ewm(alpha=0.1, adjust=False).mean() 
    
    resilience_data.append(stats_df)

df_box = pd.DataFrame(degradation_list)


def plot_temporal_recovery():
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)

    for i, fc in enumerate(FAIL_COUNTS):
        if i >= len(resilience_data): break
        df_p = resilience_data[i]
        
        ax.plot(df_p['step'], df_p['mean'], color=COLORS[i], lw=3.0, label=f'$N_{{fail}}={fc}$')
        # 置信区间适当减弱，以突出主线趋势
        ax.fill_between(df_p['step'], np.clip(df_p['mean'] - 0.5*df_p['ci'], 0, None), df_p['mean'] + 0.5*df_p['ci'], 
                         color=COLORS[i], alpha=0.15, edgecolor='none')

    ax.axvline(FAILURE_STEP, color='red', linestyle=':', lw=2)
    ax.annotate('Failure Injection', xy=(FAILURE_STEP, ax.get_ylim()[1]*0.5), 
                 xytext=(FAILURE_STEP-280, ax.get_ylim()[1]*0.7),
                 arrowprops=dict(arrowstyle="-|>", color='red', connectionstyle="arc3,rad=.2",), 
                 fontsize=8, color='red', fontweight='bold')
    

    ax.set_xlabel('Simulation Steps', fontsize=10, fontweight='bold')
    ax.set_ylabel(r"Algebraic Connectivity", fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', ncol=1, frameon=True, facecolor='white', framealpha=0.9, fontsize=8)

    ax.set_xlim(0, TOTAL_STEPS) 
    # 加入连通临界线
    ax.axhline(0.01, color='black', linestyle=':', lw=1.5, alpha=0.5)

    plt.tight_layout()
    os.makedirs("Paper_Figures", exist_ok=True)
    plt.savefig("Paper_Figures/Fig_Rescue_Connectivity_Recovery.pdf", format='pdf', bbox_inches='tight')
    # plt.show()


plot_temporal_recovery()