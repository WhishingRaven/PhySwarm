import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform
import scipy.sparse.csgraph as csgraph

plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

METHODS = {
    'PhySwarm':  {'pattern': 'data/radar_*.csv',       'color': '#0047AB'}, # 宝蓝色
    'FSM':   {'pattern': 'data/machine_*.csv',  'color': '#FF7F0E'}, # 活力橙
    'Ablation A':       {'pattern': 'data/static_1_*.csv', 'color': '#2CA02C'}, # 翡翠绿
    'Ablation B':       {'pattern': 'data/static_2_*.csv', 'color': '#D62728'}  # 玫瑰红
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
        
        # 1. 成功率 S_m 计算
        success = 1.0 if ('chain_connected' in df.columns and df['chain_connected'].max() > 0) else 0.0
        
        # 2. TTC 计算 
        if success > 0:
            nav_time = df[df['chain_connected'] == 1]['step'].min()
        else:
            nav_time = MAX_STEPS
            
        # 3. 归一化碰撞率
        N_agents = df['agent_id'].nunique()
        T_steps = MAX_STEPS
        
        # 提取智能体到目标的距离
        df['dist_to_target'] = np.sqrt(
            (df['pos_x'] - df['target_0_x'])**2 + 
            (df['pos_y'] - df['target_0_y'])**2
        )

        TARGET_ZONE_RADIUS = 0.2
        
        df['unsafe_collision'] = df['is_colliding']
        df.loc[df['dist_to_target'] < TARGET_ZONE_RADIUS, 'unsafe_collision'] = 0
        
        c_total = df['unsafe_collision'].sum() if 'unsafe_collision' in df.columns else 0
        c_rate = c_total / (N_agents * T_steps) if (N_agents * T_steps) > 0 else 0.0
        
        # 4. 控制平滑度 J_smooth 计算 
        # 严格使用左、右轮目标指令速度进行 L1 范数阶跃计算
        df_sorted = df.sort_values(by=['agent_id', 'step'])
        
        df_sorted['diff_l'] = df_sorted.groupby('agent_id')['left_speed'].diff().abs()
        df_sorted['diff_r'] = df_sorted.groupby('agent_id')['right_speed'].diff().abs()
        
        # L1-norm: ||c_i(t) - c_i(t-1)||_1
        df_sorted['c_diff_l1'] = df_sorted['diff_l'] + df_sorted['diff_r']
        
        j_smooth = df_sorted['c_diff_l1'].mean() 
                
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
            'C_rate': c_rate,         
            'J_smooth': j_smooth      
        })
            
    return pd.DataFrame(metrics_list), time_series_fiedler

print("Processing Rescue Datasets with Continuous Topological Network Theory...")
results = {name: process_rescue_data(info['pattern']) for name, info in METHODS.items() if process_rescue_data(info['pattern']) is not None}

if not results:
    print("❌ 未找到数据文件，请检查 pattern 路径！")
    exit()

radar_labels = [
    r'$\eta_{\mathrm{succ}}$',    # 右上 (Success Rate)
    r'$J_{\mathrm{smooth}}$',     # 右下 (Control Smoothness)
    r'$R_{\mathrm{coll}}$',      # 左下 (Collision Safety)
    r'$T_{\mathrm{TTC}}$'         # 左上 (Time to Conn.)
]
n_metrics = len(radar_labels)

# 计算各个方法的均值
raw_df = pd.DataFrame({name: res[0].mean() for name, res in results.items()}).T

norm_data = {}

# 全局极值获取，用于归一化
global_max_crate = max(raw_df['C_rate'].max(), 1e-4)
global_min_crate = raw_df['C_rate'].min()
global_max_jsmooth = max(raw_df['J_smooth'].max(), 1e-4)
global_min_jsmooth = raw_df['J_smooth'].min()
global_max_ttc = raw_df['TTC'].max()
global_min_ttc = raw_df['TTC'].min()

for name in raw_df.index:
    # (1) Success Rate
    sr = raw_df.loc[name, 'Success'] 
    sr = 0.2 + 0.8 * sr 
    
    # (2) TTC
    ttc_val = raw_df.loc[name, 'TTC']
    eff = (global_max_ttc * 1.05 - ttc_val) / (global_max_ttc * 1.05 - global_min_ttc + 1e-6)
    eff = 0.2 + 0.8 * eff
    
    # (3) Collision Safety
    crate_val = raw_df.loc[name, 'C_rate']
    saf = (global_max_crate * 1.1 - crate_val) / (global_max_crate * 1.1 - global_min_crate + 1e-6)
    saf = 0.2 + 0.8 * saf
    
    # (4) Control Smoothness
    jsmooth_val = raw_df.loc[name, 'J_smooth']
    smo = (global_max_jsmooth * 1.1 - jsmooth_val) / (global_max_jsmooth * 1.1 - global_min_jsmooth + 1e-6)
    smo = 0.2 + 0.8 * smo

    norm_data[name] = [sr, smo, saf, eff]

angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist() + [0]

# 画布大小与位置控制
fig1, ax1 = plt.subplots(figsize=(3.8, 3.8), subplot_kw=dict(polar=True), dpi=300)
ax1.set_position([0.15, 0.22, 0.70, 0.70]) 

ax1.set_theta_offset(np.pi / 4) # 旋转 45 度
ax1.set_theta_direction(-1)     # 顺时针

# 刻度设置
ax1.set_xticks(angles[:-1])
ticks = ax1.set_xticklabels(radar_labels, size=11.5, fontweight='bold')
ax1.tick_params(axis='x', pad=10)

# 对齐方式调整
ticks[0].set_horizontalalignment('left')
ticks[0].set_verticalalignment('bottom')
ticks[1].set_horizontalalignment('left')
ticks[1].set_verticalalignment('top')
ticks[2].set_horizontalalignment('right')
ticks[2].set_verticalalignment('top')
ticks[3].set_horizontalalignment('right')
ticks[3].set_verticalalignment('bottom')

ax1.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8.0)
plt.ylim(0, 1.0)

ax1.grid(True, color="#E3E3E3", linestyle="--", linewidth=0.8)

# 绘图
for name in norm_data:
    values = norm_data[name] + [norm_data[name][0]]
    color = METHODS[name]['color']
    ax1.plot(angles, values, color=color, linewidth=2.8, label=name, zorder=3)
    ax1.fill(angles, values, color=color, alpha=0.10)

ax1.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.10), 
    ncol=4, 
    fontsize=9.0, 
    frameon=True, 
    edgecolor='#E0E0E0', 
    facecolor='#FAFAFA'
)

os.makedirs("Paper_Figures", exist_ok=True)
plt.savefig("Paper_Figures/Radar_Rescue_Topological_Comparison.pdf", bbox_inches='tight')
plt.show()
