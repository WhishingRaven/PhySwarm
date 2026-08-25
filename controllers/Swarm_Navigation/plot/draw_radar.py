import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.ndimage import gaussian_filter1d

plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

METHODS = {
    'PhySwarm': {'pattern': 'data/radar_*.csv',       'color': '#0047AB'}, # 宝蓝色
    'FSM':    {'pattern': 'data/machine_*.csv', 'color': '#FF7F0E'}, # 活力橙
    'Ablation A':        {'pattern': 'data/static1_*.csv', 'color': '#2CA02C'}, # 翡翠绿
    'Ablation B':        {'pattern': 'data/static2_*.csv', 'color': '#D62728'}  # 玫瑰红
}

MAX_STEPS = 600
GOAL_POS = np.array([1.15, 0.0])
GOAL_THRESHOLD = 0.15
R0 = 0.2

def calc_paper_formation_error(actual_pos, target_pos):
    """
    计算消除平移和旋转的严格编队误差 (Cai & Shen, 2019)
    返回: 绝对形态误差距离
    """
    N = actual_pos.shape[0]
    p_mat = actual_pos - np.mean(actual_pos, axis=0)
    q_mat = target_pos - np.mean(target_pos, axis=0)
    
    p = p_mat.flatten()[:, np.newaxis]
    q = q_mat.flatten()[:, np.newaxis]
    
    R_2x2 = np.array([[0, -1], [1, 0]])
    R = np.kron(np.eye(N), R_2x2) 
    
    norm_q_sq = np.sum(q**2)
    norm_q = np.sqrt(norm_q_sq) 
    
    if norm_q_sq < 1e-8: 
        return np.sqrt(np.sum(p**2))
        
    q_perp = R @ q
    Q = (1.0 / norm_q_sq) * (q @ q.T + q_perp @ q_perp.T)
    
    norm_p_sq = np.sum(p**2)
    p_Q_p = float(p.T @ Q @ p)
    
    error_sq = norm_p_sq + norm_q_sq - 2.0 * np.sqrt(norm_q_sq) * np.sqrt(np.maximum(p_Q_p, 0.0))
    error_distance = np.sqrt(np.maximum(error_sq, 0.0))
    
    return error_distance

def generate_circle_template(n_agents=8):
    angles = np.linspace(0, 2*np.pi, n_agents, endpoint=False)
    xs = R0 * np.cos(angles)
    ys = R0 * np.sin(angles)
    return np.column_stack([xs, ys])

def get_circular_formation_error(curr_pos, tgt_shape):

    centroid_curr = np.mean(curr_pos, axis=0)
    angles_curr = np.arctan2(curr_pos[:, 1] - centroid_curr[1], curr_pos[:, 0] - centroid_curr[0])
    sort_idx_curr = np.argsort(angles_curr)

    centroid_tgt = np.mean(tgt_shape, axis=0)
    angles_tgt = np.arctan2(tgt_shape[:, 1] - centroid_tgt[1], tgt_shape[:, 0] - centroid_tgt[0])
    sort_idx_tgt = np.argsort(angles_tgt)

    curr_sorted = curr_pos[sort_idx_curr]
    tgt_sorted = tgt_shape[sort_idx_tgt]
    
    return calc_paper_formation_error(curr_sorted, tgt_sorted)

def calc_error_history(trajectory_df):
    steps = sorted(trajectory_df['step'].unique())
    n_agents = len(trajectory_df[trajectory_df['step'] == steps[0]])
    circle_template = generate_circle_template(n_agents)
    
    error_history = {}
    
    for s in steps:
        curr_pos = trajectory_df[trajectory_df['step'] == s][['pos_x', 'pos_y']].values
        err_dist = get_circular_formation_error(curr_pos, circle_template)
        error_history[s] = err_dist
            
    return error_history

def process_method_data(file_pattern):
    files = glob.glob(file_pattern)
    if not files: return None
    
    metrics_list = []
    time_series_data = np.zeros((len(files), MAX_STEPS)) 
    
    for trial_idx, f in enumerate(files):
        df = pd.read_csv(f)
        steps = sorted(df['step'].unique())
        
        error_history_dict = calc_error_history(df)
        
        nav_time = MAX_STEPS
        done = False

        N_agents = df['agent_id'].nunique()
        T_steps = MAX_STEPS
        
        # 仅统计 MAX_STEPS 范围内的数据
        valid_df = df[df['step'] < MAX_STEPS].copy()
        valid_df = valid_df.sort_values(['agent_id', 'step'])

        N_agents = valid_df['agent_id'].nunique()

        if 'is_colliding' in valid_df.columns:
            collision_flag = valid_df['is_colliding'].fillna(0).astype(int)

            # 1. 碰撞状态累计次数：同一次持续碰撞会按多步累计
            c_total = int(collision_flag.sum())

            # 2. 独立碰撞开始次数：统计每个机器人 0 -> 1 的上升沿
            prev_collision = valid_df.groupby('agent_id')['is_colliding'].shift(1).fillna(0).astype(int)
            collision_events = int(((collision_flag == 1) & (prev_collision == 0)).sum())

            # 3. 碰撞占用率
            collision_rate = float(collision_flag.mean())
        else:
            c_total = 0
            collision_events = 0
            collision_rate = 0.0

        collision_rate = c_total / (N_agents * T_steps) if (N_agents * T_steps) > 0 else 0.0
     
        df_sorted = df.sort_values(by=['agent_id', 'step'])
        df_sorted['diff_l'] = df_sorted.groupby('agent_id')['cmd_omega_l'].diff().abs()
        df_sorted['diff_r'] = df_sorted.groupby('agent_id')['cmd_omega_r'].diff().abs()
        df_sorted['c_diff_l1'] = df_sorted['diff_l'] + df_sorted['diff_r']
        j_smooth = df_sorted['c_diff_l1'].mean() # 自动忽略首步 NaN，分母精准对齐 N*(T-1)

        last_valid_s = 0
        FORM_RMSE_THRESHOLD = 0.05
        REQUIRED_STABLE_STEPS = 10
        stable_count = 0

        for s in steps:
            if s < MAX_STEPS:
                time_series_data[trial_idx, s] = error_history_dict[s]
                last_valid_s = s

            curr_pos = df[df['step'] == s][['pos_x', 'pos_y']].values
            centroid = np.mean(curr_pos, axis=0)

            goal_distance = np.linalg.norm(centroid - GOAL_POS)
            formation_rmse = error_history_dict[s] / np.sqrt(len(curr_pos))

            arrival_valid = (
                goal_distance < GOAL_THRESHOLD
                and formation_rmse < FORM_RMSE_THRESHOLD
            )

            stable_count = stable_count + 1 if arrival_valid else 0

            if stable_count >= REQUIRED_STABLE_STEPS and not done:
                done = True
                nav_time = s - REQUIRED_STABLE_STEPS + 1
                
        # 处理防越界：将最后一步的误差向后顺延
        if last_valid_s < MAX_STEPS - 1:
            time_series_data[trial_idx, last_valid_s+1:] = time_series_data[trial_idx, last_valid_s]
            
        mean_trial_error = np.mean(list(error_history_dict.values()))
            
        metrics_list.append({
            'Nav_Time': nav_time, 
            'Mean_Error': mean_trial_error,   
            'Collision_Rate': collision_rate, 
            'J_smooth': j_smooth    
        })
            
    return pd.DataFrame(metrics_list), time_series_data

print("Loading and processing datasets...")
results = {name: process_method_data(info['pattern']) for name, info in METHODS.items() if process_method_data(info['pattern']) is not None}

radar_labels = [
    r'$T_{\mathrm{nav}}$',      # 左上：导航效率 (Navigational Efficiency / Nav_Time)
    r'$e_{\mathrm{form}}$',     # 右上：圆阵形态维持误差 (Circular Form. Maintenance)
    r'$J_{\mathrm{smooth}}$',    # 右下：控制平滑度 (Control Smoothness)
    r'$R_{\mathrm{coll}}$'       # 左下：碰撞安全 (Collision Safety)
]
n_metrics = len(radar_labels)
raw_df = pd.DataFrame({name: res[0].mean() for name, res in results.items()}).T

BASE_SCORE = 0.25

def soft_normalize(series):
    v_min, v_max = series.min(), series.max()
    if v_max == v_min:
        return pd.Series(1.0, index=series.index)
    return BASE_SCORE + (1.0 - BASE_SCORE) * (v_max - series) / (v_max - v_min)

norm_data = {}
eff_scores = soft_normalize(raw_df['Nav_Time'])
fid_scores = soft_normalize(raw_df['Mean_Error'])
smo_scores = soft_normalize(raw_df['J_smooth'])       
saf_scores = soft_normalize(raw_df['Collision_Rate']) 

for name in raw_df.index:
    norm_data[name] = [
        eff_scores[name], # 左上
        fid_scores[name], # 右上
        smo_scores[name], # 右下
        saf_scores[name]  # 左下
    ]

angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist() + [0]

fig2, ax2 = plt.subplots(figsize=(3.8, 3.8), subplot_kw=dict(polar=True), dpi=300)

ax2.set_position([0.15, 0.22, 0.70, 0.70]) 

ax2.set_theta_offset(np.pi / 2 + np.pi/4) 
ax2.set_theta_direction(-1)           

ax2.grid(color='#E3E3E3', linestyle='--', alpha=0.8, linewidth=0.8)

ax2.set_xticks(angles[:-1])
ticks = ax2.set_xticklabels(radar_labels, size=12, fontweight='bold')

ax2.tick_params(axis='x', pad=10)

# Index 0: 左上 (T_nav)
ticks[0].set_horizontalalignment('right')
ticks[0].set_verticalalignment('bottom')

# Index 1: 右上 (e_form)
ticks[1].set_horizontalalignment('left')
ticks[1].set_verticalalignment('bottom')

# Index 2: 右下 (J_smooth)
ticks[2].set_horizontalalignment('left')
ticks[2].set_verticalalignment('top')

# Index 3: 左下 (C_rate)
ticks[3].set_horizontalalignment('right')
ticks[3].set_verticalalignment('top')

ax2.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="gray", size=8.0, alpha=0.7)
plt.ylim(0, 1.0)

for name in norm_data:
    values = norm_data[name] + [norm_data[name][0]]
    color = METHODS[name]['color']
    ax2.plot(angles, values, color=color, linewidth=2.8, label=name, zorder=3)
    ax2.fill(angles, values, color=color, alpha=0.10)

ax2.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.12), 
    ncol=4, 
    fontsize=9.0, 
    frameon=True, 
    edgecolor='#E0E0E0', 
    facecolor='#FAFAFA'
)

plt.savefig("Paper_Figures/Fig2_Four_Dimension_Radar_Beautified.pdf", format='pdf', bbox_inches='tight')
# plt.show()

print("✅ Figure 2 (4D Formation Radar Chart with Mathematics Symbols) Generated and Saved.")
