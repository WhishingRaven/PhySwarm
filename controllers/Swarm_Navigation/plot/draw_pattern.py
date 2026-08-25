import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.ndimage import gaussian_filter1d

WIDTH_CM = 12.0     # 目标宽度 (厘米)
HEIGHT_CM = 8.0    # 目标高度 (厘米)
FIG_WIDTH = WIDTH_CM / 2.54
FIG_HEIGHT = HEIGHT_CM / 2.54

plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

METHODS = {
    'PhySwarm': {'pattern': 'data/fails_0_*.csv',       'color': '#0047AB'}, # 宝蓝色
    'FSM':    {'pattern': 'data/machine_*.csv', 'color': '#FF7F0E'}, # 活力橙
    'Ablation A':        {'pattern': 'data/static1_*.csv', 'color': '#2CA02C'}, # 翡翠绿
    'Ablation B':        {'pattern': 'data/static2_*.csv', 'color': '#D62728'}  # 玫瑰红
}

MAX_STEPS = 400
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
        
        collisions = df['is_colliding'].sum() if 'is_colliding' in df.columns else 0
        df_sorted = df.sort_values(by=['agent_id', 'step'])
        jerk = df_sorted.groupby('agent_id')['cmd_omega_l'].diff().abs().sum() + \
               df_sorted.groupby('agent_id')['cmd_omega_r'].diff().abs().sum()

        last_valid_s = 0
        for s in steps:
            if s < MAX_STEPS:
                time_series_data[trial_idx, s] = error_history_dict[s]
                last_valid_s = s
            
            curr_pos = df[df['step'] == s][['pos_x', 'pos_y']].values
            centroid = np.mean(curr_pos, axis=0)
            if np.linalg.norm(centroid - GOAL_POS) < GOAL_THRESHOLD and not done:
                done = True
                nav_time = s
                
        if last_valid_s < MAX_STEPS - 1:
            time_series_data[trial_idx, last_valid_s+1:] = time_series_data[trial_idx, last_valid_s]
            
        mean_trial_error = np.mean(list(error_history_dict.values()))
            
        metrics_list.append({
            'Nav_Time': nav_time, 
            'Mean_Error': mean_trial_error, 
            'Collisions': collisions, 
            'Jerk': jerk
        })
            
    return pd.DataFrame(metrics_list), time_series_data

print("Loading and processing datasets...")
results = {name: process_method_data(info['pattern']) for name, info in METHODS.items() if process_method_data(info['pattern']) is not None}

fig1, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
steps_arr = np.arange(MAX_STEPS)

for name in results:
    color = METHODS[name]['color']
    ts_error = results[name][1] 
    
    mean_err = gaussian_filter1d(ts_error.mean(axis=0), sigma=2)
    std_err = gaussian_filter1d(ts_error.std(axis=0), sigma=2)
    ci = 1.96 * (std_err / np.sqrt(max(1, ts_error.shape[0])))
    
    ax1.plot(steps_arr, mean_err, color=color, linewidth=2.5, label=name, zorder=3)
    ax1.fill_between(steps_arr, np.clip(mean_err - ci, 0, None), mean_err + ci, 
                     color=color, alpha=0.15, edgecolor='none', zorder=2)

ax1.set_xlabel('Simulation Steps', fontsize=10, fontweight='bold')
ax1.set_ylabel("Formation error (m)", fontsize=10, fontweight='bold')
ax1.set_xlim(0, MAX_STEPS)
ax1.set_ylim(bottom=0)
ax1.tick_params(axis='both', which='major', labelsize=8)

ax1.axhline(0, color='black', linestyle='--', alpha=0.3, zorder=1)

ax1.legend(loc='upper right', fontsize=8, frameon=True, edgecolor='gray')

os.makedirs("Paper_Figures", exist_ok=True)
plt.tight_layout()
plt.savefig("Paper_Figures/Fig1_Circular_Formation_Error_TS.pdf", format='pdf', bbox_inches='tight')