import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import gaussian_kde

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

# --- 配置参数与数据预处理 ---
FAIL_COUNTS = [0, 1, 2, 3, 4]
TRIALS_PER_COUNT = 100
FAILURE_STEP = 180
STEADY_STATE_START = 350
COLORS = sns.color_palette("viridis", len(FAIL_COUNTS))

resilience_data = []  
degradation_list = [] 

def calc_paper_formation_error(actual_pos, target_pos):
    """消除平移和旋转的绝对几何编队误差"""
    N = actual_pos.shape[0]
    p_mat = actual_pos - np.mean(actual_pos, axis=0)
    q_mat = target_pos - np.mean(target_pos, axis=0)
    
    p = p_mat.flatten()[:, np.newaxis]
    q = q_mat.flatten()[:, np.newaxis]
    
    R_2x2 = np.array([[0, -1], [1, 0]])
    R = np.kron(np.eye(N), R_2x2) 
    
    norm_q_sq = np.sum(q**2)
    
    if norm_q_sq < 1e-8: 
        return np.sqrt(np.sum(p**2))
        
    q_perp = R @ q
    Q = (1.0 / norm_q_sq) * (q @ q.T + q_perp @ q_perp.T)
    
    norm_p_sq = np.sum(p**2)
    p_Q_p = float(p.T @ Q @ p)
    
    error_sq = norm_p_sq + norm_q_sq - 2.0 * np.sqrt(norm_q_sq) * np.sqrt(np.maximum(p_Q_p, 0.0))
    return np.sqrt(np.maximum(error_sq, 0.0))

def generate_circle_template(n_agents):

    R0 = 0.2
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

    return calc_paper_formation_error(curr_pos[sort_idx_curr], tgt_shape[sort_idx_tgt])

print("Processing datasets and computing Dynamic Formation Error...")

for fc in FAIL_COUNTS:
    all_trials = []
    
    for idx in range(TRIALS_PER_COUNT):
        file_path = f"data/fails_{fc}_{idx}.csv"
        if not os.path.exists(file_path): 
            continue
        
        df = pd.read_csv(file_path)
        steps = sorted(df['step'].unique())
        
        err_history = []
        for s in steps:
            curr_df = df[df['step'] == s]
            
            # 只提取存活的智能体
            if 'is_alive' in curr_df.columns:
                curr_df = curr_df[curr_df['is_alive'] == 1]
                
            curr_pos = curr_df[['pos_x', 'pos_y']].values
            n_agents = len(curr_pos)
            
            # 如果存活人数小于2人，无法构成编队，误差记为0或跳过
            if n_agents < 2:
                err_history.append({'step': s, 'error': 0.0})
                continue
            
            tgt_shape = generate_circle_template(n_agents)
            
            # 计算绝对几何误差
            err_dist = get_circular_formation_error(curr_pos, tgt_shape)
            err_history.append({'step': s, 'error': err_dist})
            
        trial_df = pd.DataFrame(err_history)
        all_trials.append(trial_df)
        
        # 提取重组后的稳态误差
        steady_val = trial_df[trial_df['step'] >= STEADY_STATE_START]['error'].mean()
        degradation_list.append({'FailCount': fc, 'SteadyError': steady_val})
    
    if len(all_trials) > 0:
        combined = pd.concat(all_trials)
        stats_df = combined.groupby('step')['error'].agg(['mean', 'std']).reset_index()
        stats_df['ci'] = 1.96 * (stats_df['std'] / np.sqrt(len(all_trials)))
        stats_df['fail_count'] = fc
        resilience_data.append(stats_df)

df_box = pd.DataFrame(degradation_list)

def plot_temporal_error():
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
    
    for i, fc in enumerate(FAIL_COUNTS):
        if i >= len(resilience_data): continue
        df_p = resilience_data[i]
        
        # 绘制误差均线与置信区间
        ax.plot(df_p['step'], df_p['mean'], color=COLORS[i], lw=2.5, label=f'$N_{{fail}}={fc}$')
        ax.fill_between(df_p['step'], np.clip(df_p['mean'] - df_p['ci'], 0, None), 
                         df_p['mean'] + df_p['ci'], 
                         color=COLORS[i], alpha=0.15, edgecolor='none')

    # 标注故障注入点
    ax.axvline(FAILURE_STEP, color='red', linestyle=':', lw=2)
    max_err = max([df['mean'].max() for df in resilience_data])
    ax.annotate('Failure Injection', xy=(FAILURE_STEP, max_err*0.8), xytext=(FAILURE_STEP-170, max_err*0.95),
                 arrowprops=dict(arrowstyle="->", color='red', connectionstyle="arc3,rad=-.2"), 
                 fontsize=8, color='red', fontweight='bold')

    ax.set_xlabel('Simulation Steps', fontsize=10, fontweight='bold')
    ax.set_ylabel("Formation error (m)", fontsize=10, fontweight='bold') 
    ax.legend(loc='upper right', ncol=1, frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
    
    ax.set_ylim(bottom=0) 
    ax.set_xlim(0, max(resilience_data[0]['step'])) 
    ax.tick_params(axis='both', labelsize=8)
    
    plt.tight_layout()
    os.makedirs("Paper_Figures", exist_ok=True)
    plt.savefig("Paper_Figures/Fig_Resilience_Temporal_Error.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ Fig A (Temporal Error Recovery) saved with size {FIG_WIDTH:.2f} x {FIG_HEIGHT:.2f} inches.")

plot_temporal_error()