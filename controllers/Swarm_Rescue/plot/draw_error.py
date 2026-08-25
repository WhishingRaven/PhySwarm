import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import glob
import os

WIDTH_CM = 12.0     # 目标宽度 (厘米)
HEIGHT_CM = 6.0     # 高度
FIG_WIDTH = WIDTH_CM / 2.54
FIG_HEIGHT = HEIGHT_CM / 2.54

plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42

plt.rcParams['axes.linewidth'] = 1.2        # 外轴线框粗细
plt.rcParams['xtick.major.width'] = 1.2     # X 轴主刻度线粗细
plt.rcParams['ytick.major.width'] = 1.2     # Y 轴主刻度线粗细

def smooth_data(data, window=15):
    if len(data) < window: return data
    return np.convolve(data, np.ones(window)/window, mode='same')

def plot_scientific_metric(steps, data_matrix, ylabel, save_filename, 
                           legend_label='Mean $E_{ADR}$', 
                           color_main='#4b0082', color_fill='#8a2be2'): 
    
    mean_val = np.mean(data_matrix, axis=0)
    lower_95 = np.percentile(data_matrix, 2.5, axis=0)
    upper_95 = np.percentile(data_matrix, 97.5, axis=0)
    lower_99 = np.percentile(data_matrix, 0.5, axis=0)
    upper_99 = np.percentile(data_matrix, 99.5, axis=0)
    
    win = max(1, len(steps) // 20)  
    mean_val = smooth_data(mean_val, win)
    lower_95, upper_95 = smooth_data(lower_95, win), smooth_data(upper_95, win)
    lower_99, upper_99 = smooth_data(lower_99, win), smooth_data(upper_99, win)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # 剔除边缘平滑带来的伪影
    plot_steps = steps[win:-win] if len(steps) > 2*win else steps

    reset_zones = [(100,250),(400,520)]

    for i, (start, end) in enumerate(reset_zones):
        current_label = 'Discover Target' if i == 0 else None
        
        ax.axvspan(start, end, color='gray', alpha=0.12, label=current_label, edgecolor='none', zorder=0)
    
    # 99% CI
    ax.fill_between(plot_steps, lower_99[win:-win], upper_99[win:-win], 
                    color=color_fill, alpha=0.15, label='99% CI', edgecolor='none', zorder=1)
    # 95% CI
    ax.fill_between(plot_steps, lower_95[win:-win], upper_95[win:-win], 
                    color=color_fill, alpha=0.3, label='95% CI', edgecolor='none', zorder=2)
    # 均线
    ax.plot(plot_steps, mean_val[win:-win], color=color_main, lw=2, label=legend_label, zorder=3)

    ax.set_xlabel('Simulation Steps', fontsize=8, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=8, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=6, edgecolor='gray')

    ax.set_xlim(plot_steps[0], plot_steps[-1])
    ax.set_ylim(0, np.max(upper_99[win:-win]) * 1.1)
    ax.tick_params(axis='both', which='major', labelsize=8, size=4, width=1.2)

    plt.tight_layout()
    os.makedirs("Paper_Figures", exist_ok=True)
    plt.savefig(f"Paper_Figures/{save_filename}", bbox_inches='tight')
    # plt.show()

X_LIMIT = 2.0 
Y_LIMIT = 1.0
GRID_RES_X = 60
GRID_RES_Y = 30
X, Y = np.meshgrid(np.linspace(-X_LIMIT, X_LIMIT, GRID_RES_X), 
                   np.linspace(-Y_LIMIT, Y_LIMIT, GRID_RES_Y))
grid_coords = np.vstack([X.ravel(), Y.ravel()])

def calc_rho_real(agents_pos):
    """
    实际 KDE 密度估计
    """
    N = len(agents_pos)
    if N == 0: return np.zeros_like(X) 

    bw = 0.30  

    try:
        noise = np.random.normal(0, 1e-4, agents_pos.shape)
        kernel = gaussian_kde((agents_pos + noise).T, bw_method=bw)
        rho_act = kernel(grid_coords).reshape(X.shape)
        return rho_act / (np.sum(rho_act) + 1e-8)
    except:
        rho = np.zeros_like(X)
        for pt in agents_pos:
            dist_sq = (X - pt[0])**2 + (Y - pt[1])**2
            rho += np.exp(-dist_sq / (2 * bw**2))
        return rho / (np.sum(rho) + 1e-8)

def calc_rho_search_th():
    """相态 A (搜索者/2D流形)：全局均匀分布"""
    rho = np.ones_like(X)
    return rho / np.sum(rho)

def calc_rho_respond_th(target_pos, w_target, k_diff):
    """相态 B (响应者/0D流形)：向目标点坍缩的高斯势阱"""
    dist_sq = (X - target_pos[0])**2 + (Y - target_pos[1])**2
    rho = np.exp(- (w_target / max(k_diff, 0.01)) * dist_sq)
    return rho / (np.sum(rho) + 1e-8)

def calc_rho_relay_th(base_pos, target_pos, w_center, k_diff):
    """相态 C (中继者/1D流形)：连接基地与目标的一维概率管 (Gaussian Tube)"""
    line_vec = target_pos - base_pos
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-6: return np.ones_like(X) / X.size
    
    line_dir = line_vec / line_len
    vec_to_base_x = X - base_pos[0]
    vec_to_base_y = Y - base_pos[1]
    
    proj_len = vec_to_base_x * line_dir[0] + vec_to_base_y * line_dir[1]
    perp_dist_sq = np.maximum((vec_to_base_x**2 + vec_to_base_y**2) - proj_len**2, 0)
    
    mask = (proj_len >= 0) & (proj_len <= line_len)
    
    rho = np.zeros_like(X)
    rho[mask] = np.exp(- (w_center / (max(k_diff, 0.01) + 0.1)) * perp_dist_sq[mask])
    return rho / (np.sum(rho) + 1e-8)

files = glob.glob("data/eadr_*.csv") 
if not files: 
    print("⚠️ 找不到文件，请检查通配符路径！")
    exit()

all_trials_errors = []
common_steps = None

for file_path in files:
    try:
        df = pd.read_csv(file_path)
    except: continue
    
    steps = sorted(df['step'].unique())
    if common_steps is None: common_steps = steps
    trial_errors = []
    
    for s in steps:
        group = df[df['step'] == s]
        if len(group) == 0: continue
        
        base_pos = np.array([group['base_x'].iloc[0], group['base_y'].iloc[0]])
        target_pos = np.array([group['target_0_x'].iloc[0], group['target_0_y'].iloc[0]])
        
        # 提取网络宏观输出
        w_target_mean = group['w_target'].mean()
        w_center_mean = group['w_center'].mean()
        k_diff_mean = group['k_diff'].mean()
        
        # 获取各相态的坐标
        searchers = group[group['role'] == 0][['pos_x', 'pos_y']].values
        responders = group[group['role'] == 1][['pos_x', 'pos_y']].values
        relays = group[group['role'] == 2][['pos_x', 'pos_y']].values
        
        err_s, err_resp, err_rel = 0.0, 0.0, 0.0
        
        # 1. 散度: 2D 探索流形
        if len(searchers) > 0:
            rho_th = calc_rho_search_th()
            rho_act = calc_rho_real(searchers)
            err_s = np.mean((rho_act - rho_th)**2)
            
        # 2. 散度: 0D 坍缩流形
        if len(responders) > 0:
            rho_th = calc_rho_respond_th(target_pos, w_target_mean, k_diff_mean)
            rho_act = calc_rho_real(responders)
            err_resp = np.mean((rho_act - rho_th)**2)
            
        # 3. 散度: 1D 连通流形
        if len(relays) > 0:
            rho_th = calc_rho_relay_th(base_pos, target_pos, w_center_mean, k_diff_mean)
            rho_act = calc_rho_real(relays)
            err_rel = np.mean((rho_act - rho_th)**2)
            
        # 根据当前各态人口比例进行加权
        N = len(group)
        weighted_err = ((len(searchers)/N) * err_s + 
                        (len(responders)/N) * err_resp + 
                        (len(relays)/N) * err_rel)
        
        trial_errors.append(weighted_err * 10000)
        
    if len(trial_errors) == len(common_steps):
        all_trials_errors.append(trial_errors)
    else:
        min_len = min(len(trial_errors), len(common_steps))
        common_steps = common_steps[:min_len]
        all_trials_errors = [t[:min_len] for t in all_trials_errors]
        all_trials_errors.append(trial_errors[:min_len])

if len(all_trials_errors) == 0:
    print("无有效数据。")
    exit()

error_matrix = np.array(all_trials_errors)


plot_scientific_metric(
    steps=np.array(common_steps), 
    data_matrix=error_matrix, 
    ylabel=r'Divergence ($E_{ADR}$)', 
    save_filename='Rescue_Topological_Divergence.pdf',
    legend_label='Mean $E_{ADR}$',
    color_main='#1a2a6c', 
    color_fill='#2c408b'
)
print("✅ 三相拓扑流形散度图已生成！")