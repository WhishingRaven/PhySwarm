import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import glob
import os

WIDTH_CM = 12.0     # 目标宽度 (厘米)
HEIGHT_CM = 6.0     # 比例：2:1
FIG_WIDTH = WIDTH_CM / 2.54
FIG_HEIGHT = HEIGHT_CM / 2.54

# ==========================================
# 1. 绘图样式函数 
# ==========================================
plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42

plt.rcParams['axes.linewidth'] = 1.2        
plt.rcParams['xtick.major.width'] = 1.2     
plt.rcParams['ytick.major.width'] = 1.2     

def smooth_data(data, window=15):
    if len(data) < window: return data
    return np.convolve(data, np.ones(window)/window, mode='same')

def plot_scientific_metric(steps, data_matrix, ylabel, save_filename, 
                           title=None, legend_label='Mean $E_{ADR}$', 
                           color_main='#1a2a6c', color_fill='#2c408b'): 
    
    mean_val = np.mean(data_matrix, axis=0)
    lower_95 = np.percentile(data_matrix, 2.5, axis=0)
    upper_95 = np.percentile(data_matrix, 97.5, axis=0)
    lower_99 = np.percentile(data_matrix, 0.5, axis=0)
    upper_99 = np.percentile(data_matrix, 99.5, axis=0)
    
    win = 30  
    mean_val = smooth_data(mean_val, win)
    lower_95, upper_95 = smooth_data(lower_95, win), smooth_data(upper_95, win)
    lower_99, upper_99 = smooth_data(lower_99, win), smooth_data(upper_99, win)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)

    # 99% CI
    ax.fill_between(steps[win:-win], lower_99[win:-win], upper_99[win:-win], 
                    color=color_fill, alpha=0.15, label='99% CI', edgecolor='none', zorder=1)
    # 95% CI
    ax.fill_between(steps[win:-win], lower_95[win:-win], upper_95[win:-win], 
                    color=color_fill, alpha=0.3, label='95% CI', edgecolor='none', zorder=2)
    # 均线
    ax.plot(steps[win:-win], mean_val[win:-win], color=color_main, lw=2, label=legend_label, zorder=3)

    ax.set_xlabel('Simulation Steps', fontsize=8, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=8, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=6, edgecolor='gray')
    
    ax.set_xlim(steps[win], steps[-win])
    ax.set_ylim(0, np.max(upper_99) * 1.1)
    ax.tick_params(axis='both', which='major', labelsize=8, size=4, width=1.2)

    plt.tight_layout()
    os.makedirs("Paper_Figures", exist_ok=True)
    plt.savefig(f"Paper_Figures/{save_filename}", bbox_inches='tight')
    plt.show()


GRID_RES_X = 80
GRID_RES_Y = 40
X_LIMIT = 2.0
Y_LIMIT = 1.0
X, Y = np.meshgrid(np.linspace(-X_LIMIT, X_LIMIT, GRID_RES_X), 
                   np.linspace(-Y_LIMIT, Y_LIMIT, GRID_RES_Y))

def calc_rho_explore_th():
    """🌟 Phase I (纯探索相)：全局均匀分布"""
    rho = np.ones_like(X)
    return rho / (np.sum(rho) + 1e-8)

def calc_rho_approach_th(food_positions, w_food, w_rand, k_diff):
    """🌟 Phase II (靠近食物相)：围绕所有活跃食物的多峰玻尔兹曼分布"""
    if len(food_positions) == 0 or w_food < 1e-3: 
        return calc_rho_explore_th()

    rho_total = np.zeros_like(X)
    effective_D = max(k_diff, 0.01) + 0.3 * w_rand 
    
    for fx, fy in food_positions:
        dist_sq = (X - fx)**2 + (Y - fy)**2
        rho_total += np.exp(- (w_food / effective_D) * dist_sq)
        
    return rho_total / (np.sum(rho_total) + 1e-8)

def calc_rho_return_th(nest_pos, w_nest, k_diff):
    """🌟 Phase III (归巢相)：围绕巢穴的单峰高斯势阱"""
    if w_nest < 1e-3: 
        return calc_rho_explore_th()
    dist_sq = (X - nest_pos[0])**2 + (Y - nest_pos[1])**2
    rho = np.exp(- (w_nest / max(k_diff, 0.01)) * dist_sq)
    return rho / (np.sum(rho) + 1e-8)

def calc_rho_real(agents_pos):
    """实际 KDE 核密度估计"""
    if len(agents_pos) < 3: 
        return np.zeros_like(X) 
    try:
        kernel = gaussian_kde(agents_pos.T, bw_method=0.3)
        grid_coords = np.vstack([X.ravel(), Y.ravel()])
        rho_act = kernel(grid_coords).reshape(X.shape)
        return rho_act / (np.sum(rho_act) + 1e-8)
    except:
        return np.zeros_like(X)


files = glob.glob("data/fails_0_*.csv")
if not files: 
    print("No target files found.")
    exit()

all_trials_errors = []
common_steps = None

for file_path in files:
    df = pd.read_csv(file_path)
    steps = sorted(df['step'].unique())
    if common_steps is None: common_steps = steps
    trial_errors = []
    
    for s in steps:
        group = df[df['step'] == s]
        
        # 1. 提取当前环境信息
        nest_pos = (group['nest_x'].iloc[0], group['nest_y'].iloc[0])
        active_foods = []
        for i in range(3): 
            cap_col = f'target_{i}_cap'
            if cap_col in group.columns and group[cap_col].iloc[0] > 0:
                active_foods.append((group[f'target_{i}_x'].iloc[0], group[f'target_{i}_y'].iloc[0]))
        
        # 2. 宏观参数平均值 
        w_nest_mean = group['w_nest'].mean()
        w_food_mean = group['w_food'].mean()
        w_rand_mean = group['w_rand'].mean()
        k_diff_mean = group['k_diff'].mean()
        
        returners_df = group[group['is_carrying'] == 1]
        searchers_df = group[group['is_carrying'] == 0]
        
        # 将未携带物料的智能体，按控制力强弱拆分为“纯探索”与“靠近食物”两类
        explorers_df = searchers_df[searchers_df['w_rand'] > searchers_df['w_food']]
        approachers_df = searchers_df[searchers_df['w_food'] >= searchers_df['w_rand']]
        
        explorers = explorers_df[['pos_x', 'pos_y']].values
        approachers = approachers_df[['pos_x', 'pos_y']].values
        returners = returners_df[['pos_x', 'pos_y']].values
        
        err_explore, err_approach, err_return = 0.0, 0.0, 0.0
        
        if len(explorers) > 0:
            rho_th_e = calc_rho_explore_th() 
            rho_act_e = calc_rho_real(explorers)
            err_explore = np.mean((rho_act_e - rho_th_e)**2)
            
        if len(approachers) > 0:
            rho_th_a = calc_rho_approach_th(active_foods, w_food_mean, w_rand_mean, k_diff_mean)
            rho_act_a = calc_rho_real(approachers)
            err_approach = np.mean((rho_act_a - rho_th_a)**2)
            
        if len(returners) > 0:
            rho_th_r = calc_rho_return_th(nest_pos, w_nest_mean, k_diff_mean)
            rho_act_r = calc_rho_real(returners)
            err_return = np.mean((rho_act_r - rho_th_r)**2)
            
        total_n = len(group)
        weighted_err = (
            (len(explorers) / total_n) * err_explore + 
            (len(approachers) / total_n) * err_approach + 
            (len(returners) / total_n) * err_return
        ) * 10000
        
        trial_errors.append(weighted_err)
        
    all_trials_errors.append(trial_errors)

error_matrix = np.array(all_trials_errors)


plot_scientific_metric(
    steps=np.array(common_steps), 
    data_matrix=error_matrix, 
    ylabel=r'Divergence ($E_{ADR}$)', # 🌟 修改 Y 轴标签
    save_filename='Foraging_ADR_Divergence_N24_Triphasic.pdf',
    title=None,
    legend_label='Mean $E_{ADR}$',
    color_main='#1a2a6c', 
    color_fill='#2c408b'
)