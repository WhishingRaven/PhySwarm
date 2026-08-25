import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
import glob
import os

plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'

METHODS = {
    'PhySwarm': {'pattern': 'data/fails_0_*.csv',      'color': '#0047AB'}, # 宝蓝色
    'FSM':    {'pattern': 'data/machine_*.csv', 'color': '#FF7F0E'}, # 活力橙
    'Ablation A':        {'pattern': 'data/static_1_*.csv','color': '#2CA02C'}, # 翡翠绿
    'Ablation B':        {'pattern': 'data/static_2_*.csv','color': '#D62728'}  # 玫瑰红
}

MAX_STEPS = 1000 
AVG_NEST_FOOD_DIST = 0.8 

GRID_RES = 50
XY_RANGE = 2.0 
X_GRID, Y_GRID = np.meshgrid(np.linspace(-XY_RANGE, XY_RANGE, GRID_RES), 
                             np.linspace(-XY_RANGE, XY_RANGE, GRID_RES))

def calc_rho_return_th(nest_pos, w_nest, k_diff):
    if w_nest < 1e-3: return np.ones_like(X_GRID) / X_GRID.size
    dist_sq = (X_GRID - nest_pos[0])**2 + (Y_GRID - nest_pos[1])**2
    rho = np.exp(- (w_nest / max(k_diff, 0.01)) * dist_sq)
    return rho / (np.sum(rho) + 1e-8)

def calc_rho_search_th(food_positions, w_food, w_rand, k_diff):
    if w_food < 1e-3 or len(food_positions) == 0: 
        return np.ones_like(X_GRID) / X_GRID.size
    rho_total = np.zeros_like(X_GRID)
    effective_D = max(k_diff, 0.01) + 0.3 * w_rand 
    for fx, fy in food_positions:
        dist_sq = (X_GRID - fx)**2 + (Y_GRID - fy)**2
        rho_total += np.exp(- (w_food / effective_D) * dist_sq)
    return rho_total / (np.sum(rho_total) + 1e-8)

def calc_rho_real(agents_pos):
    if len(agents_pos) < 3: return np.zeros_like(X_GRID) 
    try:
        kernel = gaussian_kde(agents_pos.T, bw_method=0.3)
        grid_coords = np.vstack([X_GRID.ravel(), Y_GRID.ravel()])
        rho_act = kernel(grid_coords).reshape(X_GRID.shape)
        return rho_act / (np.sum(rho_act) + 1e-8)
    except:
        return np.zeros_like(X_GRID)

def process_foraging_data(file_pattern):
    files = glob.glob(file_pattern)
    if not files: return None
    
    metrics_list = []
    time_series_throughput = np.zeros((len(files), MAX_STEPS))
    
    for trial_idx, f in enumerate(files):
        df = pd.read_csv(f)
        steps = sorted(df['step'].unique())
        n_agents = len(df['agent_id'].unique())
        
        # ---------------------------------------------------
        # [指标 1] 宏观时序吞吐量 (Cumulative Throughput) - 供折线图使用
        # ---------------------------------------------------
        delivery_series = df.groupby('step')['delivery_event'].sum()
        cum_delivery = delivery_series.reindex(range(MAX_STEPS)).fillna(0).cumsum()
        time_series_throughput[trial_idx, :] = cum_delivery.values
        
        total_delivered = cum_delivery.iloc[-1]
        nav_time = delivery_series[delivery_series > 0].index.max()
        if pd.isna(nav_time): nav_time = MAX_STEPS
            
        # ---------------------------------------------------
        # [雷达指标 A] 净运输经济性 (Net Transport Economy) - E_trans
        # ---------------------------------------------------
        df_sorted = df.sort_values(by=['agent_id', 'step'])
        df_sorted['dx'] = df_sorted.groupby('agent_id')['pos_x'].diff()
        df_sorted['dy'] = df_sorted.groupby('agent_id')['pos_y'].diff()
        actual_distance = np.sqrt(df_sorted['dx']**2 + df_sorted['dy']**2).sum()
        
        ideal_distance = total_delivered * 2.0 * AVG_NEST_FOOD_DIST
        economy = ideal_distance / (actual_distance + 1e-6)
        
        # ---------------------------------------------------
        # [雷达指标 B] 单体相对效率 (Per-Robot Efficiency) - Phi_ind
        # ---------------------------------------------------
        per_robot_efficiency = (total_delivered / nav_time) / n_agents

        # ---------------------------------------------------
        # [雷达指标 C] 碰撞安全率 (Collision Safety) - C_rate (对齐公式 12)
        # ---------------------------------------------------
        collision_rate = df['is_colliding'].sum() / (n_agents * MAX_STEPS) if (n_agents * MAX_STEPS) > 0 else 0.0
        
        # ---------------------------------------------------
        # [雷达指标 D] 控制平滑度 (Control Smoothness) - J_smooth (对齐公式 11)
        # ---------------------------------------------------
        df_sorted['diff_l'] = df_sorted.groupby('agent_id')['left_speed'].diff().abs()
        df_sorted['diff_r'] = df_sorted.groupby('agent_id')['right_speed'].diff().abs()
        df_sorted['c_diff_l1'] = df_sorted['diff_l'] + df_sorted['diff_r']
        j_smooth = df_sorted['c_diff_l1'].mean() # 自动忽略首步 NaN，分母精准对齐 N*(T-1)

        # ---------------------------------------------------
        # [雷达指标 E] 物理流形保真度 (ADR Manifold Fidelity)
        # ---------------------------------------------------
        trial_errors = []
        for s in steps[::5]:
            step_df = df[df['step'] == s]
            nest_pos = (step_df['nest_x'].iloc[0], step_df['nest_y'].iloc[0])
            active_foods = [(step_df[f'target_{i}_x'].iloc[0], step_df[f'target_{i}_y'].iloc[0]) 
                            for i in range(5) if f'target_{i}_cap' in step_df.columns and step_df[f'target_{i}_cap'].iloc[0] > 0]
            
            searchers = step_df[step_df['is_carrying'] == 0][['pos_x', 'pos_y']].values
            returners = step_df[step_df['is_carrying'] == 1][['pos_x', 'pos_y']].values
            
            err_search, err_return = 0.0, 0.0
            if len(searchers) >= 3:
                err_search = np.mean((calc_rho_real(searchers) - calc_rho_search_th(active_foods, step_df['w_food'].mean(), step_df['w_rand'].mean(), step_df['k_diff'].mean()))**2)
            if len(returners) >= 3:
                err_return = np.mean((calc_rho_real(returners) - calc_rho_return_th(nest_pos, step_df['w_nest'].mean(), step_df['k_diff'].mean()))**2)
                
            combined_error = (len(searchers)/len(step_df))*err_search + (len(returners)/len(step_df))*err_return
            trial_errors.append(combined_error * 10000)
            
        fidelity = 1.0 / (1.0 + np.mean(trial_errors))

        # --- 收集单次试验指标 ---
        metrics_list.append({
            'Economy': economy,
            'Efficiency': per_robot_efficiency,
            'Collision_Rate': collision_rate,
            'J_smooth': j_smooth, 
            'Fidelity': fidelity
        })
            
    return pd.DataFrame(metrics_list), time_series_throughput

print("Data processing engine ignited. Crunching numbers...")
results = {name: process_foraging_data(info['pattern']) for name, info in METHODS.items() if process_foraging_data(info['pattern']) is not None}
os.makedirs("Paper_Figures", exist_ok=True)

# ==========================================
#  图 2：四维雷达图 (时空控制安全综合表现 - 极简学术符号版)
# ==========================================
radar_labels = [
    r'$\Phi_{\mathrm{ind}}$',    # 右上：单体相对效率 (Per-Robot Efficiency)
    r'$J_{\mathrm{smooth}}$',    # 右下：控制平滑度 (Control Smoothness)
    r'$R_{\mathrm{coll}}$',     # 左下：碰撞安全 (Collision Safety)
    r'$E_{\mathrm{trans}}$'      # 左上：净运输经济性 (Transport Economy)
]
n_metrics = len(radar_labels)
raw_df = pd.DataFrame({name: res[0].mean() for name, res in results.items()}).T

BASE_SCORE = 0.2

def normalize_higher_is_better(series):
    v_min, v_max = series.min(), series.max()
    if v_max == v_min: return pd.Series(1.0, index=series.index)
    return BASE_SCORE + (1.0 - BASE_SCORE) * (series - v_min) / (v_max - v_min)

def normalize_lower_is_better(series):
    v_min, v_max = series.min(), series.max()
    if v_max == v_min: return pd.Series(1.0, index=series.index)
    return BASE_SCORE + (1.0 - BASE_SCORE) * (v_max - series) / (v_max - v_min)

norm_data = {}
# 正向指标 (越大越好)
eff_scores = normalize_higher_is_better(raw_df['Efficiency'])
eco_scores = normalize_higher_is_better(raw_df['Economy'])     
# 反向指标 (越小越好) -> 反转为越大越安全/平滑
smo_scores = normalize_lower_is_better(raw_df['J_smooth'])  
saf_scores = normalize_lower_is_better(raw_df['Collision_Rate'])

for name in raw_df.index:
    norm_data[name] = [
        eff_scores[name], # 右上
        smo_scores[name], # 右下 
        saf_scores[name], # 左下
        eco_scores[name]  # 左上
    ]

angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist() + [0]

fig2, ax2 = plt.subplots(figsize=(3.8, 3.8), subplot_kw=dict(polar=True), dpi=300)

# 精确限定极坐标轴圆形的大小，底部预留 22% 的边距用于摆放图例
ax2.set_position([0.15, 0.22, 0.70, 0.70]) 

ax2.set_theta_offset(np.pi / 4) # 旋转 45 度使得 4 个极轴刚好对齐正方形的 4 个对角
ax2.set_theta_direction(-1)     # 顺时针排列
ax2.grid(color='#E3E3E3', linestyle='--', alpha=0.8, linewidth=0.8)

ax2.set_xticks(angles[:-1])
ticks = ax2.set_xticklabels(radar_labels, size=11.5, fontweight='bold')

# 标签整体向外推开 10 个像素，彻底防止顶点重合
ax2.tick_params(axis='x', pad=10)

# Index 0: 右上 (\Phi_ind)
ticks[0].set_horizontalalignment('left')
ticks[0].set_verticalalignment('bottom')

# Index 1: 右下 (J_smooth)
ticks[1].set_horizontalalignment('left')
ticks[1].set_verticalalignment('top')

# Index 2: 左下 (C_rate)
ticks[2].set_horizontalalignment('right')
ticks[2].set_verticalalignment('top')

# Index 3: 左上 (E_trans)
ticks[3].set_horizontalalignment('right')
ticks[3].set_verticalalignment('bottom')

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

plt.savefig("Paper_Figures/Fig2_Foraging_Radar_4D.pdf", bbox_inches='tight')
plt.show()
print("✅ Figure 2 (4D Foraging Radar Chart with Transport Economy) Generated.")
