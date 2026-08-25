import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
import glob
import os

WIDTH_CM = 12.0     # 目标宽度 (厘米)
HEIGHT_CM = 8.0    # 目标高度 (厘米)
FIG_WIDTH = WIDTH_CM / 2.54
FIG_HEIGHT = HEIGHT_CM / 2.54

plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

METHODS = {
    'PhySwarm': {'pattern': 'data/fails_0_*.csv',      'color': '#0047AB'}, # 宝蓝色
    'FSM':    {'pattern': 'data/machine_*.csv', 'color': '#FF7F0E'}, # 活力橙
    'Ablation A':        {'pattern': 'data/static_1_*.csv','color': '#2CA02C'}, # 翡翠绿
    'Ablation B':        {'pattern': 'data/static_2_*.csv','color': '#D62728'}  # 玫瑰红
}

MAX_STEPS = 1000 
AVG_NEST_FOOD_DIST = 1.2 

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
        # 累计求和并向前填充到全尺寸 MAX_STEPS
        cum_delivery = delivery_series.reindex(range(MAX_STEPS)).fillna(0).cumsum()
        time_series_throughput[trial_idx, :] = cum_delivery.values
        
        total_delivered = cum_delivery.iloc[-1]
        nav_time = delivery_series[delivery_series > 0].index.max()
        if pd.isna(nav_time): nav_time = MAX_STEPS
            
        # ---------------------------------------------------
        # [雷达指标 A] 净运输经济性 (Net Transport Economy)
        # ---------------------------------------------------
        df_sorted = df.sort_values(by=['agent_id', 'step'])
        df_sorted['dx'] = df_sorted.groupby('agent_id')['pos_x'].diff()
        df_sorted['dy'] = df_sorted.groupby('agent_id')['pos_y'].diff()
        actual_distance = np.sqrt(df_sorted['dx']**2 + df_sorted['dy']**2).sum()
        
        ideal_distance = total_delivered * 2.0 * AVG_NEST_FOOD_DIST
        economy = ideal_distance / (actual_distance + 1e-6)
        
        # ---------------------------------------------------
        # [雷达指标 B] 单体相对效率 (Per-Robot Efficiency)
        # ---------------------------------------------------
        per_robot_efficiency = (total_delivered / nav_time) / n_agents

        # ---------------------------------------------------
        # [雷达指标 C] 碰撞干涉率 (Collision Rate) - 越低越好
        # ---------------------------------------------------
        collision_rate = df['is_colliding'].sum() / (n_agents * nav_time)
        
        # ---------------------------------------------------
        # [雷达指标 D] 控制平滑度 (Jerk) - 越低越好
        # ---------------------------------------------------
        df_sorted['ang_vel'] = df_sorted.groupby('agent_id')['angle'].diff()
        df_sorted.loc[df_sorted['ang_vel'] > np.pi, 'ang_vel'] -= 2 * np.pi
        df_sorted.loc[df_sorted['ang_vel'] < -np.pi, 'ang_vel'] += 2 * np.pi
        jerk = df_sorted.groupby('agent_id')['ang_vel'].diff().abs().mean()

        # ---------------------------------------------------
        # [雷达指标 E] 物理流形保真度 (ADR Manifold Fidelity)
        # ---------------------------------------------------
        # (为加速计算，每隔 5 步采样一次计算散度)
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
            'Jerk': jerk,
            'Fidelity': fidelity
        })
            
    return pd.DataFrame(metrics_list), time_series_throughput

print("Data processing engine ignited. Crunching numbers...")
results = {name: process_foraging_data(info['pattern']) for name, info in METHODS.items() if process_foraging_data(info['pattern']) is not None}
os.makedirs("Paper_Figures", exist_ok=True)

fig1, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
steps_arr = np.arange(MAX_STEPS)

# 记录0分消融实验的层叠偏移量，防止两条线完全重合看不见
zero_offset = 0.2 

for name in results:
    color = METHODS[name]['color']
    ts_throughput = results[name][1] # 取出累积吞吐量矩阵
    
    mean_thp = ts_throughput.mean(axis=0)
    std_thp = ts_throughput.std(axis=0)
    ci = 1.96 * (std_thp / np.sqrt(max(1, ts_throughput.shape[0])))
    
    if np.max(mean_thp) < 1.0 and "Ablation" in name:
        # 强制将0分线变成虚线，并稍微上浮一点点防止与X轴和彼此重叠
        plot_y = mean_thp + zero_offset
        line_style = '--' if "A" in name else ':'
        ax1.plot(steps_arr, plot_y, color=color, linewidth=2.5, linestyle=line_style, label=name, zorder=4)
        zero_offset += 0.2 # 下一个0分线再往上错开一点
    else:
        # 正常的高分曲线
        ax1.plot(steps_arr, mean_thp, color=color, linewidth=3.5, label=name, zorder=3)
        ax1.fill_between(steps_arr, np.clip(mean_thp - ci, 0, None), mean_thp + ci, 
                         color=color, alpha=0.15, edgecolor='none', zorder=2)

ax1.set_xlabel('Simulation Steps', fontsize=10, fontweight='bold')
ax1.set_ylabel("Cumulative Throughput", fontsize=10, fontweight='bold')
ax1.set_xlim(0, MAX_STEPS)

ax1.set_ylim(-0.2, np.max(results['PhySwarm'][1].mean(axis=0)) * 1.1) 

ax1.tick_params(axis='both', which='major', labelsize=8)

# 添加一条细细的 0 分基准线辅助视觉
ax1.axhline(0, color='black', linestyle='--', alpha=0.3, zorder=1)

ax1.legend(loc='upper left', fontsize=8, frameon=True, edgecolor='gray')

plt.tight_layout()
plt.savefig("Paper_Figures/Fig1_Foraging_Throughput_TimeSeries.pdf", bbox_inches='tight')
print("✅ Figure 1 Generated (with explicit ablation failure visualization).")
