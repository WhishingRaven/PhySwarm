import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde
import matplotlib.patches as patches
from matplotlib.collections import LineCollection 
import glob
import os
import random
from matplotlib.colors import to_rgba

# ==========================================
#  视频转场与批量配置
# ==========================================
file_paths = sorted(glob.glob('data/fails_*_0.csv'))
file_paths = [f for f in file_paths if os.path.basename(f) != 'robust_data_0_0.csv']
if not file_paths:
    print("Error: No fails_*.csv files found in data/ folder.")
    exit()

TRANSITION_TITLES = [
    f"Scene 1\n \n12.5% Failed Robots",
    f"Scene 2\n \n25.0% Failed Robots",
    f"Scene 3\n \n37.5% Failed Robots",
    f"Scene 4\n \n50.0% Failed Robots"
]

FAIL_STEP = 300
TRANSITION_FRAMES = 60 # 2秒转场(30fps)
INTRO_FRAMES = 120    # 4秒开场介绍(30fps)

PARAM_SMOOTH_WINDOW = 21

def smooth_parameter_by_role(df_agent, column, window=21):
    values = df_agent[column].astype(float).copy()
    if 'role' not in df_agent.columns:
        return values.rolling(window=window, center=True, min_periods=1).mean()

    role_block = df_agent['role'].ne(df_agent['role'].shift()).cumsum()
    return df_agent.groupby(role_block, group_keys=False)[column].transform(lambda x: x.astype(float).rolling(window=window, center=True, min_periods=1).mean())


FOOD_POS = np.array([[-0.75, 0.25],[0.75, -0.25]])
NEST_POS = np.array([0.0, 0.0])

GRID_RES_X = 80
GRID_RES_Y = 40
X_LIMIT = 2.0
Y_LIMIT = 1.0
X, Y = np.meshgrid(np.linspace(-X_LIMIT, X_LIMIT, GRID_RES_X), 
                   np.linspace(-Y_LIMIT, Y_LIMIT, GRID_RES_Y))

def calc_rho_explore_th():
    """Phase I (纯探索相)：全局均匀分布"""
    rho = np.ones_like(X)
    return rho / (np.sum(rho) + 1e-8)

def calc_rho_approach_th(food_positions, w_food, w_rand, k_diff):
    """Phase II (靠近食物相)：围绕所有活跃食物的多峰玻尔兹曼分布"""
    if len(food_positions) == 0 or w_food < 1e-3: 
        return calc_rho_explore_th()

    rho_total = np.zeros_like(X)
    # w_rand 作为等效热噪声展宽
    effective_D = max(k_diff, 0.01) + 0.3 * w_rand 
    
    for fx, fy in food_positions:
        dist_sq = (X - fx)**2 + (Y - fy)**2
        rho_total += np.exp(- (w_food / effective_D) * dist_sq)
        
    return rho_total / (np.sum(rho_total) + 1e-8)

def calc_rho_return_th(nest_pos, w_nest, k_diff):
    """Phase III (归巢相)：围绕巢穴的单峰高斯势阱"""
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

def smooth(data, window=5):
    return np.convolve(data, np.ones(window)/window, mode='same')

all_datasets = []
max_agents = 0
global_max_food_cap = 50 

for idx, f_path in enumerate(file_paths):
    print(f"[{idx+1}/{len(file_paths)}] Loading and pre-computing {f_path} ...")
    df = pd.read_csv(f_path)
    num_agents = df['agent_id'].nunique()
    max_agents = max(max_agents, num_agents)
    total_steps = df['step'].max()
    
    if 'target_0_cap' in df.columns:
        global_max_food_cap = max(global_max_food_cap, df['target_0_cap'].max())

    safe_agent_ids = (
        df.groupby('agent_id')['is_alive']
        .apply(lambda values: (values == 1).all())
    )

    safe_agent_ids = safe_agent_ids[safe_agent_ids].index.to_numpy()

    if len(safe_agent_ids) == 0:
        raise ValueError(f"No always-alive agent found in {f_path}")

    selected_agent_id = int(random.choice(safe_agent_ids))
    df_agent = df[df['agent_id'] == selected_agent_id].sort_values('step').reset_index(drop=True)
    df_agent['w_sum'] = df_agent['w_rand'] + df_agent['w_info'] + df_agent['w_food'] + df_agent['w_nest'] + 1e-8
    df_agent['w_food_norm'] = df_agent['w_food'] / df_agent['w_sum']
    df_agent['w_nest_norm'] = df_agent['w_nest'] / df_agent['w_sum']
    df_agent['w_info_norm'] = df_agent['w_info'] / df_agent['w_sum']
    df_agent['w_rand_norm'] = df_agent['w_rand'] / df_agent['w_sum']

    df_agent['w_food_smooth'] = smooth_parameter_by_role(df_agent, 'w_food_norm', PARAM_SMOOTH_WINDOW)
    df_agent['w_nest_smooth'] = smooth_parameter_by_role(df_agent, 'w_nest_norm', PARAM_SMOOTH_WINDOW)
    df_agent['w_info_smooth'] = smooth_parameter_by_role(df_agent, 'w_info_norm', PARAM_SMOOTH_WINDOW)
    df_agent['w_rand_smooth'] = smooth_parameter_by_role(df_agent, 'w_rand_norm', PARAM_SMOOTH_WINDOW)
    df_agent['lambda_pick_smooth'] = smooth_parameter_by_role(df_agent, 'lambda_pick', PARAM_SMOOTH_WINDOW)
    df_agent['lambda_drop_smooth'] = smooth_parameter_by_role(df_agent, 'lambda_drop', PARAM_SMOOTH_WINDOW)
    df_agent['k_diff_smooth'] = smooth_parameter_by_role(df_agent, 'k_diff', PARAM_SMOOTH_WINDOW)

    smooth_w_sum = df_agent['w_food_smooth'] + df_agent['w_nest_smooth'] + df_agent['w_info_smooth'] + df_agent['w_rand_smooth'] + 1e-8
    df_agent['w_food_smooth'] = df_agent['w_food_smooth'] / smooth_w_sum
    df_agent['w_nest_smooth'] = df_agent['w_nest_smooth'] / smooth_w_sum
    df_agent['w_info_smooth'] = df_agent['w_info_smooth'] / smooth_w_sum
    df_agent['w_rand_smooth'] = df_agent['w_rand_smooth'] / smooth_w_sum

    df_agent['lambda_pick_smooth'] = np.clip(df_agent['lambda_pick_smooth'], 0.0, 1.0)
    df_agent['lambda_drop_smooth'] = np.clip(df_agent['lambda_drop_smooth'], 0.0, 1.0)
    df_agent['k_diff_smooth'] = np.clip(df_agent['k_diff_smooth'], 0.0, None)
    
    df_macro = df.groupby('step').agg({
        'w_food': 'mean', 'w_nest': 'mean', 'w_rand': 'mean', 'w_info': 'mean',
        'k_diff': 'mean'
    }).reset_index()
    
    # --- 动态三相流形散度 E_ADR 计算 ---
    e_adr_list = []
    for s in df_macro['step']:
        group = df[(df['step'] == s) & (df['is_alive'] == 1)]
        if len(group) < 2:
            e_adr_list.append(0); continue
            
        row = df_macro[df_macro['step'] == s].iloc[0]
        
        # 1. 动态提取当前物理环境特征
        nest_pos = (group['nest_x'].iloc[0], group['nest_y'].iloc[0]) if 'nest_x' in group.columns else NEST_POS
        active_foods = []
        for i in range(3): 
            cap_col = f'target_{i}_cap'
            if cap_col in group.columns and group[cap_col].iloc[0] > 0:
                tx = group[f'target_{i}_x'].iloc[0] if f'target_{i}_x' in group.columns else FOOD_POS[i][0]
                ty = group[f'target_{i}_y'].iloc[0] if f'target_{i}_y' in group.columns else FOOD_POS[i][1]
                active_foods.append((tx, ty))
        
        # 2. 宏观参数平均值
        w_nest_mean = row['w_nest']
        w_food_mean = row['w_food']
        w_rand_mean = row['w_rand']
        k_diff_mean = row['k_diff']
        
        # 3. 动态三相分离
        returners_df = group[group['is_carrying'] == 1]
        searchers_df = group[group['is_carrying'] == 0]
        explorers_df = searchers_df[searchers_df['w_rand'] > searchers_df['w_food']]
        approachers_df = searchers_df[searchers_df['w_food'] >= searchers_df['w_rand']]
        
        explorers = explorers_df[['pos_x', 'pos_y']].values
        approachers = approachers_df[['pos_x', 'pos_y']].values
        returners = returners_df[['pos_x', 'pos_y']].values
        
        err_explore, err_approach, err_return = 0.0, 0.0, 0.0
        
        # 4. 分别计算三相流形误差
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
        
        e_adr_list.append(weighted_err)
        
    df_macro['e_adr'] = smooth(e_adr_list, window=7)
    max_err = df_macro['e_adr'].max()
    
    # 提取个体路径
    agent_paths = {}
    for i in range(num_agents):
        sub_df = df[df['agent_id'] == i].sort_values('step')
        agent_paths[i] = {
            'x': sub_df['pos_x'].values, 'y': sub_df['pos_y'].values, 
            'alive': sub_df['is_alive'].values, 'carrying': sub_df['is_carrying'].values, 
            'steps': sub_df['step'].values
        }
        
    all_datasets.append({
        'df': df, 'macro': df_macro, 'paths': agent_paths, 
        'df_agent': df_agent, 'selected_agent_id': selected_agent_id, 
        'total_steps': total_steps, 'max_err': max_err, 
        'title': TRANSITION_TITLES[min(idx, len(TRANSITION_TITLES)-1)]
    })

print("All pre-computations finished! Generating timeline...")


frames_schedule = []
for _ in range(INTRO_FRAMES):
    frames_schedule.append({'mode': 'intro', 'd_idx': 0, 'step': 1})
for d_idx, data in enumerate(all_datasets):
    for _ in range(TRANSITION_FRAMES):
        frames_schedule.append({'mode': 'transition', 'd_idx': d_idx, 'step': 1})
    for step in range(1, data['total_steps'], 3): 
        frames_schedule.append({'mode': 'sim', 'd_idx': d_idx, 'step': step})

fig = plt.figure(figsize=(12, 11), dpi=150) 

gs_top = gridspec.GridSpec(1, 1, left=0.08, right=0.92, top=0.92, bottom=0.56) 
gs_bottom = gridspec.GridSpec(2, 1, left=0.17, right=0.83, top=0.48, bottom=0.05, hspace=0.68)

ax_map = fig.add_subplot(gs_top[0, 0]) 
ax_metric = fig.add_subplot(gs_bottom[0, 0]) 
ax_param = fig.add_subplot(gs_bottom[1, 0])  

# --- 地图设置 ---
ax_map.set_xlim(-1.2, 1.2); ax_map.set_ylim(-0.6, 0.6)
ax_map.set_aspect('equal'); ax_map.set_facecolor('#fdfdfd')
ax_map.set_xlabel("x position (m)", fontweight='bold', fontsize=12); ax_map.set_ylabel("y position (m)", fontweight='bold', fontsize=12)

# 绘制巢穴与食物
ax_map.scatter(0, 0, s=1200, c='green', alpha=0.4, edgecolors='green', lw=1, zorder=1)
scat_nest = ax_map.scatter([],[], s=80, c='green', edgecolors='green', lw=1, zorder=10, label='Nest')

food_circles = []
for fp in FOOD_POS:
    circle = patches.Circle((fp[0], fp[1]), 0.25, color='red', alpha=0.4, ec='darkred', lw=1, zorder=2)
    ax_map.add_patch(circle)
    food_circles.append(circle)

# 轨迹线集合 
trail_cols = []

for _ in range(max_agents):
    # 线宽、颜色和透明度将在每一帧中按轨迹状态分别设置
    lc = LineCollection(
        [],
        zorder=3,
    )

    ax_map.add_collection(lc)
    trail_cols.append(lc)

# 散点图层
scat_search = ax_map.scatter([],[], s=80, c='dodgerblue', edgecolors='k', zorder=10, label='Searching')
scat_carry = ax_map.scatter([],[], s=80, c='red', edgecolors='k', zorder=10, label='Carrying')

scat_tracked = ax_map.scatter([],[], s=200, facecolors='none', edgecolors='gold', lw=3, zorder=50, label='Tracked Agent')
scat_dead = ax_map.scatter([],[], s=100, c='crimson', edgecolors='k', zorder=11, label='Failed')
scat_dead_cross = ax_map.scatter([],[], s=40, color='white', marker='x', linewidths=2, zorder=12)

ax_map.legend(handles=[scat_search, scat_carry], loc='upper right', fontsize=10.5, framealpha=0.9, edgecolor='#EAEAEA')

# --- 第二张子图：宏观 E_ADR ---
ax_metric.set_ylabel(r'Divergence ($E_{ADR}$)', fontweight='bold', fontsize=12)
ax_metric.set_xlabel("Simulation Steps", fontweight='bold', fontsize=12)
line_e, = ax_metric.plot([],[], color='#1a2a6c', lw=2.5)
fail_line_e = ax_metric.axvline(FAIL_STEP, color='red', linestyle='--', alpha=0.7)
fail_text_e = ax_metric.text(FAIL_STEP+5, 0, "Failure Injection", color='red', fontweight='bold')
ax_metric.grid(True, alpha=0.3)

# --- 第三张子图：双轴参数重分配 ---
ax_param.set_xlabel("Simulation Steps", fontweight='bold', fontsize=12)

ax_param.set_ylim(-0.05, 1.1) 
ax_param.set_ylabel(r"Coefficient ($\tilde{\omega}, \lambda$)", fontweight='bold', fontsize=10)

line_w_rand_norm, = ax_param.plot([],[], color='#ff2200', lw=2.5, label=r'$\tilde{\omega}_{rand}$')
line_w_info_norm, = ax_param.plot([],[], color='#8000ff', lw=2.5, label=r'$\tilde{\omega}_{info}$')
line_w_food_norm, = ax_param.plot([],[], color='#16c41c', lw=2.5, label=r'$\tilde{\omega}_{food}$')
line_w_nest_norm, = ax_param.plot([],[], color='#3dd5e9', lw=2.5, label=r'$\tilde{\omega}_{nest}$')
line_l_pick, = ax_param.plot([],[], color='#4b6cff', lw=2.5, label=r'$\lambda_{pick}$')
line_l_drop, = ax_param.plot([],[], color='#1697f3', lw=2.5, label=r'$\lambda_{drop}$')

fail_line_p = ax_param.axvline(FAIL_STEP, color='red', linestyle='--', alpha=0.7)
ax_param.grid(True, alpha=0.3)

# 【右侧轴：D】
ax_param_twin = ax_param.twinx()
ax_param_twin.set_ylim(-0.01, 0.04) 
ax_param_twin.set_ylabel(r"Coefficient ($D$)", fontweight='bold', fontsize=10, labelpad=15)
ax_param_twin.tick_params(axis='y')
ax_param_twin.grid(False) 
ax_param_twin.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

line_k_diff, = ax_param_twin.plot([],[], color='#ff9a51', lw=2.5, label=r'$D$')

# 合并参数图图例
lines_all = [line_w_food_norm, line_w_nest_norm, line_w_info_norm, line_w_rand_norm, line_k_diff, line_l_pick, line_l_drop]
ax_param.legend(lines_all, [l.get_label() for l in lines_all], loc='lower center', 
                bbox_to_anchor=(0.5, 1.02), ncol=7, fontsize=9.5, framealpha=0.9)


overlay_rect = plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, color='black', alpha=0.85, zorder=100)
fig.patches.append(overlay_rect)
overlay_text = fig.text(0.5, 0.5, '', color='white', fontsize=24, fontweight='bold', ha='center', va='center', zorder=101)

# ==========================================
# 7. 动画驱动逻辑
# ==========================================
def update(frame_info):
    mode = frame_info['mode']
    d_idx = frame_info['d_idx']
    step = frame_info['step']
    data = all_datasets[d_idx]

    if mode == 'intro':
        overlay_rect.set_visible(True)
        overlay_text.set_visible(True)
        overlay_text.set_text("Fault-tolerance verification in simulation\n \n \n For Trail-Guided Swarm Foraging")
    
    # --- 转场控制 ---
    elif mode == 'transition':
        overlay_rect.set_visible(True)
        overlay_text.set_visible(True)
        overlay_text.set_text(data['title']) 
    else:
        overlay_rect.set_visible(False)
        overlay_text.set_visible(False)

    ax_map.set_title(f"Fault-Tolerance Verification", fontsize=14, fontweight='bold')
    ax_metric.set_xlim(0, data['total_steps'])
    ax_param.set_xlim(0, data['total_steps'])
    
    max_e = data['max_err']
    y_lim_top = max_e * 1.1 if max_e > 0 else 0.05
    ax_metric.set_ylim(0, y_lim_top)
    fail_text_e.set_position((FAIL_STEP + 5, y_lim_top * 0.8)) 

    # 故障参考线显示控制
    show_failure_marks = (d_idx >= 0)
    fail_line_e.set_visible(show_failure_marks)
    fail_line_p.set_visible(show_failure_marks)
    fail_text_e.set_visible(show_failure_marks)

    # --- 获取数据帧并更新画面 ---
    curr_df = data['df'][data['df']['step'] == step]
    if curr_df.empty: 
        return [scat_search, scat_carry, scat_dead, scat_dead_cross, scat_tracked, line_e, 
                line_w_food_norm, line_w_nest_norm, line_w_info_norm, line_w_rand_norm, 
                line_k_diff, line_l_pick, line_l_drop] + trail_cols + food_circles
    
    alive = curr_df[curr_df['is_alive'] == 1]
    dead = curr_df[curr_df['is_alive'] == 0]
    
    scat_search.set_offsets(alive[alive['is_carrying'] == 0][['pos_x', 'pos_y']] if not alive.empty else np.empty((0,2)))
    scat_carry.set_offsets(alive[alive['is_carrying'] == 1][['pos_x', 'pos_y']] if not alive.empty else np.empty((0,2)))
    
    dead_coords = dead[['pos_x', 'pos_y']] if not dead.empty else np.empty((0,2))
    scat_dead.set_offsets(dead_coords)
    scat_dead_cross.set_offsets(dead_coords)
    
    # 动态更新被追踪智能体的光圈及状态
    tracked_agent_id = data['selected_agent_id']
    tracked_data = curr_df[curr_df['agent_id'] == tracked_agent_id]
    
    if not tracked_data.empty:
        scat_tracked.set_offsets(tracked_data[['pos_x', 'pos_y']].values)
    else:
        scat_tracked.set_offsets(np.empty((0, 2)))
    
    # 动态食物缩放
    for i in range(len(FOOD_POS)):
        cap = curr_df[f'target_{i}_cap'].iloc[0] if f'target_{i}_cap' in curr_df.columns else 0
        food_circles[i].set_radius(np.sqrt(max(cap, 0) / (global_max_food_cap + 1e-6)) * 0.25)


    for i in range(max_agents):
        if i not in data['paths']:
            trail_cols[i].set_segments([])
            continue

        path_data = data['paths'][i]

        steps_arr = path_data['steps']

        current_idx = np.searchsorted(
            steps_arr,
            step,
            side='right',
        )

        if current_idx < 2:
            trail_cols[i].set_segments([])
            continue

        hist_x = path_data['x'][:current_idx].astype(float)
        hist_y = path_data['y'][:current_idx].astype(float)

        carrying = (
            path_data['carrying'][:current_idx]
            > 0
        )

        is_alive_status = bool(
            path_data['alive'][current_idx - 1]
        )

        # 相邻位置点组成轨迹线段
        points = np.column_stack(
            [hist_x, hist_y]
        ).reshape(-1, 1, 2)

        segments = np.concatenate(
            [
                points[:-1],
                points[1:],
            ],
            axis=1,
        )

        # 每条线段的状态取起点对应的 carrying 状态
        segment_carrying = carrying[:-1]

        if is_alive_status:
            search_alpha = 0.4
            carry_alpha = 0.8
        else:
            # 故障机器人轨迹整体褪色
            search_alpha = 0.1
            carry_alpha = 0.1

        segment_colors = [
            to_rgba(
                'crimson',
                carry_alpha,
            )
            if is_carrying
            else to_rgba(
                'gray',
                search_alpha,
            )
            for is_carrying in segment_carrying
        ]

        segment_widths = np.where(
            segment_carrying,
            2.5,  # 搬运轨迹
            1.2,  # 搜索轨迹
        )

        trail_cols[i].set_segments(segments)
        trail_cols[i].set_colors(segment_colors)
        trail_cols[i].set_linewidths(segment_widths)

        # 透明度已经写入每条线段的 RGBA 颜色，
        # 不要再为整个 LineCollection 设置统一 alpha
        trail_cols[i].set_alpha(None)

    # 宏观 E_ADR 更新
    hist_macro = data['macro'][data['macro']['step'] <= step]
    line_e.set_data(hist_macro['step'], hist_macro['e_adr'])
    
    # 微观单体参数更新 (左轴线更新、右轴线异步更新)
    df_agent = data['df_agent']
    hist_agent = df_agent[df_agent['step'] <= step]
    
    # 归一化平流项与反应项至左轴
    line_w_food_norm.set_data(hist_agent['step'], hist_agent['w_food_smooth'])
    line_w_nest_norm.set_data(hist_agent['step'], hist_agent['w_nest_smooth'])
    line_w_info_norm.set_data(hist_agent['step'], hist_agent['w_info_smooth'])
    line_w_rand_norm.set_data(hist_agent['step'], hist_agent['w_rand_smooth'])
    line_l_pick.set_data(hist_agent['step'], hist_agent['lambda_pick_smooth'])
    line_l_drop.set_data(hist_agent['step'], hist_agent['lambda_drop_smooth'])
    
    # 物理扩散项至右轴
    line_k_diff.set_data(hist_agent['step'], hist_agent['k_diff_smooth'])

    return [scat_search, scat_carry, scat_dead, scat_dead_cross, scat_tracked, line_e, 
            line_w_food_norm, line_w_nest_norm, line_w_info_norm, line_w_rand_norm, 
            line_k_diff, line_l_pick, line_l_drop] + trail_cols + food_circles

print("Starting render engine. This may take a few minutes for a massive compilation video...")
ani = animation.FuncAnimation(fig, update, frames=frames_schedule, interval=30, blit=False)
# plt.show()
out_name = 'elasticity_video/foraging_resilience_compilation.mp4'
os.makedirs('elasticity_video', exist_ok=True)
print(f"Saving compilation video to {out_name} ...")

ani.save(out_name, 
         writer='ffmpeg', 
         fps=30,       
         dpi=150,      
         bitrate=-1,   
         extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '15', '-preset', 'fast'])
print("Masterpiece Rendering Done! Enjoy your video.")
