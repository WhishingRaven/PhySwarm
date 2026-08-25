import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.collections import LineCollection 
import os
import matplotlib.ticker as ticker
import matplotlib.lines as mlines 

# ==========================================
#  统一的多剧本场景配置 
# ==========================================
SCENARIO_CONFIGS = [
    {
        'file_path': 'data/fails_0_4.csv',
        'title': "Scene 1: Standard Swarm\n \n8 Benign Robots",
        'lim_x': 1.1,
        'lim_y': 0.6,
        'has_failure': False,
        'focus_agent_id': 5
    },
    {
        'file_path': 'data/fails_2_0.csv',
        'title': "Scene 2: Fault-Tolerant Swarm\n \n8 Robots (2 Failed Robots)",
        'lim_x': 1.1,
        'lim_y': 0.6,
        'has_failure': True,
        'focus_agent_id': 0
    },
    {
        'file_path': 'data/expand_3_24.csv',
        'title': "Scene 3: Large-Scale Swarm\n \n24 Benign Robots",
        'lim_x': 2.1,
        'lim_y': 1.1,
        'has_failure': False,
        'focus_agent_id': 1
    }
]

FAIL_STEP = 300
TRANSITION_FRAMES = 60 # 2秒转场黑幕
INTRO_FRAMES = 120    # 4秒开场介绍(30fps)
PAUSE_DURATION = 120  # 关键物理事件发生时，画面暂停冻结4秒

PARAM_SMOOTH_WINDOW = 21

def smooth_parameter_by_role(df_agent, column, window=21):
    values = df_agent[column].astype(float).copy()
    if 'role' not in df_agent.columns:
        return values.rolling(window=window, center=True, min_periods=1).mean()

    role_block = df_agent['role'].ne(df_agent['role'].shift()).cumsum()
    return df_agent.groupby(role_block, group_keys=False)[column].transform(lambda x: x.astype(float).rolling(window=window, center=True, min_periods=1).mean())

NEST_POS = np.array([0.0, 0.0])

def get_subjective_potential_field(X_grid, Y_grid, w_food, w_nest, w_info, active_foods, info_agents_pos):
    """计算主观吸引势场梯度矢量 (用于对流流向)"""
    u_adv = np.zeros_like(X_grid)
    v_adv = np.zeros_like(Y_grid)
    potential = np.zeros_like(X_grid)
    sigma_sq = 0.25 
    
    if w_nest > 0.01:
        dist_sq = (X_grid - NEST_POS[0])**2 + (Y_grid - NEST_POS[1])**2
        pot_n = w_nest * np.exp(-dist_sq / sigma_sq)
        potential += pot_n
        u_adv += pot_n * (-2 * (X_grid - NEST_POS[0]) / sigma_sq)
        v_adv += pot_n * (-2 * (Y_grid - NEST_POS[1]) / sigma_sq)

    if w_food > 0.01 and len(active_foods) > 0:
        for fx, fy in active_foods:
            dist_sq = (X_grid - fx)**2 + (Y_grid - fy)**2
            pot_f = w_food * np.exp(-dist_sq / sigma_sq)
            potential += pot_f
            u_adv += pot_f * (-2 * (X_grid - fx) / sigma_sq)
            v_adv += pot_f * (-2 * (Y_grid - fy) / sigma_sq)
            
    if w_info > 0.01 and len(info_agents_pos) > 0:
        for ix, iy in info_agents_pos:
            dist_sq = (X_grid - ix)**2 + (Y_grid - iy)**2
            pot_i = (w_info * 0.5) * np.exp(-dist_sq / (sigma_sq*0.5))
            potential += pot_i
            u_adv += pot_i * (-2 * (X_grid - ix) / (sigma_sq*0.5))
            v_adv += pot_i * (-2 * (Y_grid - iy) / (sigma_sq*0.5))

    u_adv += 1e-7; v_adv += 1e-7
    return u_adv, v_adv, potential

def get_density_normalized_diffusion_field(
    X,
    Y,
    kd,
    agent_pos,
    focus_pos,
    lim_y,
    eps_rho=0.05,
    sigma_agent=0.060,
    sigma_wall=0.050,
    wall_weight=1.0
):
    """
    密度归一化扩散场：

        V_diff = -kd / (rho_norm + eps_rho) * grad(rho_norm)

    rho_norm 由机器人局部密度和上下墙壁密度构成。
    计算焦点机器人主观场时，不将焦点机器人自身计入邻居密度。
    """

    # ======================================
    # 1. 机器人连续密度场
    # ======================================
    rho_agent = np.zeros_like(X, dtype=float)

    for agent_x, agent_y in agent_pos:
        is_focus = (
            abs(agent_x - focus_pos[0]) < 1e-3
            and abs(agent_y - focus_pos[1]) < 1e-3
        )

        if is_focus:
            continue

        dist_sq = (
            (X - agent_x) ** 2
            + (Y - agent_y) ** 2
        )

        rho_agent += np.exp(
            -dist_sq / (2.0 * sigma_agent ** 2)
        )

    # ======================================
    # 2. 上下墙壁连续密度场
    # ======================================
    dist_top = np.abs(lim_y - Y)
    dist_bottom = np.abs(Y + lim_y)

    rho_wall = (
        np.exp(
            -dist_top ** 2 / (2.0 * sigma_wall ** 2)
        )
        +
        np.exp(
            -dist_bottom ** 2 / (2.0 * sigma_wall ** 2)
        )
    )

    # ======================================
    # 3. 合成并归一化局部密度
    # ======================================
    rho_raw = rho_agent + wall_weight * rho_wall

    rho_norm = rho_raw / (
        np.max(rho_raw) + 1e-8
    )

    # ======================================
    # 4. 使用真实网格间距计算梯度
    # ======================================
    dx = float(X[0, 1] - X[0, 0])
    dy = float(Y[1, 0] - Y[0, 0])

    grad_y, grad_x = np.gradient(
        rho_norm,
        dy,
        dx
    )

    # ======================================
    # 5. 扩散速度定义
    # ======================================
    diffusion_gain = kd / (
        rho_norm + eps_rho
    )

    u_diff = -diffusion_gain * grad_x
    v_diff = -diffusion_gain * grad_y

    # 地图范围外不绘制
    inside_mask = np.abs(Y) <= lim_y

    u_diff = np.where(
        inside_mask,
        u_diff,
        0.0
    )

    v_diff = np.where(
        inside_mask,
        v_diff,
        0.0
    )

    speed_diff = np.sqrt(
        u_diff ** 2 + v_diff ** 2
    )

    return u_diff, v_diff, speed_diff, rho_norm

all_datasets = []
max_agents_global = 0

step_scene1_random = 5 
step_scene1_info = 110  
step_scene1_food = 180  
step_scene1_nest = 360  

for d_idx, config in enumerate(SCENARIO_CONFIGS):
    f_path = config['file_path']
    if not os.path.exists(f_path):
        print(f"Warning: {f_path} not found. Skipping Scenario {d_idx+1} ...")
        continue
        
    print(f"[{d_idx+1}/{len(SCENARIO_CONFIGS)}] Pre-computing data for {f_path} ...")
    df = pd.read_csv(f_path)
    df = df[df['step'] <= 1000].copy()
    
    # 动态补全列
    if 'is_alive' not in df.columns:
        df['is_alive'] = 1  
    if 'is_carrying' not in df.columns:
        df['is_carrying'] = 0  
    for col in ['lambda_pick', 'lambda_drop', 'w_food', 'w_nest', 'w_info', 'w_rand', 'k_diff']:
        if col not in df.columns:
            df[col] = 0.0

    num_agents = df['agent_id'].nunique()
    max_agents_global = max(max_agents_global, num_agents)
    total_steps = df['step'].max()
    
    # 动态解析目标点
    target_cols = sorted([c for c in df.columns if c.startswith('target_') and c.endswith('_x')])
    food_pos = []
    for t_col in target_cols:
        t_idx = t_col.split('_')[1]
        tx = df.iloc[0][t_col]
        ty = df.iloc[0][f'target_{t_idx}_y']
        food_pos.append([tx, ty])
    food_pos = np.array(food_pos)
    
    # 自动分析场景 1 各物理相态切换的关键临界帧
    if d_idx == 0:
        configured_focus_id = int(
            config['focus_agent_id']
        )

        focus_df = df[
            df['agent_id'].astype(int).eq(
                configured_focus_id
            )
        ].sort_values('step')
        
        # 寻找信息共享场主导阈值
        found_info = False
        for _, r in focus_df.iterrows():
            w_sum = r['w_food'] + r['w_nest'] + r['w_info'] + r['w_rand'] + 1e-6
            if r['w_info'] / w_sum > 0.30 and r['is_carrying'] == 0:
                step_scene1_info = int(r['step'])
                found_info = True
                break
        if not found_info: step_scene1_info = 110

        step_scene1_info = 225
        
        # 寻找食物场主导阈值
        found_food = False
        for _, r in focus_df.iterrows():
            w_sum = r['w_food'] + r['w_nest'] + r['w_info'] + r['w_rand'] + 1e-6
            if r['w_food'] / w_sum > 0.40 and r['is_carrying'] == 0:
                step_scene1_food = int(r['step'])
                found_food = True
                break
        if not found_food: step_scene1_food = 180

        step_scene1_food = 120
        
        # 寻找巢穴场主导阈值
        found_nest = False
        for _, r in focus_df.iterrows():
            if r['is_carrying'] == 1:
                step_scene1_nest = int(r['step'])
                found_nest = True
                break
        if not found_nest: step_scene1_nest = 360

        step_scene1_nest = 160
        print(f"-> Scene 1 Phase Steps Detected | Random: {step_scene1_random} | Info-guided: {step_scene1_info} | Food-Seek: {step_scene1_food} | Homing: {step_scene1_nest}")

    # 提取轨迹
    agent_paths = {}
    for i in range(num_agents):
        sub_df = df[df['agent_id'] == i].sort_values('step')
        agent_paths[i] = {
            'x': sub_df['pos_x'].values, 'y': sub_df['pos_y'].values, 
            'alive': sub_df['is_alive'].values, 'carrying': sub_df['is_carrying'].values, 
            'steps': sub_df['step'].values
        }
        
    # 高分辨率流线与热力图网格
    X_GRID, Y_V_GRID = np.meshgrid(
        np.linspace(-config['lim_x'], config['lim_x'], 65),
        np.linspace(-config['lim_y'], config['lim_y'], 33)
    )

    # 稀疏扩散箭头网格
    X_Q_GRID, Y_Q_GRID = np.meshgrid(
        np.linspace(-config['lim_x'], config['lim_x'], 33),
        np.linspace(-config['lim_y'], config['lim_y'], 17)
    )
        
    all_datasets.append({
        'df': df,
        'paths': agent_paths,
        'total_steps': total_steps,
        'X_GRID': X_GRID,
        'Y_V_GRID': Y_V_GRID,
        'X_Q_GRID': X_Q_GRID,
        'Y_Q_GRID': Y_Q_GRID,
        'food_pos': food_pos,
        'lim_x': config['lim_x'],
        'lim_y': config['lim_y'],
        'title': config['title'],
        'has_failure': config['has_failure'],
        'focus_agent_id': config.get('focus_agent_id', 0)
    })

frames_schedule = []
for _ in range(INTRO_FRAMES):
    frames_schedule.append({'mode': 'intro', 'd_idx': 0, 'step': 1, 'pause_msg': None})

for d_idx, data in enumerate(all_datasets):
    # 转场
    for _ in range(TRANSITION_FRAMES):
        frames_schedule.append({'mode': 'transition', 'd_idx': d_idx, 'step': 1, 'pause_msg': None})
    
    # 状态锁
    triggered_rand = False
    triggered_info = False  
    triggered_food = False
    triggered_nest = False
    triggered_fail = False
    
    step = 1
    while step <= data['total_steps']:
        pause_msg = None
        trigger_pause = False
        
        # 第一、第二子视频动态插入暂停与解释文字逻辑
        if d_idx == 0:
            if step >= step_scene1_random and not triggered_rand:
                trigger_pause = True
                triggered_rand = True
                pause_msg = r"Phase I: Distributed Exploration\n\nRobots enter the $\sigma_{exp}$ phase.\nThe swarm disperses to explore the unknown workspace.".replace('\\n', '\n')
            elif step >= step_scene1_info and not triggered_info:
                trigger_pause = True
                triggered_info = True
                pause_msg = r"Phase IV: Peer Sharing and Trail Following\n\nRobots enter the $\sigma_{trail}$ phase.\nRobots follow peer-shared information or trail cues to navigate towards the resource region.".replace('\\n', '\n')
            elif step >= step_scene1_food and not triggered_food:
                trigger_pause = True
                triggered_food = True
                pause_msg = r"Phase II: Resource Discovery and Approach\n\nRobots enter the $\sigma_{app}$ phase.\nAfter a food source is localized, goal-directed attraction guides robots towards the resource.".replace('\\n', '\n')
            elif step >= step_scene1_nest and not triggered_nest:
                trigger_pause = True
                triggered_nest = True
                pause_msg = r"Phase III: Carrying and Homing\n\nRobots enter the $\sigma_{home}$ phase.\nRobots collect the food item and return to the nest.".replace('\\n', '\n')
        
        elif d_idx == 1:
            if step >= FAIL_STEP and not triggered_fail:
                trigger_pause = True
                triggered_fail = True
                pause_msg = "System Disturbance: Robot Failures\n\nTwo robots fail. The local repulsion field adapts\nto route the remaining robots safely around the failed nodes."
        
        # 触发物理阶段切变时，冻结步长并产生连续暂停帧
        if trigger_pause:
            for _ in range(PAUSE_DURATION):
                frames_schedule.append({'mode': 'sim', 'd_idx': d_idx, 'step': step, 'pause_msg': pause_msg})
                
        frames_schedule.append({'mode': 'sim', 'd_idx': d_idx, 'step': step, 'pause_msg': None})
        step += 3


fig = plt.figure(figsize=(14, 11), dpi=120)
gs = gridspec.GridSpec(2, 1, left=0.08, right=0.90, top=0.92, bottom=0.08, hspace=0.35, height_ratios=[2.2, 1.0])

ax_map = fig.add_subplot(gs[0, 0]) 
ax_param = fig.add_subplot(gs[1, 0])  

# --- 地图基础状态 ---
ax_map.set_aspect('equal'); ax_map.set_facecolor('#fdfdfd')
ax_map.scatter(0, 0, s=1200, c='green', alpha=0.4, edgecolors='green', lw=2)
ax_map.set_xlabel("x position (m)", fontweight='bold'); ax_map.set_ylabel("y position (m)", fontweight='bold')

trail_lines = [ax_map.plot([],[], lw=1.2, alpha=0.3, color='gray', zorder=1)[0] for _ in range(max_agents_global)]
scat_alive = ax_map.scatter([],[], s=100, edgecolors='k', linewidths=1.5, zorder=30)
scat_dead = ax_map.scatter([],[], s=110, c='crimson', edgecolors='k', zorder=11)
scat_dead_cross = ax_map.scatter([],[], s=35, color='white', marker='x', linewidths=1.5, zorder=12)
focus_halo = ax_map.scatter([],[], s=350, facecolors='none', edgecolors='gold', linewidths=3, zorder=31)

legend_elements = [
    mlines.Line2D([], [], color='dodgerblue', marker='o', linestyle='None', markersize=8, markeredgecolor='k', label='Searcher'),
    mlines.Line2D([], [], color='crimson', marker='o', linestyle='None', markersize=8, markeredgecolor='k', label='Carrier')
]
ax_map.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=10.5)

explanation_box = ax_map.text(0.04, 0.05, "", transform=ax_map.transAxes,
                              fontsize=11.5, fontweight='bold', color='black', family='sans-serif',
                              bbox=dict(boxstyle="round,pad=0.8", facecolor='#FFFEE6', alpha=0.92, edgecolor='darkorange', lw=2),
                              zorder=60, visible=False)

# --- 参数图 ---
ax_param.set_xlabel("Simulation Steps", fontweight='bold')
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

# 右轴
ax_param_twin = ax_param.twinx()
ax_param_twin.set_ylim(-0.01, 0.04) 
ax_param_twin.set_ylabel(r"Coefficient ($D$)", fontweight='bold', fontsize=10, labelpad=15)
ax_param_twin.tick_params(axis='y')
ax_param_twin.grid(False) 
ax_param_twin.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

line_k_diff, = ax_param_twin.plot([],[], color='#ff9a51', lw=2.5, label=r'$D$')

lines_all = [line_w_food_norm, line_w_nest_norm, line_w_info_norm, line_w_rand_norm, line_k_diff, line_l_pick, line_l_drop]
ax_param.legend(lines_all, [l.get_label() for l in lines_all], loc='lower center', 
                bbox_to_anchor=(0.5, 1.02), ncol=7, fontsize=9.5, framealpha=0.9)


overlay_rect = plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, color='black', alpha=0.85, zorder=100)
fig.patches.append(overlay_rect)
overlay_text = fig.text(0.5, 0.5, '', color='white', fontsize=22, fontweight='bold', ha='center', va='center', zorder=101)

food_circles = []
map_border = None

def update(frame_info):
    global food_circles, map_border, current_focus_id

    mode = frame_info['mode']
    d_idx = frame_info['d_idx']
    step = frame_info['step']
    pause_msg = frame_info['pause_msg']
    data = all_datasets[d_idx]
    ax_param_twin.set_ylim(-0.01, 0.04 if d_idx < 2 else 0.09)
    current_focus_id = int(data['focus_agent_id'])
    
    if mode == 'intro':
        overlay_rect.set_visible(True)
        overlay_text.set_visible(True)
        overlay_text.set_text("Density-field evolution and physical visualization\n \n \n Of Trail-Guided Swarm Foraging")
    elif mode == 'transition':
        overlay_rect.set_visible(True)
        overlay_text.set_visible(True)
        overlay_text.set_text(data['title'])

        focus_halo.set_offsets(np.empty((0, 2)))
        focus_halo.set_visible(False)
        
        ax_map.set_xlim(-data['lim_x'], data['lim_x'])
        ax_map.set_ylim(-data['lim_y'], data['lim_y'])
        
        if map_border:
            map_border.remove()
        lx, ly = data['lim_x'], data['lim_y']
        map_border, = ax_map.plot([-lx, lx, lx, -lx, -lx], [-ly, -ly, ly, ly, -ly], color='#444444', lw=4, zorder=0)
        
        for c in food_circles:
            c.remove()
        food_circles.clear()
        
        current_focus_id = int(data['focus_agent_id'])
        explanation_box.set_visible(False)
        return [scat_alive]
    else:
        overlay_rect.set_visible(False)
        overlay_text.set_visible(False)

    if pause_msg:
        explanation_box.set_text(pause_msg)
        explanation_box.set_visible(True)
    else:
        explanation_box.set_visible(False)

    # 绘制食物点
    if len(food_circles) == 0:
        for fp in data['food_pos']:
            c = patches.Circle((fp[0], fp[1]), 0.2, color='red', alpha=0.4, ec='darkred', lw=2, zorder=2)
            ax_map.add_patch(c)
            food_circles.append(c)

    ax_map.set_title(f"Density-field visualization", fontsize=14, fontweight='bold')
    ax_param.set_xlim(0, data['total_steps'])
    fail_line_p.set_visible(data['has_failure'])

    curr_df = data['df'][data['df']['step'] == step]
    if curr_df.empty: 
        return [scat_alive, scat_dead, scat_dead_cross, focus_halo] + trail_lines + food_circles

    # 清理上一帧的流场
    for coll in list(ax_map.collections):
        if coll.get_zorder() in [2.5, 2.6]: coll.remove()
    for patch in list(ax_map.patches):
        if patch.get_zorder() in [2.5, 2.6]: patch.remove()

    alive_df = curr_df[curr_df['is_alive'] == 1]
    dead_df = curr_df[curr_df['is_alive'] == 0]
    
    focus_agent_id = int(data['focus_agent_id'])

    # 当前帧中指定焦点机器人的唯一记录
    focus_df_curr = curr_df[
        curr_df['agent_id'].astype(int).eq(focus_agent_id)
    ].copy()

    if focus_df_curr.empty:
        print(
            f"[Warning] Scene {d_idx + 1}, step {step}: "
            f"Agent {focus_agent_id} is missing."
        )

        focus_halo.set_offsets(np.empty((0, 2)))
        focus_halo.set_visible(False)

    else:
        # 若同一 agent-step 存在重复记录，统一采用最后一条
        focus_row = focus_df_curr.iloc[-1]

        fx = float(focus_row['pos_x'])
        fy = float(focus_row['pos_y'])

        focus_alive = (
            bool(int(focus_row['is_alive']))
            if 'is_alive' in focus_row.index
            else True
        )

        # ==========================================
        # 1. 金色焦点圈
        # ==========================================
        if focus_alive:
            focus_halo.set_offsets(
                np.array([[fx, fy]], dtype=float)
            )
            focus_halo.set_visible(True)
        else:
            focus_halo.set_offsets(np.empty((0, 2)))
            focus_halo.set_visible(False)

        # ==========================================
        # 2. 当前可用食物
        # ==========================================
        active_foods = []

        for i in range(len(data['food_pos'])):
            cap_col = f'target_{i}_cap'

            if (
                cap_col in curr_df.columns
                and curr_df[cap_col].iloc[0] > 0
            ):
                active_foods.append(data['food_pos'][i])

                if i < len(food_circles):
                    food_circles[i].set_radius(
                        np.sqrt(
                            curr_df[cap_col].iloc[0] / 50.0
                        ) * 0.2
                    )

        # ==========================================
        # 3. 焦点机器人感知的信息机器人
        # ==========================================
        info_agents = alive_df[
            (alive_df['is_carrying'] == 1)
            & (
                alive_df['agent_id'].astype(int)
                != focus_agent_id
            )
        ]

        info_pos = (
            info_agents[['pos_x', 'pos_y']].values
            if not info_agents.empty
            else np.empty((0, 2))
        )

        # ==========================================
        # 4. 必须使用焦点机器人自身参数
        # ==========================================
        wf = float(focus_row['w_food'])
        wn = float(focus_row['w_nest'])
        wi = float(focus_row['w_info'])
        wr = float(focus_row['w_rand'])
        kd = float(focus_row['k_diff'])

        # ==========================================
        # 5. 平流场
        # ==========================================
        u_adv, v_adv, rho_adv = get_subjective_potential_field(
            data['X_GRID'],
            data['Y_V_GRID'],
            wf,
            wn,
            wi,
            active_foods,
            info_pos
        )

        if wr > 0.02:
            w_sum = wf + wn + wi + wr + 1e-6
            w_rand_norm = wr / w_sum

            chaos_noise = (
                0.40
                * np.cos(6.0 * data['X_GRID'])
                * np.sin(6.5 * data['Y_V_GRID'])
                +
                0.18
                * np.sin(12.0 * data['X_GRID'])
                * np.cos(10.5 * data['Y_V_GRID'])
            )

            rho_adv = np.clip(
                rho_adv * (
                    1.0 + w_rand_norm * chaos_noise
                )
                - w_rand_norm * 0.08,
                0.0,
                1.0
            )

        # 故障机器人仍留在环境中时，将全部机器人计入密度
        field_agent_pos = curr_df[
            ['pos_x', 'pos_y']
        ].values

        # ==========================================
        # 6. 高分辨率扩散场
        # ==========================================
        (
            u_d_stream,
            v_d_stream,
            V_diff_mag,
            rho_diff_norm
        ) = get_density_normalized_diffusion_field(
            X=data['X_GRID'],
            Y=data['Y_V_GRID'],
            kd=kd,
            agent_pos=field_agent_pos,
            focus_pos=(fx, fy),
            lim_y=data['lim_y'],
            eps_rho=0.05
        )

        u_total = u_adv + u_d_stream
        v_total = v_adv + v_d_stream

        positive_speed = V_diff_mag[
            V_diff_mag > 1e-10
        ]

        if positive_speed.size > 0:
            V_ref = np.percentile(
                positive_speed,
                95.0
            )
        else:
            V_ref = 1.0

        V_diff_vis = np.clip(
            V_diff_mag / (V_ref + 1e-8),
            0.0,
            1.0
        )

        rho_total = np.clip(
            rho_adv
            * np.exp(-0.35 * V_diff_vis),
            0.0,
            1.0
        )

        rho_normalized = (
            rho_total
            / max(np.max(rho_total), 1e-8)
        )

        ax_map.contourf(
            data['X_GRID'],
            data['Y_V_GRID'],
            rho_normalized,
            levels=14,
            cmap='Reds',
            alpha=0.35,
            zorder=2.5,
            antialiased=True
        )

        ax_map.streamplot(
            data['X_GRID'],
            data['Y_V_GRID'],
            u_total,
            v_total,
            color='darkred',
            linewidth=0.6,
            density=1.1,
            zorder=2.5,
            arrowsize=0.8
        )

        # ==========================================
        # 7. 稀疏网格蓝色扩散箭头
        # ==========================================
        u_d, v_d, mag_d, rho_d = (
            get_density_normalized_diffusion_field(
                X=data['X_Q_GRID'],
                Y=data['Y_Q_GRID'],
                kd=kd,
                agent_pos=field_agent_pos,
                focus_pos=(fx, fy),
                lim_y=data['lim_y'],
                eps_rho=0.05
            )
        )

        positive_mag = mag_d[
            mag_d > 1e-10
        ]

        if positive_mag.size > 0:
            mag_ref = np.percentile(
                positive_mag,
                90.0
            )

            mask_active = (
                mag_d > 0.01 * mag_ref
            )

            u_unit = np.divide(
                u_d,
                mag_d,
                out=np.zeros_like(u_d),
                where=mag_d > 1e-10
            )

            v_unit = np.divide(
                v_d,
                mag_d,
                out=np.zeros_like(v_d),
                where=mag_d > 1e-10
            )

            relative_mag = np.clip(
                mag_d / (mag_ref + 1e-8),
                0.25,
                1.0
            )

            arrow_length = (
                0.085
                if data['lim_x'] > 1.5
                else 0.045
            )

            u_show = (
                arrow_length
                * relative_mag
                * u_unit
            )

            v_show = (
                arrow_length
                * relative_mag
                * v_unit
            )

            ax_map.quiver(
                data['X_Q_GRID'][mask_active],
                data['Y_Q_GRID'][mask_active],
                u_show[mask_active],
                v_show[mask_active],
                color='#2457C5',
                angles='xy',
                scale_units='xy',
                scale=1.0,
                width=0.0035,
                headwidth=3.8,
                headlength=4.8,
                headaxislength=4.2,
                alpha=0.72,
                zorder=2.6
            )

    if not alive_df.empty:
        positions = alive_df[['pos_x', 'pos_y']].values
        colors = np.where(alive_df['is_carrying'].values, 'crimson', 'dodgerblue')
        scat_alive.set_offsets(positions)
        scat_alive.set_facecolors(colors)
    else:
        scat_alive.set_offsets(np.empty((0, 2)))

    dead_coords = dead_df[['pos_x', 'pos_y']] if not dead_df.empty else np.empty((0, 2))
    scat_dead.set_offsets(dead_coords)
    scat_dead_cross.set_offsets(dead_coords)
    
    # 动态更新轨迹线
    for i in range(max_agents_global):
        if i in data['paths']:
            idx = np.searchsorted(data['paths'][i]['steps'], step, side='right')
            trail_lines[i].set_data(data['paths'][i]['x'][:idx], data['paths'][i]['y'][:idx])
            if idx > 0:
                is_carry = data['paths'][i]['carrying'][idx-1]
                trail_lines[i].set_color('crimson' if is_carry else 'gray')
                if data['paths'][i]['alive'][idx-1] == 0:
                    trail_lines[i].set_alpha(0.05) 
                    trail_lines[i].set_color('gray')
                else:
                    trail_lines[i].set_alpha(0.3)
        else:
            trail_lines[i].set_data([], [])

    # 更新参数折线图
    if not alive_df.empty:
        focus_hist = data['df'][
            data['df']['agent_id'].astype(int) == focus_agent_id
        ].sort_values('step').copy()
        focus_hist['w_sum'] = focus_hist['w_food'] + focus_hist['w_nest'] + focus_hist['w_info'] + focus_hist['w_rand']

        focus_hist['w_food_norm'] = focus_hist['w_food'] / (focus_hist['w_sum'] + 1e-8)
        focus_hist['w_nest_norm'] = focus_hist['w_nest'] / (focus_hist['w_sum'] + 1e-8)
        focus_hist['w_info_norm'] = focus_hist['w_info'] / (focus_hist['w_sum'] + 1e-8)
        focus_hist['w_rand_norm'] = focus_hist['w_rand'] / (focus_hist['w_sum'] + 1e-8)

        focus_hist['w_food_smooth'] = smooth_parameter_by_role(focus_hist, 'w_food_norm', PARAM_SMOOTH_WINDOW)
        focus_hist['w_nest_smooth'] = smooth_parameter_by_role(focus_hist, 'w_nest_norm', PARAM_SMOOTH_WINDOW)
        focus_hist['w_info_smooth'] = smooth_parameter_by_role(focus_hist, 'w_info_norm', PARAM_SMOOTH_WINDOW)
        focus_hist['w_rand_smooth'] = smooth_parameter_by_role(focus_hist, 'w_rand_norm', PARAM_SMOOTH_WINDOW)
        focus_hist['lambda_pick_smooth'] = smooth_parameter_by_role(focus_hist, 'lambda_pick', PARAM_SMOOTH_WINDOW)
        focus_hist['lambda_drop_smooth'] = smooth_parameter_by_role(focus_hist, 'lambda_drop', PARAM_SMOOTH_WINDOW)
        focus_hist['k_diff_smooth'] = smooth_parameter_by_role(focus_hist, 'k_diff', PARAM_SMOOTH_WINDOW)

        smooth_w_sum = focus_hist['w_food_smooth'] + focus_hist['w_nest_smooth'] + focus_hist['w_info_smooth'] + focus_hist['w_rand_smooth'] + 1e-8
        focus_hist['w_food_smooth'] = focus_hist['w_food_smooth'] / smooth_w_sum
        focus_hist['w_nest_smooth'] = focus_hist['w_nest_smooth'] / smooth_w_sum
        focus_hist['w_info_smooth'] = focus_hist['w_info_smooth'] / smooth_w_sum
        focus_hist['w_rand_smooth'] = focus_hist['w_rand_smooth'] / smooth_w_sum

        focus_hist['lambda_pick_smooth'] = np.clip(focus_hist['lambda_pick_smooth'], 0.0, 1.0)
        focus_hist['lambda_drop_smooth'] = np.clip(focus_hist['lambda_drop_smooth'], 0.0, 1.0)
        focus_hist['k_diff_smooth'] = np.clip(focus_hist['k_diff_smooth'], 0.0, None)

        focus_hist = focus_hist[focus_hist['step'] <= step]
        line_w_food_norm.set_data(focus_hist['step'], focus_hist['w_food_smooth'])
        line_w_nest_norm.set_data(focus_hist['step'], focus_hist['w_nest_smooth'])
        line_w_info_norm.set_data(focus_hist['step'], focus_hist['w_info_smooth'])
        line_w_rand_norm.set_data(focus_hist['step'], focus_hist['w_rand_smooth'])
        line_l_pick.set_data(focus_hist['step'], focus_hist['lambda_pick_smooth'])
        line_l_drop.set_data(focus_hist['step'], focus_hist['lambda_drop_smooth'])
        line_k_diff.set_data(focus_hist['step'], focus_hist['k_diff_smooth'])

    return [scat_alive, scat_dead, scat_dead_cross, focus_halo, 
            line_w_food_norm, line_w_nest_norm, line_w_info_norm, line_w_rand_norm, 
            line_k_diff, line_l_pick, line_l_drop] + trail_lines + food_circles

print("Rendering Masterpiece compilation of Swarm Foraging Resilience...")
ani = animation.FuncAnimation(fig, update, frames=frames_schedule, interval=30, blit=False)
# plt.show()
os.makedirs('unified_video', exist_ok=True)
out_name = 'unified_video/forage_field.mp4'
ani.save(out_name, 
         writer='ffmpeg', 
         fps=30,       
         dpi=120,      
         bitrate=-1,   
         extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-preset', 'fast'])
print(f"Masterpiece Compilation Video Rendering Done! Enjoy your video: {out_name}")
