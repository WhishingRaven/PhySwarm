import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.collections import LineCollection 
import os
import matplotlib.ticker as ticker


datasets_config = [
    {
        'file_path': 'data/fails_0_0.csv',
        'title': "Scene 1: Standard Swarm\n \n8 Benign Robots",
        'is_extended': False,
        'has_failures': False,
        'max_steps': 500,          
        'key_steps': [50, 130, 250, 400],
        'labels': ["formation-keeping", "goal-directed navigation", "morphology-adaptation", "formation-recovery"]
    },
    {
        'file_path': 'data/fails_2_0.csv',
        'title': "Scene 2: Fault-Tolerant Swarm\n \n8 Robots (2 Failed Robots)",
        'is_extended': False,
        'has_failures': True,
        'max_steps': 400,          
        'key_steps': [30, 120, 200, 300],
        'labels': ["formation-keeping", "goal-directed navigation", "morphology-adaptation", "formation-recovery"]
    },
    {
        'file_path': 'data/expand_0.4_24.csv',
        'title': "Scene 3: Large-Scale Swarm\n \n24 Benign Robots",
        'is_extended': True,
        'has_failures': False,
        'max_steps': 1000,         
        'key_steps': [100, 300, 600, 800],
        'labels': ["formation-keeping", "goal-directed navigation", "morphology-adaptation", "formation-recovery"]
    }
]

FAIL_STEP = 180
TRANSITION_FRAMES = 60 # 2秒转场(30fps)
INTRO_FRAMES = 120    # 4秒开场介绍(30fps)
PAUSE_DURATION = 120  # 🌟 关键物理事件发生时，画面暂停冻结4秒

WORLD_WIDTH = 6.0
WORLD_HEIGHT = 2.0

X_MIN = -WORLD_WIDTH / 2.0       # -3.0
X_MAX = WORLD_WIDTH / 2.0        #  3.0
Y_MIN = -WORLD_HEIGHT / 2.0      # -1.0
Y_MAX = WORLD_HEIGHT / 2.0       #  1.0

OPEN_HALF_WIDTH = 1.0
NARROW_HALF_WIDTH = 0.38

START_FUNNEL_X = -1.7
ENTRANCE_X = -0.5
EXIT_X = 0.9
END_TRANSITION_X = 1.5

# 绘图上下额外留白
Y_PLOT_MARGIN = 0.10
Y_PLOT_MIN = Y_MIN - Y_PLOT_MARGIN
Y_PLOT_MAX = Y_MAX + Y_PLOT_MARGIN

PARAM_SMOOTH_WINDOW = 21

def smooth_parameter_by_role(df_agent, column, window=21):
    values = df_agent[column].astype(float).copy()
    if 'role' not in df_agent.columns:
        return values.rolling(window=window, center=True, min_periods=1).mean()

    role_block = df_agent['role'].ne(df_agent['role'].shift()).cumsum()
    return df_agent.groupby(role_block, group_keys=False)[column].transform(lambda x: x.astype(float).rolling(window=window, center=True, min_periods=1).mean())

R0 = 0.2
W_SHAPE_IDEAL = 0.12
K_DIFF_IDEAL = 0.05
SHAPE_GAIN = 50.0

def get_corridor_geometry(px):
    if px < -0.7: 
        return 0.6, 0.0
    elif -0.7 <= px < -0.1: 
        return 0.6 + (px + 0.7)/0.6 * (0.12 - 0.6), -0.8
    elif -0.1 <= px < 0.6: 
        return 0.12, 0.0
    elif 0.6 <= px < 0.9: 
        return 0.12 + (px - 0.6)/0.3 * (0.6 - 0.12), 1.6
    else: 
        return 0.6, 0.0

def get_corridor_geometry_ext(px):
    if px < START_FUNNEL_X:
        return OPEN_HALF_WIDTH, 0.0
    elif px < ENTRANCE_X:
        slope = (
            (NARROW_HALF_WIDTH - OPEN_HALF_WIDTH)
            / (ENTRANCE_X - START_FUNNEL_X)
        )
        half_width = (
            OPEN_HALF_WIDTH
            + (px - START_FUNNEL_X) * slope
        )
        return half_width, slope
    elif px < EXIT_X:
        return NARROW_HALF_WIDTH, 0.0
    elif px < END_TRANSITION_X:
        slope = (
            (OPEN_HALF_WIDTH - NARROW_HALF_WIDTH)
            / (END_TRANSITION_X - EXIT_X)
        )
        half_width = (
            NARROW_HALF_WIDTH
            + (px - EXIT_X) * slope
        )
        return half_width, slope
    else:
        return OPEN_HALF_WIDTH, 0.0


all_datasets = []
max_agents_global = 0

step_scene1_keep = 30
step_scene1_nav = 130
step_scene1_morph = 250
step_scene1_rec = 400

for idx, conf in enumerate(datasets_config):
    f_path = conf['file_path']
    is_ext = conf['is_extended']
    limit_steps = conf['max_steps']
    
    if not os.path.exists(f_path):
        print(f"Warning: {f_path} not found. Skipped.")
        continue
        
    print(f"[{idx+1}/3] Loading and Pre-computing {f_path} (Limit: {limit_steps} steps) ...")
    df = pd.read_csv(f_path)
    df = df[df['step'] <= limit_steps].copy()
    
    num_agents = df['agent_id'].nunique()
    max_agents_global = max(max_agents_global, num_agents)
    total_steps = df['step'].max()
    
    # df_macro 整合
    df_macro = df.groupby('step').agg({'w_flow':'mean', 'w_shape':'mean', 'k_diff':'mean', 'beta':'mean'}).reset_index()
    df_macro['w_sum'] = df_macro['w_flow'] + df_macro['w_shape'] + 1e-8
    df_macro['lambda_val'] = (df_macro['beta'] - 1.0) / 4.0
    df_macro['w_flow_norm'] = df_macro['w_flow'] / df_macro['w_sum']
    df_macro['w_shape_norm'] = df_macro['w_shape'] / df_macro['w_sum']

    df_macro['w_flow_smooth'] = smooth_parameter_by_role(df_macro, 'w_flow_norm', PARAM_SMOOTH_WINDOW)
    df_macro['w_shape_smooth'] = smooth_parameter_by_role(df_macro, 'w_shape_norm', PARAM_SMOOTH_WINDOW)
    df_macro['lambda_smooth'] = smooth_parameter_by_role(df_macro, 'lambda_val', PARAM_SMOOTH_WINDOW)
    df_macro['k_diff_smooth'] = smooth_parameter_by_role(df_macro, 'k_diff', PARAM_SMOOTH_WINDOW)

    smooth_w_sum = df_macro['w_flow_smooth'] + df_macro['w_shape_smooth'] + 1e-8
    df_macro['w_flow_smooth'] = df_macro['w_flow_smooth'] / smooth_w_sum
    df_macro['w_shape_smooth'] = df_macro['w_shape_smooth'] / smooth_w_sum

    df_macro['lambda_smooth'] = np.clip(df_macro['lambda_smooth'], 0.0, 1.0)
    df_macro['k_diff_smooth'] = np.clip(df_macro['k_diff_smooth'], 0.0, None)
    
    # 场景 1 阶段临界帧捕获
    if idx == 0:
        step_scene1_keep = 10
        
        found_p2 = False
        for _, r in df_macro.iterrows():
            if 0.3 <= r['lambda_val'] <= 0.5:
                step_scene1_nav = int(r['step'])
                found_p2 = True
                break
        if not found_p2: step_scene1_nav = 130

        step_scene1_nav = 60
        
        found_p3 = False
        for _, r in df_macro.iterrows():
            if 0.98 <= r['lambda_val'] <= 1.0:
                step_scene1_morph = int(r['step'])
                found_p3 = True
                break
        if not found_p3: step_scene1_morph = 250

        step_scene1_morph = 120
        
        found_p4 = False
        for _, r in df_macro.iterrows():
            if r['step'] > step_scene1_morph and r['lambda_val'] <= 0.3:
                step_scene1_rec = int(r['step'])
                found_p4 = True
                break
        if not found_p4: step_scene1_rec = 400

        step_scene1_rec = 230

        print(f"-> Scene 1 Phasing Detected | Keep: {step_scene1_keep} | Nav: {step_scene1_nav} | Morph: {step_scene1_morph} | Recover: {step_scene1_rec}")

    agent_paths = {}
    for i in range(num_agents):
        sub_df = df[df['agent_id'] == i].sort_values('step')
        agent_paths[i] = {
            'x': sub_df['pos_x'].values, 'y': sub_df['pos_y'].values, 
            'alive': sub_df['is_alive'].values if 'is_alive' in sub_df.columns else np.ones(len(sub_df)),
            'steps': sub_df['step'].values
        }
        
    all_datasets.append({
        'df': df, 'macro': df_macro, 'paths': agent_paths, 
        'total_steps': total_steps, 
        'is_extended': is_ext, 'has_failures': conf['has_failures'],
        'title': conf['title']
    })


frames_schedule = []
for _ in range(INTRO_FRAMES):
    frames_schedule.append({'mode': 'intro', 'd_idx': 0, 'step': 1, 'pause_msg': None})

for d_idx, data in enumerate(all_datasets):
    for _ in range(TRANSITION_FRAMES):
        frames_schedule.append({'mode': 'transition', 'd_idx': d_idx, 'step': 1, 'pause_msg': None})
        
    triggered_p1 = False
    triggered_p2 = False
    triggered_p3 = False
    triggered_p4 = False
    triggered_fail = False
    
    step = 1
    while step <= data['total_steps']:
        pause_msg = None
        trigger_pause = False
        
        if d_idx == 0:
            if step >= step_scene1_keep and not triggered_p1:
                trigger_pause = True
                triggered_p1 = True
                pause_msg = r"Phase I: Initial Formation Keeping\n\nRobots enter the $\sigma_{keep}$ phase.\nThe swarm aggregates from disordered states and establishes a stable circular formation.".replace('\\n', '\n')
            elif step >= step_scene1_nav and not triggered_p2:
                trigger_pause = True
                triggered_p2 = True
                pause_msg = r"Phase II: Goal-Directed Navigation\n\nRobots enter the $\sigma_{nav}$ phase.\nThe swarm moves towards the target while maintaining formation coherence.".replace('\\n', '\n')
            elif step >= step_scene1_morph and not triggered_p3:
                trigger_pause = True
                triggered_p3 = True
                pause_msg = r"Phase III: Morphology Adaptation\n\nRobots enter the $\sigma_{morph}$ phase.\nThe swarm continuously reshapes its geometry to satisfy corridor constraints.".replace('\\n', '\n')
            elif step >= step_scene1_rec and not triggered_p4:
                trigger_pause = True
                triggered_p4 = True
                pause_msg = r"Phase IV: Formation Recovery\n\nRobots enter the $\sigma_{rec}$ phase.\nAfter leaving the bottleneck, the swarm reconstructs the standard circular formation.".replace('\\n', '\n')

        elif d_idx == 1:
            if step >= FAIL_STEP and not triggered_fail:
                trigger_pause = True
                triggered_fail = True
                pause_msg = "System Disturbance: Robot Failures\n\nTwo robots fail. The local repulsion field adapts\nto route the remaining robots safely around the failed nodes."
        
        if trigger_pause:
            for _ in range(PAUSE_DURATION):
                frames_schedule.append({'mode': 'sim', 'd_idx': d_idx, 'step': step, 'pause_msg': pause_msg})
                
        frames_schedule.append({'mode': 'sim', 'd_idx': d_idx, 'step': step, 'pause_msg': None})
        step += 2


fig = plt.figure(figsize=(14, 9.5), dpi=120) 
gs = gridspec.GridSpec(2, 1, figure=fig, left=0.08, right=0.90, top=0.92, bottom=0.08, hspace=0.35, height_ratios=[1.8, 1.0])

ax_map = fig.add_subplot(gs[0, 0]) 
ax_param = fig.add_subplot(gs[1, 0])  

ax_map.set_xlabel("x position (m)", fontweight='bold'); ax_map.set_ylabel("y position (m)", fontweight='bold')


trail_lines_base = []
for i in range(max_agents_global):
    line_base, = ax_map.plot([], [], lw=1.2, alpha=0.35, color='#95A5A6', zorder=10) 
    trail_lines_base.append(line_base)

scat_alive = ax_map.scatter([],[], s=95, c='dodgerblue', edgecolors='k', zorder=20, label='Benign Robot')
scat_dead = ax_map.scatter([],[], s=95, c='crimson', edgecolors='k', zorder=21, label='Failed Robot')
scat_dead_cross = ax_map.scatter([],[], s=40, color='white', marker='x', linewidths=2, zorder=22)

# 阶段解释框
explanation_box = ax_map.text(0.04, 0.05, "", transform=ax_map.transAxes,
                              fontsize=11.5, fontweight='bold', color='black', family='sans-serif',
                              bbox=dict(boxstyle="round,pad=0.8", facecolor='#FFFEE6', alpha=0.92, edgecolor='darkorange', lw=2),
                              zorder=60, visible=False)

# --- 第二张子图：双轴参数层 ---
ax_param.set_xlabel("Simulation Steps", fontweight='bold')
ax_param.set_ylim(-0.05, 1.1) 
ax_param.set_ylabel(r"Coefficient ($\tilde{\omega}, \lambda$)", fontweight='bold', fontsize=10)
line_w_norm, = ax_param.plot([],[], color='#16c41c', lw=2.5, label=r'$\tilde{\omega}_{flow}$')
line_w_shape_norm, = ax_param.plot([],[], color='#4d4dfb', lw=2.5, label=r'$\tilde{\omega}_{shape}$')
line_lambda, = ax_param.plot([],[], color='#1697f3', lw=2.5, label=r'$\lambda$')
fail_line_p = ax_param.axvline(FAIL_STEP, color='red', linestyle='--', alpha=0.7)
ax_param.grid(True, alpha=0.3)

ax_param_twin = ax_param.twinx()
ax_param_twin.set_ylim(-0.01, 0.04) 
ax_param_twin.set_ylabel(r"Coefficient ($D$)", fontweight='bold', fontsize=10, labelpad=15)
ax_param_twin.tick_params(axis='y')
ax_param_twin.grid(False) 

line_k_diff, = ax_param_twin.plot([],[], color='#ff9a51', lw=2.5, label=r'$D$')

lines_p = [line_w_norm, line_w_shape_norm, line_lambda, line_k_diff]
ax_param.legend(lines_p, [l.get_label() for l in lines_p], loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, framealpha=0.9, fontsize=10)

current_d_idx = -1
geom_lines = []



def update_static_geometry(d_idx):
    global current_d_idx, geom_lines
    if d_idx == current_d_idx:
        return
    current_d_idx = d_idx
    data = all_datasets[d_idx]
    is_ext = data['is_extended']
    
    for line in geom_lines:
        try: line.remove()
        except: pass
    geom_lines.clear()
    
    for coll in list(ax_map.collections):
        if coll.get_label() in ['Fill_Wall', 'Target_Star']:
            coll.remove()
            
    if is_ext:
        ax_map.set_xlim(-3.0, 3.0)
        ax_map.set_ylim(-1.1, 1.1)

        walls_x = [
            X_MIN,
            START_FUNNEL_X,
            ENTRANCE_X,
            EXIT_X,
            END_TRANSITION_X,
            X_MAX,
        ]

        walls_y_upper = [
            OPEN_HALF_WIDTH,
            OPEN_HALF_WIDTH,
            NARROW_HALF_WIDTH,
            NARROW_HALF_WIDTH,
            OPEN_HALF_WIDTH,
            OPEN_HALF_WIDTH,
        ]

        walls_y_lower = [
            -OPEN_HALF_WIDTH,
            -OPEN_HALF_WIDTH,
            -NARROW_HALF_WIDTH,
            -NARROW_HALF_WIDTH,
            -OPEN_HALF_WIDTH,
            -OPEN_HALF_WIDTH,
        ]
        
        l1, = ax_map.plot(walls_x, walls_y_upper, color='#444444', lw=3, zorder=0)
        l2, = ax_map.plot(walls_x, walls_y_lower, color='#444444', lw=3, zorder=0)

        ax_map.fill_between(
            walls_x,
            walls_y_upper,
            Y_PLOT_MAX,
            color="gray",
            alpha=0.20,
            zorder=3,
        )

        ax_map.fill_between(
            walls_x,
            Y_PLOT_MIN,
            walls_y_lower,
            color="gray",
            alpha=0.20,
            zorder=3,
        )
        
        ax_map.scatter([2.0], [0.0], marker='*', s=400, color='gold', edgecolors='k', lw=1.5, zorder=9, label='Target_Star')
        geom_lines.extend([l1, l2])
    else:
        ax_map.set_xlim(-1.5, 1.5)
        ax_map.set_ylim(-0.6, 0.6)
        
        walls_x = [-1.5, -0.7, -0.1, 0.6, 0.9, 1.5]
        walls_y_upper = [0.6, 0.6, 0.12, 0.12, 0.6, 0.6]
        walls_y_lower = [-0.6, -0.6, -0.12, -0.12, -0.6, -0.6]
        
        l1, = ax_map.plot(walls_x, walls_y_upper, color='#444444', lw=3, zorder=0)
        l2, = ax_map.plot(walls_x, walls_y_lower, color='#444444', lw=3, zorder=0)
        ax_map.fill_between(walls_x, walls_y_upper, 0.6, color='gray', alpha=0.20, label='Fill_Wall', zorder=0)
        ax_map.fill_between(walls_x, -0.6, walls_y_lower, color='gray', alpha=0.20, label='Fill_Wall', zorder=0)
        
        ax_map.scatter([1.05], [0.0], marker='*', s=400, color='gold', edgecolors='k', lw=1.5, zorder=9, label='Target_Star')
        geom_lines.extend([l1, l2])

def get_corridor_half_width_field(XQ, is_ext):
    """返回网格中每个位置对应的通道半宽。"""
    if is_ext:
        open_hw = OPEN_HALF_WIDTH       # 1.0
        narrow_hw = NARROW_HALF_WIDTH   # 0.38
        start_x = START_FUNNEL_X         # -1.7
        entrance_x = ENTRANCE_X          # -0.5
        exit_x = EXIT_X                  # 0.9
        end_x = END_TRANSITION_X         # 1.5
    else:
        open_hw = 0.60
        narrow_hw = 0.12
        start_x = -0.70
        entrance_x = -0.10
        exit_x = 0.60
        end_x = 0.90

    hw = np.full_like(XQ, open_hw, dtype=float)

    mask = (XQ >= start_x) & (XQ < entrance_x)
    hw[mask] = (
        open_hw
        + (XQ[mask] - start_x)
        / (entrance_x - start_x)
        * (narrow_hw - open_hw)
    )

    mask = (XQ >= entrance_x) & (XQ < exit_x)
    hw[mask] = narrow_hw

    mask = (XQ >= exit_x) & (XQ < end_x)
    hw[mask] = (
        narrow_hw
        + (XQ[mask] - exit_x)
        / (end_x - exit_x)
        * (open_hw - narrow_hw)
    )

    return hw

def get_grid_mesh(is_ext):
    if is_ext:
        # 红色平流场网格
        XV, YV = np.meshgrid(
            np.linspace(-3.1, 3.1, 120),
            np.linspace(-1.1, 1.1, 41)
        )

        # 蓝色扩散箭头网格：纵向使用奇数点，保证关于 y=0 对称
        xq = np.linspace(-3.1, 3.1, 81)
        yq = np.linspace(-1.1, 1.1, 31)
    else:
        XV, YV = np.meshgrid(
            np.linspace(-1.6, 1.6, 40),
            np.linspace(-0.6, 0.6, 25)
        )

        xq = np.linspace(-1.6, 1.6, 33)
        yq = np.linspace(-0.6, 0.6, 25)

    XQ, YQ = np.meshgrid(xq, yq)

    return XV, YV, XQ, YQ, xq, yq

def get_diffusion_field_vectorized(
    kd,
    agent_pos,
    is_ext,
    XQ,
    YQ,
    eps_rho=0.05,
    sigma_agent=0.060,
    sigma_wall=0.050,
    wall_weight=1.0
):

    # --------------------------------------
    # 1. 机器人连续密度场
    # --------------------------------------
    rho_agent = np.zeros_like(XQ, dtype=float)

    for agent_x, agent_y in agent_pos:
        dist_sq = (
            (XQ - agent_x)**2
            + (YQ - agent_y)**2
        )

        rho_agent += np.exp(
            -dist_sq / (2.0 * sigma_agent**2)
        )

    # --------------------------------------
    # 2. 墙壁连续密度场
    # --------------------------------------
    half_width = get_corridor_half_width_field(
        XQ, is_ext
    )

    dist_top = np.abs(half_width - YQ)
    dist_bottom = np.abs(YQ + half_width)

    rho_wall = (
        np.exp(
            -dist_top**2
            / (2.0 * sigma_wall**2)
        )
        + np.exp(
            -dist_bottom**2
            / (2.0 * sigma_wall**2)
        )
    )

    # --------------------------------------
    # 3. 广义局部密度归一化
    # --------------------------------------
    rho_raw = rho_agent + wall_weight * rho_wall

    rho_norm = rho_raw / (
        np.max(rho_raw) + 1e-8
    )

    # --------------------------------------
    # 4. 使用真实网格间距计算梯度
    # --------------------------------------
    dx = float(XQ[0, 1] - XQ[0, 0])
    dy = float(YQ[1, 0] - YQ[0, 0])

    grad_y, grad_x = np.gradient(
        rho_norm,
        dy,
        dx
    )

    # --------------------------------------
    # 5. 新的密度归一化扩散速度
    # --------------------------------------
    diffusion_gain = kd / (
        rho_norm + eps_rho
    )

    u_diff = -diffusion_gain * grad_x
    v_diff = -diffusion_gain * grad_y

    # --------------------------------------
    # 6. 屏蔽墙壁外部区域
    # --------------------------------------
    inside_mask = (
        np.abs(YQ) <= half_width
    )

    u_diff = np.where(
        inside_mask, u_diff, 0.0
    )
    v_diff = np.where(
        inside_mask, v_diff, 0.0
    )

    speed_diff = np.sqrt(
        u_diff**2 + v_diff**2
    )

    return (
        u_diff,
        v_diff,
        speed_diff,
        rho_norm
    )


overlay_rect = plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, color='black', alpha=0.85, zorder=100)
fig.patches.append(overlay_rect)

overlay_text = fig.text(0.5, 0.5, '', color='white', fontsize=22, fontweight='bold', ha='center', va='center', zorder=101)

def update(frame_info):
    mode = frame_info['mode']
    d_idx = frame_info['d_idx']
    step = frame_info['step']
    pause_msg = frame_info['pause_msg']
    data = all_datasets[d_idx]
    is_ext = data['is_extended']

    if mode == 'intro':
        overlay_rect.set_visible(True)
        overlay_text.set_visible(True)
        overlay_text.set_text("Density-field evolution and physical visualization\n \n \n Of Formation-Reconfigurable Swarm Navigation")
    elif mode == 'transition':
        overlay_rect.set_visible(True)
        overlay_text.set_visible(True)
        overlay_text.set_text(data['title'])
        explanation_box.set_visible(False)
    else:
        overlay_rect.set_visible(False)
        overlay_text.set_visible(False)

    if pause_msg:
        explanation_box.set_text(pause_msg)
        explanation_box.set_visible(True)
    else:
        explanation_box.set_visible(False)

    update_static_geometry(d_idx)
    
    limit_steps_curr = datasets_config[d_idx]['max_steps']
    ax_map.set_title(f"Density-field visualization", fontsize=14, fontweight='bold')
    ax_param.set_xlim(0, limit_steps_curr)

    show_failures = data['has_failures']
    fail_line_p.set_visible(show_failures)

    curr_df = data['df'][data['df']['step'] == step]
    if curr_df.empty: 
        return [scat_alive, scat_dead, scat_dead_cross, line_w_norm, line_w_shape_norm, line_k_diff, line_lambda] + trail_lines_base

    for coll in list(ax_map.collections):
        if coll.get_zorder() in [1, 4, 5]: coll.remove()
    for patch in list(ax_map.patches):
        if patch.get_zorder() in [4, 5]: patch.remove()

    alive_mask = curr_df['is_alive'] == 1 if 'is_alive' in curr_df.columns else np.ones(len(curr_df), dtype=bool)
    dead_mask = ~alive_mask
    
    if alive_mask.any():
        alive_df = curr_df[alive_mask]
        cx, cy = alive_df['pos_x'].mean(), alive_df['pos_y'].mean()
        
        dists = (alive_df['pos_x'] - cx)**2 + (alive_df['pos_y'] - cy)**2
        focus_id = alive_df.loc[dists.idxmin(), 'agent_id']
        curr_leader = alive_df[alive_df['agent_id'] == focus_id].iloc[0]
        
        ws, wf, bt, kd = curr_leader['w_shape'], curr_leader['w_flow'], curr_leader['beta'], curr_leader['k_diff']
        XV, YV, XQ, YQ, XQV, YQV = get_grid_mesh(is_ext)
        
        dx, dy = XV - cx, YV - cy
        dw = np.sqrt(dx**2 + bt * dy**2) + 1e-6
        u_adv = -ws * ((dw - R0)/dw) * dx + wf
        v_adv = -ws * ((dw - R0)/dw) * (bt * dy)
        pot = np.exp(-15 * ws * (dw - R0)**2)
        
        ax_map.contourf(XV, YV, pot, levels=12, cmap='Reds', alpha=0.35, zorder=1)
        ax_map.streamplot(XV, YV, u_adv, v_adv, color='r', linewidth=0.6, density=1.2, zorder=4, arrowsize=0.8)
        
        field_agent_pos = curr_df[
            ['pos_x', 'pos_y']
        ].values

        u_d, v_d, speed_d, rho_norm = (
            get_diffusion_field_vectorized(
                kd=kd,
                agent_pos=field_agent_pos,
                is_ext=is_ext,
                XQ=XQ,
                YQ=YQ,
                eps_rho=0.05
            )
        )

        # 降低箭头密度，避免视频画面过于拥挤
        skip = (
            slice(None, None, 2),
            slice(None, None, 2)
        )

        X_plot = XQ[skip]
        Y_plot = YQ[skip]
        u_plot = u_d[skip]
        v_plot = v_d[skip]

        mag_plot = np.sqrt(u_plot**2 + v_plot**2)
        positive_mag = mag_plot[mag_plot > 1e-10]

        if positive_mag.size > 0:
            mag_ref = np.percentile(positive_mag, 90.0)

            valid = mag_plot > 0.01 * mag_ref

            u_unit = np.divide(
                u_plot,
                mag_plot,
                out=np.zeros_like(u_plot),
                where=mag_plot > 1e-10
            )

            v_unit = np.divide(
                v_plot,
                mag_plot,
                out=np.zeros_like(v_plot),
                where=mag_plot > 1e-10
            )

            relative_mag = np.clip(
                mag_plot / (mag_ref + 1e-8),
                0.25,
                1.0
            )

            arrow_length = 0.11 if is_ext else 0.055

            u_show = arrow_length * relative_mag * u_unit
            v_show = arrow_length * relative_mag * v_unit

            ax_map.quiver(
                X_plot[valid],
                Y_plot[valid],
                u_show[valid],
                v_show[valid],
                color='#2457C5',
                angles='xy',
                scale_units='xy',
                scale=1.0,
                width=0.0035,
                headwidth=3.8,
                headlength=4.8,
                headaxislength=4.2,
                alpha=0.72,
                zorder=5
            )

    scat_alive.set_offsets(curr_df[alive_mask][['pos_x', 'pos_y']] if alive_mask.any() else np.empty((0, 2)))
    dead_coords = curr_df[dead_mask][['pos_x', 'pos_y']] if dead_mask.any() else np.empty((0, 2))
    scat_dead.set_offsets(dead_coords)
    scat_dead_cross.set_offsets(dead_coords)

    num_agents_curr = data['df'].shape[0] // len(data['df']['step'].unique())
    for i in range(max_agents_global):
        if i < num_agents_curr:
            idx = np.searchsorted(data['paths'][i]['steps'], step, side='right')
            hist_x = data['paths'][i]['x'][:idx]
            hist_y = data['paths'][i]['y'][:idx]
            
            trail_lines_base[i].set_data(hist_x, hist_y)
            
            # 区分存活与失效节点的轨迹透明度
            if idx > 0 and data['paths'][i]['alive'][idx-1] == 0:
                trail_lines_base[i].set_alpha(0.08)
                trail_lines_base[i].set_color('#BDC3C7') 
            else:
                trail_lines_base[i].set_alpha(0.35)
                trail_lines_base[i].set_color('#95A5A6')
        else:
            trail_lines_base[i].set_data([], [])

    hist_macro = data['macro'][data['macro']['step'] <= step]
    line_w_norm.set_data(hist_macro['step'], hist_macro['w_flow_smooth'])
    line_w_shape_norm.set_data(hist_macro['step'], hist_macro['w_shape_smooth'])
    line_lambda.set_data(hist_macro['step'], hist_macro['lambda_smooth'])
    line_k_diff.set_data(hist_macro['step'], hist_macro['k_diff_smooth'])

    return [scat_alive, scat_dead, scat_dead_cross, line_w_norm, line_w_shape_norm, line_k_diff, line_lambda] + trail_lines_base

print("Running dynamically balanced ADR rendering engine (No E_ADR)...")
ani = animation.FuncAnimation(fig, update, frames=frames_schedule, blit=False, interval=30)
# plt.show()
out_dir = 'unified_videos'
os.makedirs(out_dir, exist_ok=True)
out_name = os.path.join(out_dir, 'formation_field.mp4')

print(f"Saving compiled video to {out_name} ...")
ani.save(out_name, 
         writer='ffmpeg', 
         fps=30,       
         dpi=150,      
         bitrate=-1,   
         extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-preset', 'fast'])
print("Masterpiece Unified Video generated successfully!")
