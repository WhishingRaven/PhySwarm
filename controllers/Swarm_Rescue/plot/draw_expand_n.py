import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.collections import LineCollection 
import glob
import re
import os   
import random
from collections import deque

file_paths = sorted(glob.glob('data/expand_*.csv'), key=lambda x: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', x)])
if not file_paths:
    print("Error: No expand_*.csv files found in data/ folder.")
    exit()

TRANSITION_FRAMES = 60 # 2秒转场(30fps)
INTRO_FRAMES = 120    # 4秒开场介绍(30fps)
SIM_FRAME_STEP = 5     # 播放加速步长
PAUSE_DURATION = 120  

COMM_RANGE_M = 0.60  
base_x, base_y = -1.2, 0.0

PARAM_SMOOTH_WINDOW = 21

def smooth_parameter_by_role(df_agent, column, window=21):
    values = df_agent[column].astype(float).copy()
    if 'role' not in df_agent.columns:
        return values.rolling(window=window, center=True, min_periods=1).mean()

    role_block = df_agent['role'].ne(df_agent['role'].shift()).cumsum()
    return df_agent.groupby(role_block, group_keys=False)[column].transform(lambda x: x.astype(float).rolling(window=window, center=True, min_periods=1).mean())


def check_and_get_chain_segments(agents_df, base_pos, target_pos, target_idx, comm_range, target_map_col='assigned_target'):
    required_cols = {'agent_id', 'pos_x', 'pos_y', 'role', 'is_alive', target_map_col}
    missing_cols = required_cols - set(agents_df.columns)

    if missing_cols:
        raise KeyError(f"CSV 缺少列：{sorted(missing_cols)}")

    chain_agents = agents_df[(agents_df['is_alive'] == 1) & (agents_df['role'].isin([1, 2])) & (agents_df[target_map_col] == target_idx)].copy()

    if chain_agents.empty:
        return []

    nodes = []
    id_to_position = {}

    for _, row in chain_agents.iterrows():
        agent_id = int(row['agent_id'])
        position = np.array([float(row['pos_x']), float(row['pos_y'])], dtype=np.float32)
        role = int(row['role'])
        nodes.append({'id': agent_id, 'position': position, 'role': role})
        id_to_position[agent_id] = position

    base_id = 'Base'
    base_position = np.array(base_pos, dtype=np.float32)
    adjacency = {base_id: []}

    for node in nodes:
        adjacency[node['id']] = []

    for node in nodes:
        distance_to_base = np.linalg.norm(node['position'] - base_position)

        if distance_to_base < comm_range:
            adjacency[base_id].append(node['id'])
            adjacency[node['id']].append(base_id)

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            distance = np.linalg.norm(nodes[i]['position'] - nodes[j]['position'])

            if distance < comm_range:
                adjacency[nodes[i]['id']].append(nodes[j]['id'])
                adjacency[nodes[j]['id']].append(nodes[i]['id'])

    target_position = np.array(target_pos, dtype=np.float32)
    terminal_responders = set()

    for node in nodes:
        if node['role'] != 1:
            continue

        distance_to_target = np.linalg.norm(node['position'] - target_position)

        if distance_to_target < comm_range:
            terminal_responders.add(node['id'])

    if not terminal_responders:
        return []

    queue = deque([(base_id, [base_id])])
    visited = {base_id}
    successful_path = None

    while queue:
        current_id, current_path = queue.popleft()

        if current_id in terminal_responders:
            successful_path = current_path
            break

        for neighbor_id in adjacency[current_id]:
            if neighbor_id in visited:
                continue

            visited.add(neighbor_id)
            queue.append((neighbor_id, current_path + [neighbor_id]))

    if successful_path is None:
        return []

    coordinate_path = [base_position]

    for node_id in successful_path[1:]:
        coordinate_path.append(id_to_position[node_id])

    coordinate_path.append(target_position)

    segments = []

    for i in range(len(coordinate_path) - 1):
        segments.append([coordinate_path[i], coordinate_path[i + 1]])

    return segments

all_datasets = []
max_agents_global = 0  
max_targets_global = 0

for idx, f_path in enumerate(file_paths):
    print(f"[{idx+1}/{len(file_paths)}] Loading and pre-computing {f_path} ...")
    df = pd.read_csv(f_path)
    
    total_steps = df['step'].max()
    num_agents = df['agent_id'].nunique()
    max_agents_global = max(max_agents_global, num_agents)
    
    target_cols = [c for c in df.columns if c.startswith('target_') and c.endswith('_x')]
    num_targets = len(target_cols)
    max_targets_global = max(max_targets_global, num_targets)
    
    selected_agent_id = int(random.choice(df['agent_id'].unique()))
    df_agent = df[df['agent_id'] == selected_agent_id].sort_values('step').reset_index(drop=True)
    
    df_agent['w_sum'] = df_agent['w_rand'] + df_agent['w_target'] + df_agent['w_center'] + 1e-8
    df_agent['w_target_norm'] = df_agent['w_target'] / df_agent['w_sum']
    df_agent['w_center_norm'] = df_agent['w_center'] / df_agent['w_sum']
    df_agent['w_rand_norm'] = df_agent['w_rand'] / df_agent['w_sum']

    df_agent['w_target_smooth'] = smooth_parameter_by_role(df_agent, 'w_target_norm', PARAM_SMOOTH_WINDOW)
    df_agent['w_center_smooth'] = smooth_parameter_by_role(df_agent, 'w_center_norm', PARAM_SMOOTH_WINDOW)
    df_agent['w_rand_smooth'] = smooth_parameter_by_role(df_agent, 'w_rand_norm', PARAM_SMOOTH_WINDOW)
    df_agent['lambda_anchor_smooth'] = smooth_parameter_by_role(df_agent, 'lambda_anchor', PARAM_SMOOTH_WINDOW)
    df_agent['lambda_release_smooth'] = smooth_parameter_by_role(df_agent, 'lambda_release', PARAM_SMOOTH_WINDOW)
    df_agent['k_diff_smooth'] = smooth_parameter_by_role(df_agent, 'k_diff', PARAM_SMOOTH_WINDOW)

    smooth_w_sum = df_agent['w_target_smooth'] + df_agent['w_center_smooth'] + df_agent['w_rand_smooth'] + 1e-8
    df_agent['w_target_smooth'] = df_agent['w_target_smooth'] / smooth_w_sum
    df_agent['w_center_smooth'] = df_agent['w_center_smooth'] / smooth_w_sum
    df_agent['w_rand_smooth'] = df_agent['w_rand_smooth'] / smooth_w_sum

    df_agent['lambda_anchor_smooth'] = np.clip(df_agent['lambda_anchor_smooth'], 0.0, 1.0)
    df_agent['lambda_release_smooth'] = np.clip(df_agent['lambda_release_smooth'], 0.0, 1.0)
    df_agent['k_diff_smooth'] = np.clip(df_agent['k_diff_smooth'], 0.0, None)
    
    match = re.search(r'expand_(\d+)', f_path)
    agent_count_str = match.group(1) if match else str(num_agents)
    transition_title = f"Scene {idx + 1} \n \n Swarm Size: {agent_count_str} Robots"
    
    agent_paths = {}
    for i in range(num_agents):
        sub_df = df[df['agent_id'] == i].sort_values('step')
        agent_paths[i] = {
            'x': sub_df['pos_x'].values,
            'y': sub_df['pos_y'].values,
            'steps': sub_df['step'].values,
            'role': sub_df['role'].values if 'role' in sub_df.columns else np.zeros(len(sub_df)), 
            'is_alive': sub_df['is_alive'].values if 'is_alive' in sub_df.columns else np.ones(len(sub_df))
        }
        
    all_datasets.append({
        'df': df, 
        'paths': agent_paths, 
        'df_agent': df_agent, 
        'selected_agent_id': selected_agent_id,
        'total_steps': total_steps, 
        'num_agents': num_agents,
        'num_targets': num_targets,
        'title': transition_title
    })

print(f"Pre-computations finished! Max agents: {max_agents_global} | Max targets: {max_targets_global}")

frames_schedule = []
for _ in range(INTRO_FRAMES):
    frames_schedule.append({'mode': 'intro', 'd_idx': 0, 'step': 1, 'pause_msg': None})

# 设定暂停事件的关键帧
step_explain_simultaneously = 750  # Scene 1 & 2 解释不能同时解救目标的帧
step_explain_wandering = 750       # Scene 3 解释右上角机器人探索的帧

for d_idx, data in enumerate(all_datasets):
    for _ in range(TRANSITION_FRAMES):
        frames_schedule.append({'mode': 'transition', 'd_idx': d_idx, 'step': 1, 'pause_msg': None})
    
    triggered_scene_explain = False
    triggered_wander_explain = False
    
    step = 1
    while step <= data['total_steps']:
        pause_msg = None
        trigger_pause = False
        
        if d_idx in [0, 1, 2]:
            if step >= step_explain_simultaneously and not triggered_scene_explain:
                trigger_pause = True
                triggered_scene_explain = True
                pause_msg = "Due to constraints on physical communication range and the limitednumber of robots, \nthe swarm cannot connect to and rescue all three targets simultaneously."
        
        elif d_idx == 3:
            if step >= step_explain_wandering and not triggered_wander_explain:
                trigger_pause = True
                triggered_wander_explain = True
                pause_msg = "Some robots in the upper-right corner continue to wander. This is \nbecause they are guided by a localized random exploration field and have not yet\nlocalized the target signals due to communication range limits."
        
        if trigger_pause:
            for _ in range(PAUSE_DURATION):
                frames_schedule.append({'mode': 'sim', 'd_idx': d_idx, 'step': step, 'pause_msg': pause_msg})
                
        frames_schedule.append({'mode': 'sim', 'd_idx': d_idx, 'step': step, 'pause_msg': None})
        step += SIM_FRAME_STEP


fig = plt.figure(figsize=(14, 11), dpi=120)
gs = gridspec.GridSpec(2, 1, left=0.15, right=0.85, top=0.92, bottom=0.06, hspace=0.32, height_ratios=[2.0, 1.0])

ax_map = fig.add_subplot(gs[0, 0]) 
ax_param = fig.add_subplot(gs[1, 0])  

ax_map.set_xlim(-3.1, 3.1) 
ax_map.set_ylim(-1.6, 1.6)
ax_map.set_aspect('equal')
ax_map.set_facecolor('#f4f4f4') 
ax_map.set_xlabel("x position (m)", fontweight='bold', fontsize=12)
ax_map.set_ylabel("y position (m)", fontweight='bold', fontsize=12)

# 绘制基地
ax_map.scatter(base_x, base_y, s=1200, c='green', alpha=0.4, edgecolors='green', lw=1, zorder=1)
scat_base = ax_map.scatter([],[], s=80, c='green', edgecolors='green', lw=1, zorder=10)

# 初始化中继通信链路层
relay_links_col = LineCollection([], colors='#FFD700', linewidths=2.0, linestyles='--', zorder=16)
ax_map.add_collection(relay_links_col)

# 初始化目标圆形
target_circles = []
for t_idx in range(max_targets_global):
    circle = patches.Circle((0, 0), 0.12, color='red', alpha=0.6, ec='darkred', lw=2, zorder=2)
    ax_map.add_patch(circle)
    target_circles.append(circle)

# 初始化轨迹笔刷
trail_lines_search = []
trail_lines_respond = [] 
trail_lines_relay = []

for i in range(max_agents_global):
    line_s, = ax_map.plot([],[], lw=1.2, alpha=0.4, color='gray', zorder=3)
    trail_lines_search.append(line_s)
    line_resp, = ax_map.plot([],[], lw=2.5, color='crimson', alpha=0.8, zorder=4)
    trail_lines_respond.append(line_resp)
    line_rel, = ax_map.plot([],[], lw=3.0, color='purple', alpha=0.6, zorder=5)
    trail_lines_relay.append(line_rel)

agents_scat = ax_map.scatter([],[], s=70, edgecolors='k', linewidths=1.5, zorder=30)
scat_tracked = ax_map.scatter([],[], s=200, facecolors='none', edgecolors='gold', lw=3, zorder=50)
scat_dead = ax_map.scatter([],[], s=70, c='black', alpha=0.3, edgecolors='k', linewidths=1.5, zorder=11)
scat_dead_cross = ax_map.scatter([],[], s=30, color='white', marker='x', linewidths=1.5, zorder=12)

explanation_box = ax_map.text(0.04, 0.05, "", transform=ax_map.transAxes,
                              fontsize=11.5, fontweight='bold', color='black', family='sans-serif',
                              bbox=dict(boxstyle="round,pad=0.8", facecolor='#FFFEE6', alpha=0.92, edgecolor='darkorange', lw=2),
                              zorder=60, visible=False)

# 参数演化图设置
ax_param.set_xlabel("Simulation Steps", fontweight='bold')
ax_param.set_ylim(-0.05, 1.1) 
ax_param.set_ylabel(r"Coefficient ($\tilde{\omega}, \lambda$)", fontweight='bold', fontsize=11)

line_w_rand_norm, = ax_param.plot([],[], color='#4d4dfb', lw=2.5, alpha=0.4, label=r'$\tilde{\omega}_{rand}$')
line_w_center_norm, = ax_param.plot([],[], color='#8000ff', lw=2.5, label=r'$\tilde{\omega}_{center}$')
line_w_target_norm, = ax_param.plot([],[], color='#16c41c', lw=2, label=r'$\tilde{\omega}_{target}$')
line_l_anchor, = ax_param.plot([],[], color='#1697f3', lw=2.5, label=r'$\lambda_{anchor}$')
line_l_release, = ax_param.plot([],[], color='#4b6cff', lw=2.5, label=r'$\lambda_{release}$')
ax_param.grid(True, alpha=0.3)

ax_param_twin = ax_param.twinx()
ax_param_twin.set_ylim(-0.01, 0.04) 
ax_param_twin.set_ylabel(r"Coefficient ($D$)",  fontweight='bold', fontsize=11, labelpad=15)
ax_param_twin.tick_params(axis='y')
ax_param_twin.grid(False) 

line_k_diff, = ax_param_twin.plot([],[], color='#ff9a51', lw=2.5, label=r'$D$')

lines_p = [line_w_target_norm, line_w_center_norm, line_w_rand_norm, line_k_diff, line_l_anchor, line_l_release]
ax_param.legend(lines_p, [l.get_label() for l in lines_p], 
                loc='lower center', bbox_to_anchor=(0.5, 1.02), 
                ncol=6, framealpha=0.9, fontsize=9)

import matplotlib.lines as mlines
legend_elements = [
    mlines.Line2D([], [], color='dodgerblue', marker='o', linestyle='None', markersize=8, markeredgecolor='k', label='Searcher'),
    mlines.Line2D([], [], color='purple', marker='o', linestyle='None', markersize=8, markeredgecolor='k', label='Relay'),
    mlines.Line2D([], [], color='crimson', marker='o', linestyle='None', markersize=8, markeredgecolor='k', label='Responder')
]
ax_map.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=10.5)

overlay_rect = plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, color='black', alpha=0.85, zorder=100)
fig.patches.append(overlay_rect)
overlay_text = fig.text(0.5, 0.5, '', color='white', fontsize=24, fontweight='bold', ha='center', va='center', zorder=101)

def update(frame_info):
    mode = frame_info['mode']
    d_idx = frame_info['d_idx']
    step = frame_info['step']
    pause_msg = frame_info['pause_msg']
    data = all_datasets[d_idx]

    if mode == 'intro':
        overlay_rect.set_visible(True)
        overlay_text.set_visible(True)
        overlay_text.set_text("Scalability verification in simulation\n \n \n For Role-Adaptive Swarm Search and Rescue")
        explanation_box.set_visible(False)
    
    elif mode == 'transition':
        overlay_rect.set_visible(True)
        overlay_text.set_visible(True)
        overlay_text.set_text(data['title'])
        explanation_box.set_visible(False)
    else:
        overlay_rect.set_visible(False)
        overlay_text.set_visible(False)

    if mode == 'sim' and pause_msg:
        explanation_box.set_text(pause_msg)
        explanation_box.set_visible(True)
    else:
        explanation_box.set_visible(False)

    ax_map.set_title(f"Scalability Verification", fontsize=16, fontweight='bold', pad=15)
    ax_param.set_xlim(0, data['total_steps'])
    
    current_data = data['df'][data['df']['step'] == step]
    if current_data.empty: 
        return (agents_scat, line_w_target_norm, line_w_center_norm, line_w_rand_norm, line_k_diff, 
                line_l_anchor, line_l_release, relay_links_col, *trail_lines_search, *trail_lines_respond, *trail_lines_relay)

    if 'is_alive' in current_data.columns:
        alive = current_data[current_data['is_alive'] == 1]
        dead = current_data[current_data['is_alive'] == 0]
    else:
        alive = current_data
        dead = pd.DataFrame()
    
    # 1. 动态更新目标的坐标并计算中继线段
    segments = []
    target_map_col = 'assigned_target' if 'assigned_target' in current_data.columns else 'agent_target_map'

    for t_idx, circle in enumerate(target_circles):
        if t_idx >= data['num_targets']:
            circle.set_visible(False)
            continue

        circle.set_visible(True)
        tx = float(current_data[f'target_{t_idx}_x'].iloc[0])
        ty = float(current_data[f'target_{t_idx}_y'].iloc[0])
        circle.center = (tx, ty)

        seg = check_and_get_chain_segments(current_data, (base_x, base_y), (tx, ty), t_idx, COMM_RANGE_M, target_map_col)
        segments.extend(seg)

    relay_links_col.set_segments(segments)

    # 2. 动态更新分层轨迹笔刷 
    for i in range(max_agents_global):
        if i < data['num_agents']:
            current_idx = np.searchsorted(data['paths'][i]['steps'], step, side='right')
            if current_idx < 2: continue
            
            hist_x = data['paths'][i]['x'][:current_idx].astype(float).copy()
            hist_y = data['paths'][i]['y'][:current_idx].astype(float).copy()
            hist_role = data['paths'][i]['role'][:current_idx]
            is_alive = data['paths'][i]['is_alive'][current_idx-1]
            
            # Searcher 轨迹
            trail_lines_search[i].set_data(hist_x, hist_y)
            
            # Responder 轨迹
            resp_x, resp_y = hist_x.copy(), hist_y.copy()
            resp_x[hist_role != 1] = np.nan
            resp_y[hist_role != 1] = np.nan
            trail_lines_respond[i].set_data(resp_x, resp_y)

            # Relay 轨迹
            relay_x, relay_y = hist_x.copy(), hist_y.copy()
            relay_x[hist_role != 2] = np.nan
            relay_y[hist_role != 2] = np.nan
            trail_lines_relay[i].set_data(relay_x, relay_y)
            
            alpha_val = 0.4 if is_alive else 0.05
            trail_lines_search[i].set_alpha(alpha_val)
            trail_lines_respond[i].set_alpha(alpha_val * 2)
            trail_lines_relay[i].set_alpha(alpha_val * 1.5)
        else:
            trail_lines_search[i].set_data([], [])
            trail_lines_respond[i].set_data([], [])
            trail_lines_relay[i].set_data([], [])

    # 3. 动态更新智能体散点与位置颜色
    if not alive.empty:
        positions = alive[['pos_x', 'pos_y']].values
        roles = alive['role'].values
        face_colors = np.where(roles == 1, 'crimson', np.where(roles == 2, 'purple', 'dodgerblue'))
                               
        agents_scat.set_offsets(positions)
        agents_scat.set_facecolors(face_colors)
        
        # 追踪标志
        tracked_agent_id = data['selected_agent_id']
        tracked_agent_data = alive[alive['agent_id'] == tracked_agent_id]
        if not tracked_agent_data.empty:
            scat_tracked.set_offsets(tracked_agent_data[['pos_x', 'pos_y']].values)
        else:
            scat_tracked.set_offsets(np.empty((0, 2)))
    else:
        agents_scat.set_offsets(np.empty((0,2)))
        scat_tracked.set_offsets(np.empty((0, 2)))
        
    if not dead.empty:
        dead_pos = dead[['pos_x', 'pos_y']].values
        scat_dead.set_offsets(dead_pos)
        scat_dead_cross.set_offsets(dead_pos)

    # 4. 更新下方子图
    df_agent = data['df_agent']
    hist_agent = df_agent[df_agent['step'] <= step]
    
    line_w_target_norm.set_data(hist_agent['step'], hist_agent['w_target_smooth'])
    line_w_center_norm.set_data(hist_agent['step'], hist_agent['w_center_smooth'])
    line_w_rand_norm.set_data(hist_agent['step'], hist_agent['w_rand_smooth'])
    line_l_anchor.set_data(hist_agent['step'], hist_agent['lambda_anchor_smooth'])
    line_l_release.set_data(hist_agent['step'], hist_agent['lambda_release_smooth'])
    line_k_diff.set_data(hist_agent['step'], hist_agent['k_diff_smooth'])

    return (agents_scat, line_w_target_norm, line_w_center_norm, line_w_rand_norm, line_k_diff, 
            line_l_anchor, line_l_release, relay_links_col, *trail_lines_search, *trail_lines_respond, *trail_lines_relay)

print("Starting video rendering engine...")
ani = animation.FuncAnimation(fig, update, frames=frames_schedule, blit=False, interval=30)
plt.show()
out_name = 'expand_video/rescue_compilation_all.mp4'
os.makedirs('expand_video', exist_ok=True)
print(f"Saving compilation video to {out_name} ...")
ani.save(out_name, 
         writer='ffmpeg', 
         fps=30,       
         dpi=150,      
         bitrate=-1,   
         extra_args=[
             '-vcodec', 'libx264', 
             '-pix_fmt', 'yuv420p',
             '-crf', '15',       
             '-preset', 'fast' 
         ])
print(f"Masterpiece Rendering Done! Check the folder: {out_name}")