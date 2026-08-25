import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors 
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import collections  
import random

datasets_config = [
    {
        'file_path': 'data/fails_0_1.csv',
        'title': "Scene 1: Standard Swarm\n \n8 Benign Robots",
        'is_extended': False,
        'has_failures': False,
        'focus_agent_id': 4,  
    },
    {
        'file_path': 'data/fails_resp_rel_2.csv',
        'title': "Scene 2: Fault-Tolerant Swarm\n \n8 Robots (2 Failed Robots)",
        'is_extended': False,
        'has_failures': True,
        'focus_agent_id': 6,   
    },
    {
        'file_path': 'data/expand_24.csv',
        'title': "Scene 3: Large-Scale Swarm\n \n24 Benign Robots",
        'is_extended': True,
        'has_failures': False,
        'focus_agent_id': 12, 
    }
]

TRANSITION_FRAMES = 60 
FAIL_STEP = 300 
INTRO_FRAMES = 120   
PAUSE_DURATION = 120  
R0 = 0.12             
COMM_RANGE_M = 0.6    # 物理通信半径

PARAM_SMOOTH_WINDOW = 21

def smooth_parameter_by_role(df_agent, column, window=21):
    values = df_agent[column].astype(float).copy()
    if 'role' not in df_agent.columns:
        return values.rolling(window=window, center=True, min_periods=1).mean()

    role_block = df_agent['role'].ne(df_agent['role'].shift()).cumsum()
    return df_agent.groupby(role_block, group_keys=False)[column].transform(lambda x: x.astype(float).rolling(window=window, center=True, min_periods=1).mean())


def get_subjective_potential_field(X_grid, Y_grid, w_target, w_center, w_rand, active_targets, base_pos):
    """计算救援任务主观吸引势场梯度矢量与势能 (对流项)"""
    u_adv = np.zeros_like(X_grid)
    v_adv = np.zeros_like(Y_grid)
    potential = np.zeros_like(X_grid)
    sigma_sq = 0.25 
    
    if w_target > 0.01 and len(active_targets) > 0:
        for tx, ty in active_targets:
            dist_sq = (X_grid - tx)**2 + (Y_grid - ty)**2
            pot_t = w_target * np.exp(-dist_sq / sigma_sq)
            potential += pot_t
            u_adv += pot_t * (-2 * (X_grid - tx) / sigma_sq)
            v_adv += pot_t * (-2 * (Y_grid - ty) / sigma_sq)

    if w_center > 0.01 and len(active_targets) > 0:
        tx, ty = active_targets[0] 
        bx, by = base_pos
        
        line_vec = np.array([tx - bx, ty - by])
        line_len_sq = line_vec[0]**2 + line_vec[1]**2 + 1e-8
        
        dx = X_grid - bx
        dy = Y_grid - by
        t = (dx * line_vec[0] + dy * line_vec[1]) / line_len_sq
        t_clamped = np.clip(t, 0.0, 1.0)
        
        closest_x = bx + t_clamped * line_vec[0]
        closest_y = by + t_clamped * line_vec[1]
        
        dist_to_line_sq = (X_grid - closest_x)**2 + (Y_grid - closest_y)**2
        
        pot_c = w_center * np.exp(-dist_to_line_sq / sigma_sq)
        potential += pot_c
        u_adv += pot_c * (-2 * (X_grid - closest_x) / sigma_sq)
        v_adv += pot_c * (-2 * (Y_grid - closest_y) / sigma_sq)

    u_adv += 1e-7; v_adv += 1e-7
    return u_adv, v_adv, potential

def get_density_normalized_diffusion_field(
    X,
    Y,
    kd,
    agent_pos,
    focus_pos,
    wall_y_limit,
    eps_rho=0.05,
    sigma_agent=0.060,
    sigma_wall=0.050,
    wall_weight=1.0
):

    # ======================================
    # 1. 机器人连续密度场
    # ======================================
    rho_agent = np.zeros_like(X, dtype=float)

    for agent_x, agent_y in agent_pos:
        # 对被追踪机器人而言，局部邻居密度通常不包括自身
        is_focus = (
            abs(agent_x - focus_pos[0]) < 1e-3
            and abs(agent_y - focus_pos[1]) < 1e-3
        )

        if is_focus:
            continue

        dist_sq = (
            (X - agent_x)**2
            + (Y - agent_y)**2
        )

        rho_agent += np.exp(
            -dist_sq / (2.0 * sigma_agent**2)
        )

    # ======================================
    # 2. 上下墙壁连续密度场
    # ======================================
    dist_top = np.abs(wall_y_limit - Y)
    dist_bottom = np.abs(Y + wall_y_limit)

    rho_wall = (
        np.exp(
            -dist_top**2
            / (2.0 * sigma_wall**2)
        )
        +
        np.exp(
            -dist_bottom**2
            / (2.0 * sigma_wall**2)
        )
    )

    # ======================================
    # 3. 广义局部密度归一化
    # ======================================
    rho_raw = rho_agent + wall_weight * rho_wall

    rho_norm = rho_raw / (
        np.max(rho_raw) + 1e-8
    )

    # ======================================
    # 4. 使用真实空间网格间距计算梯度
    # ======================================
    dx = float(X[0, 1] - X[0, 0])
    dy = float(Y[1, 0] - Y[0, 0])

    grad_y, grad_x = np.gradient(
        rho_norm,
        dy,
        dx
    )

    # ======================================
    # 5. 新扩散速度定义
    # ======================================
    diffusion_gain = kd / (
        rho_norm + eps_rho
    )

    u_diff = -diffusion_gain * grad_x
    v_diff = -diffusion_gain * grad_y

    # ======================================
    # 6. 清除墙外区域
    # ======================================
    inside_mask = (
        np.abs(Y) <= wall_y_limit
    )

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
        u_diff**2 + v_diff**2
    )

    return (
        u_diff,
        v_diff,
        speed_diff,
        rho_norm
    )


def point_to_segment_distance(point, seg_start, seg_end):
    point = np.asarray(point, dtype=float)
    seg_start = np.asarray(seg_start, dtype=float)
    seg_end = np.asarray(seg_end, dtype=float)

    segment = seg_end - seg_start
    segment_sq = np.dot(segment, segment)

    if segment_sq < 1e-12:
        return np.linalg.norm(point - seg_start)

    t = np.dot(point - seg_start, segment) / segment_sq
    t = np.clip(t, 0.0, 1.0)

    projection = seg_start + t * segment
    return np.linalg.norm(point - projection)


def find_connected_chains(curr_df, base_pos, target_positions, comm_view=0.6, catch_distance=0.15, target_map_col=None, require_all_responders=True):
    required_cols = {'agent_id', 'pos_x', 'pos_y', 'role'}
    missing_cols = required_cols - set(curr_df.columns)

    if missing_cols:
        raise KeyError(f"CSV 缺少必要列：{sorted(missing_cols)}")

    if 'is_alive' in curr_df.columns:
        routing_df = curr_df[(curr_df['is_alive'] == 1) & (curr_df['role'].isin([1, 2]))].copy()
    else:
        routing_df = curr_df[curr_df['role'].isin([1, 2])].copy()

    if target_map_col is None:
        if 'assigned_target' in curr_df.columns:
            target_map_col = 'assigned_target'
        elif 'agent_target_map' in curr_df.columns:
            target_map_col = 'agent_target_map'

    multi_target = len(target_positions) > 1

    if multi_target and target_map_col is None:
        raise KeyError("多目标数据必须保存 assigned_target 或 agent_target_map，否则无法构建目标专属通信链。")

    all_chains = []
    base_id = 'Base'
    base_position = np.asarray(base_pos, dtype=float)

    for t_idx, target_pos in enumerate(target_positions):
        target_position = np.asarray(target_pos, dtype=float)

        if target_map_col is not None:
            target_agents = routing_df[routing_df[target_map_col] == t_idx].copy()
        else:
            target_agents = routing_df.copy()

        if target_agents.empty:
            continue

        responders_df = target_agents[target_agents['role'] == 1].copy()

        if responders_df.empty:
            continue

        responder_distances = np.linalg.norm(responders_df[['pos_x', 'pos_y']].values - target_position, axis=1)
        near_responders_df = responders_df[(responder_distances < catch_distance) & (responder_distances < comm_view)].copy()

        if len(near_responders_df) < 3:
            continue

        nodes = [{'id': base_id, 'pos': base_position}]
        id_to_pos = {base_id: base_position}

        for _, row in target_agents.iterrows():
            agent_id = int(row['agent_id'])
            agent_position = np.array([float(row['pos_x']), float(row['pos_y'])], dtype=float)
            nodes.append({'id': agent_id, 'pos': agent_position})
            id_to_pos[agent_id] = agent_position

        adjacency = collections.defaultdict(list)

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i = nodes[i]
                node_j = nodes[j]
                distance = np.linalg.norm(node_i['pos'] - node_j['pos'])

                if distance < comm_view:
                    adjacency[node_i['id']].append(node_j['id'])
                    adjacency[node_j['id']].append(node_i['id'])

        queue = collections.deque([base_id])
        visited = {base_id}
        parent = {base_id: None}

        while queue:
            current_id = queue.popleft()

            for neighbor_id in adjacency[current_id]:
                if neighbor_id in visited:
                    continue

                visited.add(neighbor_id)
                parent[neighbor_id] = current_id
                queue.append(neighbor_id)

        near_responder_ids = [int(agent_id) for agent_id in near_responders_df['agent_id'].tolist()]
        connected_responder_ids = [agent_id for agent_id in near_responder_ids if agent_id in visited]

        if require_all_responders:
            if len(connected_responder_ids) < 3:
                continue

            terminal_ids = connected_responder_ids
        else:
            if len(connected_responder_ids) == 0:
                continue

            terminal_ids = [connected_responder_ids[0]]

        for terminal_id in terminal_ids:
            node_path = []
            current_id = terminal_id

            while current_id is not None:
                node_path.append(current_id)
                current_id = parent[current_id]

            node_path.reverse()
            coordinate_path = [tuple(id_to_pos[node_id]) for node_id in node_path]
            coordinate_path.append(tuple(target_position))
            all_chains.append(coordinate_path)

    return all_chains


all_datasets = []
max_agents_global = 0

for idx, conf in enumerate(datasets_config):
    f_path = conf['file_path']
    
    if not os.path.exists(f_path):
        print(f"Warning: {f_path} not found. Skipped.")
        continue
        
    print(f"[{idx+1}/{len(datasets_config)}] Pre-computing data for {f_path}...")
    df = pd.read_csv(f_path)
    
    if 'role' not in df.columns:
        df['role'] = 0 
    
    num_agents = df['agent_id'].nunique()
    max_agents_global = max(max_agents_global, num_agents)
    total_steps = df['step'].max()
    
    target_pattern = re.compile(r"target_(\d+)_x")
    target_ids = []
    for col in df.columns:
        m = target_pattern.match(col)
        if m: target_ids.append(int(m.group(1)))
            
    targets_by_step = {}
    df_step_first = df.groupby('step').first().reset_index()
    
    for _, row in df_step_first.iterrows():
        step = int(row['step'])
        targets_by_step[step] = {}
        for t_id in target_ids:
            x_col = f"target_{t_id}_x"
            y_col = f"target_{t_id}_y"
            if x_col in df.columns and y_col in df.columns:
                targets_by_step[step][t_id] = (float(row[x_col]), float(row[y_col]))
        if not targets_by_step[step] and 'target_x' in df.columns and 'target_y' in df.columns:
            targets_by_step[step][0] = (float(row['target_x']), float(row['target_y']))
        if not targets_by_step[step]:
            targets_by_step[step][0] = (0.75, 0.0)

    min_x_obs, max_x_obs = df['pos_x'].min(), df['pos_x'].max()
    min_y_obs, max_y_obs = df['pos_y'].min(), df['pos_y'].max()
    if min_x_obs < -1.6 or max_x_obs > 1.6 or min_y_obs < -0.5 or max_y_obs > 0.5:
        xlim, ylim, wall_y_limit = (-3.1, 3.1), (-1.5, 1.5), 1.5
        default_base = (-2.5, 0.0)
    else:
        xlim, ylim, wall_y_limit = (-1.6, 1.6), (-0.5, 0.5), 0.5
        default_base = (-1.2, 0.0)

    base_x_cols = [c for c in df.columns if 'base' in c.lower() and c.lower().endswith('_x')]
    base_y_cols = [c for c in df.columns if 'base' in c.lower() and c.lower().endswith('_y')]
    
    if base_x_cols and base_y_cols:
        bx_col, by_col = base_x_cols[0], base_y_cols[0]
        base_pos_detected = (float(df[bx_col].iloc[0]), float(df[by_col].iloc[0]))
    else:
        step_1_df = df[df['step'] == 1]
        if not step_1_df.empty and xlim[0] < -1.6:
            mean_start_x = step_1_df['pos_x'].mean()
            base_pos_detected = (mean_start_x - 0.4, 0.0)
        else:
            base_pos_detected = default_base

    # 流线和热力图网格
    X_GRID, Y_V_GRID = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 65),
        np.linspace(-wall_y_limit, wall_y_limit, 33)
    )

    # 蓝色扩散箭头网格
    X_Q_GRID, Y_Q_GRID = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 33),
        np.linspace(-wall_y_limit, wall_y_limit, 17)
    )

    unique_agents = df['agent_id'].astype(int).unique()
    requested_focus_id = conf.get('focus_agent_id', None)

    if requested_focus_id is not None:
        requested_focus_id = int(requested_focus_id)

        if requested_focus_id not in unique_agents:
            raise ValueError(
                f"Scenario {idx+1}: specified focus agent "
                f"{requested_focus_id} does not exist in {f_path}. "
                f"Available agent IDs: {sorted(unique_agents.tolist())}"
            )

        selected_agent_id = requested_focus_id

        print(
            f"-> Scenario {idx+1}: manually selected "
            f"Agent {selected_agent_id} as the tracked agent."
        )

    else:
        # 未指定时，保留原来的自动选择机制
        candidate_agents = []

        for a_id in unique_agents:
            sub = df[df['agent_id'] == a_id]

            w_sum = (
                sub['w_rand']
                + sub['w_target']
                + sub['w_center']
                + 1e-8
            )

            has_rand = np.any(
                (sub['w_rand'] / w_sum) > 0.45
            )
            has_target = np.any(
                (sub['w_target'] / w_sum) > 0.45
            )
            has_center = np.any(
                (sub['w_center'] / w_sum) > 0.45
            )

            if has_rand and has_target and has_center:
                candidate_agents.append(int(a_id))

        if candidate_agents:
            selected_agent_id = candidate_agents[0]

            print(
                f"-> Scenario {idx+1}: automatically selected "
                f"Agent {selected_agent_id}."
            )
        else:
            selected_agent_id = int(unique_agents[0])

            print(
                f"-> Scenario {idx+1}: fallback to "
                f"Agent {selected_agent_id}."
            )
    
    df_agent = df[df['agent_id'] == selected_agent_id].sort_values('step').reset_index(drop=True)
    
    anchor_col = 'lambda_anchor' if 'lambda_anchor' in df_agent.columns else None
    release_col = 'lambda_release' if 'lambda_release' in df_agent.columns else None
    if anchor_col not in df_agent.columns: df_agent['lambda_anchor'] = 0.0
    if release_col not in df_agent.columns: df_agent['lambda_release'] = 0.0
    if 'k_diff' not in df_agent.columns: df_agent['k_diff'] = 0.0
    
    df_agent['w_sum'] = df_agent['w_rand'] + df_agent['w_target'] + df_agent['w_center'] + 1e-8
    df_agent['w_target_norm'] = df_agent['w_target'] / df_agent['w_sum']
    df_agent['w_center_norm'] = df_agent['w_center'] / df_agent['w_sum']
    df_agent['w_rand_norm'] = df_agent['w_rand'] / df_agent['w_sum']
    df_agent['lambda_anchor_val'] = df_agent[anchor_col] if anchor_col else np.zeros(len(df_agent))
    df_agent['lambda_release_val'] = df_agent[release_col] if release_col else np.zeros(len(df_agent))

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

    step_scene1_rand = None
    step_scene1_target = None
    step_scene1_center = None
    
    if idx == 0:
        for _, r in df_agent.iterrows():
            w_sum_r = r['w_rand'] + r['w_target'] + r['w_center'] + 1e-8
            step_val = int(r['step'])
            if step_scene1_rand is None and (r['w_rand'] / w_sum_r) > 0.45:
                step_scene1_rand = step_val
            if step_scene1_target is None and (r['w_target'] / w_sum_r) > 0.45:
                step_scene1_target = step_val
            if step_scene1_center is None and (r['w_center'] / w_sum_r) > 0.45:
                step_scene1_center = step_val
                
        if step_scene1_rand is None: step_scene1_rand = 20
        if step_scene1_target is None: step_scene1_target = 180
        if step_scene1_center is None: step_scene1_center = 340

        step_scene1_rand = 5
        step_scene1_target = 140
        step_scene1_center = 200

        print(f"   Scene 1 Trigger Frames -> Rand: {step_scene1_rand} | Target: {step_scene1_target} | Center: {step_scene1_center}")

    agent_paths = {}
    for i in range(num_agents):
        sub_df = df[df['agent_id'] == i].sort_values('step')
        agent_paths[i] = {
            'x': sub_df['pos_x'].values, 'y': sub_df['pos_y'].values, 
            'steps': sub_df['step'].values, 'role': sub_df['role'].values
        }
        
    all_datasets.append({
        'df': df, 
        'df_agent': df_agent, 
        'selected_agent_id': selected_agent_id,
        'paths': agent_paths, 
        'targets_by_step': targets_by_step, 'total_steps': total_steps,
        'xlim': xlim, 'ylim': ylim, 'wall_y_limit': wall_y_limit, 
        'base_pos': base_pos_detected, 'title': conf['title'],
        'has_failures': conf['has_failures'],
        'X_GRID': X_GRID, 'Y_V_GRID': Y_V_GRID,
        'X_Q_GRID': X_Q_GRID, 'Y_Q_GRID': Y_Q_GRID,
        'step_rand': step_scene1_rand,
        'step_target': step_scene1_target,
        'step_center': step_scene1_center
    })

frames_schedule = []
for _ in range(INTRO_FRAMES):
    frames_schedule.append({'mode': 'intro', 'd_idx': 0, 'step': 1, 'pause_msg': None})
    
for d_idx, data in enumerate(all_datasets):
    for _ in range(TRANSITION_FRAMES):
        frames_schedule.append({'mode': 'transition', 'd_idx': d_idx, 'step': 1, 'pause_msg': None})
    
    triggered_rand = False
    triggered_target = False
    triggered_center = False
    triggered_fail = False
    
    step = 1
    while step <= data['total_steps']:
        pause_msg = None
        trigger_pause = False
        
        if d_idx == 0:
            if step >= data['step_rand'] and not triggered_rand:
                trigger_pause = True
                triggered_rand = True
                pause_msg = r"Phase I: Distributed Area Exploration\n\nRobots enter the $\sigma_{exp}$ phase.\nThe swarm departs from the base and disperses as Searchers to maximize coverage of the unknown region.".replace('\\n', '\n')
            elif step >= data['step_target'] and not triggered_target:
                trigger_pause = True
                triggered_target = True
                pause_msg = r"Phase II: Target Discovery and Response\n\nRobots enter the $\sigma_{resp}$ phase.\nAfter the target is localized, nearby robots dynamically transition into Responders and converge on the target region.".replace('\\n', '\n')
            elif step >= data['step_center'] and not triggered_center:
                trigger_pause = True
                triggered_center = True
                pause_msg = r"Phase III: Self-Organized Communication Relay\n\nRobots enter the $\sigma_{relay}$ phase.\nIntermediate robots dynamically organize as Relays to maintain a robust multi-hop link between the base and responders.".replace('\\n', '\n')
        
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

# --- 轨迹与散点画笔 ---
trail_lines = [ax_map.plot([],[], lw=1.2, alpha=0.3, color='gray', zorder=1)[0] for _ in range(max_agents_global)]

scat_alive = ax_map.scatter([],[], s=70, edgecolors='k', linewidths=1.5, zorder=30)
scat_dead = ax_map.scatter([],[], s=80, c='red', edgecolors='k', linewidths=1.5, zorder=11)
scat_dead_cross = ax_map.scatter([],[], s=35, color='white', marker='x', linewidths=1.5, zorder=12)

scat_tracked = ax_map.scatter([],[], s=250, facecolors='none', edgecolors='gold', lw=3, zorder=50, label='Tracked Agent')

ax_map.set_xlabel("x position (m)", fontweight='bold'); ax_map.set_ylabel("y position (m)", fontweight='bold')

explanation_box = ax_map.text(0.04, 0.05, "", transform=ax_map.transAxes,
                              fontsize=11.5, fontweight='bold', color='black', family='sans-serif',
                              bbox=dict(boxstyle="round,pad=0.8", facecolor='#FFFEE6', alpha=0.92, edgecolor='darkorange', lw=2),
                              zorder=60, visible=False)

import matplotlib.lines as mlines
legend_elements = [
    mlines.Line2D([], [], color='dodgerblue', marker='o', linestyle='None', markersize=8, markeredgecolor='k', markeredgewidth=1.5, label='Searcher'),
    mlines.Line2D([], [], color='purple', marker='o', linestyle='None', markersize=8, markeredgecolor='k', markeredgewidth=1.5, label='Relay'),
    mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, markeredgecolor='k', markeredgewidth=1.5, label='Responder'),
]
ax_map.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=9.5)

# --- 参数图 ---
ax_param.set_xlabel("Simulation Steps", fontweight='bold')
ax_param.set_ylim(-0.05, 1.1) 
ax_param.set_ylabel(r"Coefficient ($\tilde{\omega}, \lambda$)", fontweight='bold', fontsize=11)

line_w_rand_norm, = ax_param.plot([],[], color='#4d4dfb', lw=2.5, alpha=0.4, label=r'$\tilde{\omega}_{rand}$')
line_w_center_norm, = ax_param.plot([],[], color='#8000ff', lw=2.5, label=r'$\tilde{\omega}_{center}$')
line_w_target_norm, = ax_param.plot([],[], color='#16c41c', lw=2, label=r'$\tilde{\omega}_{target}$')
line_l_anchor, = ax_param.plot([],[], color='#1697f3', lw=2.5, label=r'$\lambda_{anchor}$')
line_l_release, = ax_param.plot([],[], color='#4b6cff', lw=2.5, label=r'$\lambda_{release}$')

fail_line_p = ax_param.axvline(FAIL_STEP, color='red', linestyle='--', alpha=0.7)
ax_param.grid(True, alpha=0.3)

# 右轴
ax_param_twin = ax_param.twinx()
ax_param_twin.set_ylim(-0.01, 0.04) 
ax_param_twin.set_ylabel(r"Diffusion Coefficient ($D$)",  fontweight='bold', fontsize=11, labelpad=15)
ax_param_twin.tick_params(axis='y')
ax_param_twin.grid(False) 
ax_param_twin.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

line_k_diff, = ax_param_twin.plot([],[], color='#ff9a51', lw=2.5, label=r'$D$')

lines_all = [line_w_target_norm, line_w_center_norm, line_w_rand_norm, line_k_diff, line_l_anchor, line_l_release]
ax_param.legend(lines_all, [l.get_label() for l in lines_all], loc='lower center', 
                bbox_to_anchor=(0.5, 1.02), ncol=6, fontsize=9, framealpha=0.9)

current_d_idx = -1
geom_lines = []

def update_static_geometry(d_idx):
    global current_d_idx, geom_lines
    if d_idx == current_d_idx: return
    current_d_idx = d_idx
    data = all_datasets[d_idx]
    
    ax_map.set_position(gs[0, 0].get_position(fig))
    ax_param.set_position(gs[1, 0].get_position(fig))
    ax_param_twin.set_position(ax_param.get_position(fig))
    
    for line in geom_lines:
        try: line.remove()
        except: pass
    geom_lines.clear()
    
    for coll in list(ax_map.collections):
        if coll.get_label() in ['Fill_Wall', 'Base_Square']: coll.remove()
            
    xlim, ylim = data['xlim'], data['ylim']
    ax_map.set_xlim(xlim[0], xlim[1])
    ax_map.set_ylim(ylim[0], ylim[1])
    
    wall_y = data['wall_y_limit']
    walls_x = [xlim[0], xlim[1]]
    walls_y_upper = [wall_y, wall_y]
    walls_y_lower = [-wall_y, -wall_y]
    
    l1, = ax_map.plot(walls_x, walls_y_upper, color='#444444', lw=3, zorder=0)
    l2, = ax_map.plot(walls_x, walls_y_lower, color='#444444', lw=3, zorder=0)
    fill_u = ax_map.fill_between(walls_x, walls_y_upper, ylim[1], color='gray', alpha=0.15, label='Fill_Wall', zorder=0)
    fill_l = ax_map.fill_between(walls_x, ylim[0], walls_y_lower, color='gray', alpha=0.15, label='Fill_Wall', zorder=0)
    
    base_pos = data['base_pos']
    base = ax_map.scatter([base_pos[0]], [base_pos[1]], marker='s', s=300, color='lime', edgecolors='k', lw=1.5, zorder=9, label='Base_Square')
    geom_lines.extend([l1, l2])


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
        overlay_text.set_text("Density-field evolution and physical visualization\n \n \n Of Role-Adaptive Swarm Search and Rescue")
    elif mode == 'transition':
        overlay_rect.set_visible(True)
        overlay_text.set_visible(True)
        overlay_text.set_text(data['title']) 
    else:
        overlay_rect.set_visible(False)
        overlay_text.set_visible(False)

    update_static_geometry(d_idx)
    
    ax_map.set_title(f"Density-field visualization", fontsize=14, fontweight='bold')
    
    has_fail = data['has_failures']
    fail_line_p.set_visible(has_fail)
    
    ax_param.set_xlim(0, data['total_steps'])  
    
    curr_df = data['df'][data['df']['step'] == step]
    if curr_df.empty: 
        return [scat_alive, scat_dead, scat_dead_cross, scat_tracked, line_w_rand_norm, line_w_target_norm, line_w_center_norm, line_l_anchor, line_l_release, line_k_diff] + trail_lines
    
    if pause_msg:
        explanation_box.set_text(pause_msg)
        explanation_box.set_visible(True)
    else:
        explanation_box.set_visible(False)

    if hasattr(ax_map, 'current_stream') and ax_map.current_stream is not None:
        try:
            ax_map.current_stream.lines.remove() 
        except Exception:
            pass
        ax_map.current_stream = None
        
    for patch in list(ax_map.patches):
        if patch.get_zorder() in [2.5, 2.6]:
            patch.remove()

    for coll in list(ax_map.collections):
        if coll.get_label() in ['Target_Star', 'relay_link']: 
            coll.remove()
        elif coll.get_zorder() in [2.5, 2.6]: 
            coll.remove()

    for line in list(ax_map.lines):
        if line.get_label() == 'relay_link': 
            line.remove()

    current_targets = data['targets_by_step'].get(step, {0: (0.75, 0.0)})
    for t_id, pos in current_targets.items():
        ax_map.scatter([pos[0]], [pos[1]], marker='*', s=400, color='gold', edgecolors='k', lw=1.5, zorder=9, label='Target_Star')

    base_pos = data['base_pos']
    import re

    target_indices = sorted({int(re.match(r'target_(\d+)_x$', col).group(1)) for col in curr_df.columns if re.match(r'target_(\d+)_x$', col)})
    target_positions = [(float(curr_df[f'target_{t_idx}_x'].iloc[0]), float(curr_df[f'target_{t_idx}_y'].iloc[0])) for t_idx in target_indices]
    if 'assigned_target' in curr_df.columns:
        target_map_col = 'assigned_target'
    elif 'agent_target_map' in curr_df.columns:
        target_map_col = 'agent_target_map'
    else:
        target_map_col = None

    connected_chains = find_connected_chains(curr_df, base_pos, target_positions, comm_view=0.6, catch_distance=0.15, target_map_col=target_map_col, require_all_responders=True)
    for chain in connected_chains:
        for i in range(len(chain) - 1):
            ax_map.plot(
                [chain[i][0], chain[i+1][0]], 
                [chain[i][1], chain[i+1][1]], 
                color='#FFD700', linewidth=2.0, linestyle='--', zorder=8, label='relay_link'
            )

    if 'is_alive' in curr_df.columns:
        alive_df = curr_df[curr_df['is_alive'] == 1]
        dead_df = curr_df[curr_df['is_alive'] == 0]
    else:
        alive_df = curr_df[curr_df['role'] != -1]
        dead_df = curr_df[curr_df['role'] == -1]

    tracked_agent_id = data['selected_agent_id']
    tracked_agent_data = alive_df[
        alive_df['agent_id'] == tracked_agent_id
    ]

    assigned_target_id = None

    if not tracked_agent_data.empty:
        tracked_row = tracked_agent_data.iloc[0]

        fx = float(tracked_row['pos_x'])
        fy = float(tracked_row['pos_y'])

        wt = float(tracked_row['w_target'])
        wc = float(tracked_row['w_center'])
        wr = float(tracked_row['w_rand'])
        kd = float(tracked_row['k_diff'])

        # 优先读取该机器人的目标分配
        for target_col in ['assigned_target', 'agent_target_map']:
            if target_col in tracked_row.index and pd.notna(tracked_row[target_col]):
                candidate_id = int(tracked_row[target_col])

                if candidate_id in current_targets:
                    assigned_target_id = candidate_id
                    break

        # 没有明确分配信息时，选择最接近焦点机器人的基站—目标线段
        if assigned_target_id is None and len(current_targets) > 0:
            focus_point = np.array([fx, fy], dtype=float)
            base_point = np.asarray(base_pos, dtype=float)

            assigned_target_id = min(
                current_targets.keys(),
                key=lambda target_id: point_to_segment_distance(
                    focus_point,
                    base_point,
                    np.asarray(current_targets[target_id], dtype=float)
                )
            )

    else:
        fx, fy = base_pos

        wt = float(curr_df['w_target'].mean())
        wc = float(curr_df['w_center'].mean())
        wr = float(curr_df['w_rand'].mean())
        kd = float(curr_df['k_diff'].mean())

        if len(current_targets) > 0:
            assigned_target_id = sorted(current_targets.keys())[0]

    # 选择焦点机器人所属链条对应的目标
    if assigned_target_id is not None:
        primary_target_pos = current_targets[assigned_target_id]
    else:
        primary_target_pos = next(iter(current_targets.values()))
    
    u_adv, v_adv, rho_adv = get_subjective_potential_field(
        X_grid=data['X_GRID'], Y_grid=data['Y_V_GRID'], w_target=wt, w_center=wc, w_rand=wr, 
        active_targets=[primary_target_pos], base_pos=base_pos
    )
    
    if wr > 0.02:
        w_sum = wt + wc + wr + 1e-6
        w_rand_norm = wr / w_sum
        chaos_noise = 0.40 * np.cos(6.0 * data['X_GRID']) * np.sin(6.5 * data['Y_V_GRID']) + \
                      0.18 * np.sin(12.0 * data['X_GRID']) * np.cos(10.5 * data['Y_V_GRID'])
        rho_adv = np.clip(rho_adv * (1.0 + w_rand_norm * chaos_noise) - w_rand_norm * 0.08, 0.0, 1.0)

    field_agent_pos = curr_df[
        ['pos_x', 'pos_y']
    ].values

    # ==================================================
    # 在高分辨率网格上计算新的扩散速度场
    # ==================================================
    u_d_stream, v_d_stream, V_diff_mag, rho_diff_norm = (
        get_density_normalized_diffusion_field(
            X=data['X_GRID'],
            Y=data['Y_V_GRID'],
            kd=kd,
            agent_pos=field_agent_pos,
            focus_pos=(fx, fy),
            wall_y_limit=data['wall_y_limit'],
            eps_rho=0.05
        )
    )

    # ==================================================
    # 平流与扩散速度真实相加
    # ==================================================
    u_total = u_adv + u_d_stream
    v_total = v_adv + v_d_stream

    # ==================================================
    # 仅用于背景热力图的扩散强度压缩
    # ==================================================
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

    DIFFUSION_VIS_GAIN = 0.35

    rho_total = np.clip(
        rho_adv
        * np.exp(
            -DIFFUSION_VIS_GAIN
            * V_diff_vis
        ),
        0.0,
        1.0
    )

    max_pot = max(
        np.max(rho_total),
        1e-8
    )

    rho_normalized = (
        rho_total / max_pot
    )
    
    ax_map.contourf(
        data['X_GRID'], data['Y_V_GRID'], rho_normalized, levels=14, 
        cmap='Reds', alpha=0.35, zorder=2.5, antialiased=True
    )
    
    ax_map.current_stream = ax_map.streamplot(
        data['X_GRID'], data['Y_V_GRID'], u_total, v_total, 
        color=mcolors.to_rgba('crimson', alpha=0.3), linewidth=0.5, zorder=2.5, arrowsize=0.8
    )
    
    # ==================================================
    # 在稀疏对称网格上计算蓝色扩散箭头
    # ==================================================
    u_d_quiver, v_d_quiver, mag_quiver, rho_quiver = (
        get_density_normalized_diffusion_field(
            X=data['X_Q_GRID'],
            Y=data['Y_Q_GRID'],
            kd=kd,
            agent_pos=field_agent_pos,
            focus_pos=(fx, fy),
            wall_y_limit=data['wall_y_limit'],
            eps_rho=0.05
        )
    )

    positive_mag = mag_quiver[
        mag_quiver > 1e-10
    ]

    if positive_mag.size > 0:
        mag_ref = np.percentile(
            positive_mag,
            90.0
        )

        # 使用较低阈值，避免上下墙一侧刚好被过滤掉
        mask_active = (
            mag_quiver
            > 0.01 * mag_ref
        )

        # 单位方向
        u_unit = np.divide(
            u_d_quiver,
            mag_quiver,
            out=np.zeros_like(u_d_quiver),
            where=mag_quiver > 1e-10
        )

        v_unit = np.divide(
            v_d_quiver,
            mag_quiver,
            out=np.zeros_like(v_d_quiver),
            where=mag_quiver > 1e-10
        )

        # 保留有限的相对幅值，防止箭头忽长忽短
        relative_mag = np.clip(
            mag_quiver
            / (mag_ref + 1e-8),
            0.25,
            1.0
        )

        x_span = (
            data['xlim'][1]
            - data['xlim'][0]
        )

        arrow_length = (
            0.12
            if x_span > 4.0
            else 0.055
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

    colors = []
    for r in alive_df['role']:
        if r == 1: colors.append('red')       
        elif r == 2: colors.append('purple')     
        else: colors.append('dodgerblue')            
        
    scat_alive.set_offsets(alive_df[['pos_x', 'pos_y']])
    scat_alive.set_facecolors(colors)
    scat_alive.set_edgecolors('k')
    scat_alive.set_linewidths(1.5)
    
    if not tracked_agent_data.empty:
        scat_tracked.set_offsets(tracked_agent_data[['pos_x', 'pos_y']].values)
        scat_tracked.set_visible(True)
    else:
        scat_tracked.set_offsets(np.empty((0, 2)))
        scat_tracked.set_visible(False)

    if not dead_df.empty:
        scat_dead.set_offsets(dead_df[['pos_x', 'pos_y']])
        scat_dead_cross.set_offsets(dead_df[['pos_x', 'pos_y']])
        scat_dead.set_visible(True)
        scat_dead_cross.set_visible(True)
    else:
        scat_dead.set_visible(False)
        scat_dead_cross.set_visible(False)

    num_agents_curr = data['df']['agent_id'].nunique()
    for i in range(max_agents_global):
        if i < num_agents_curr:
            idx = np.searchsorted(data['paths'][i]['steps'], step, side='right')
            hist_x = data['paths'][i]['x'][:idx]
            hist_y = data['paths'][i]['y'][:idx]
            hist_role = data['paths'][i]['role'][:idx]
            
            trail_lines[i].set_data(hist_x, hist_y)
            if len(hist_role) > 0:
                last_role = hist_role[-1]
                if last_role == 1: trail_lines[i].set_color('crimson')
                elif last_role == 2: trail_lines[i].set_color('magenta')
                else: trail_lines[i].set_color('gray')
        else:
            trail_lines[i].set_data([], [])

    df_agent_dataset = data['df_agent']
    hist_agent = df_agent_dataset[df_agent_dataset['step'] <= step]
    
    line_w_target_norm.set_data(hist_agent['step'], hist_agent['w_target_smooth'])
    line_w_center_norm.set_data(hist_agent['step'], hist_agent['w_center_smooth'])
    line_w_rand_norm.set_data(hist_agent['step'], hist_agent['w_rand_smooth'])
    line_l_anchor.set_data(hist_agent['step'], hist_agent['lambda_anchor_smooth'])
    line_l_release.set_data(hist_agent['step'], hist_agent['lambda_release_smooth'])
    line_k_diff.set_data(hist_agent['step'], hist_agent['k_diff_smooth'])

    return [scat_alive, scat_dead, scat_dead_cross, scat_tracked, line_w_rand_norm, line_w_target_norm, line_w_center_norm, line_l_anchor, line_l_release, line_k_diff] + trail_lines


print("Running dynamically balanced three-in-one unified ADR rendering engine (Rescue Mode, High-Fidelity Contourf)...")
ani = animation.FuncAnimation(fig, update, frames=frames_schedule, blit=False, interval=30)
# plt.show()
out_dir = 'unified_videos'
os.makedirs(out_dir, exist_ok=True)
out_name = os.path.join(out_dir, 'rescue_field.mp4')

print(f"Saving compiled video to {out_name}...")
ani.save(out_name, 
         writer='ffmpeg', 
         fps=30,       
         dpi=150,      
         bitrate=-1,   
         extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '15', '-preset', 'fast'])
print("Rescue Masterpiece Video with High-Fidelity Contourf generated successfully!")