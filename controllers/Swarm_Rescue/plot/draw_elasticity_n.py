import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection 
import matplotlib.lines as mlines 
from scipy.stats import gaussian_kde
import collections 
import os

datasets_config = [
    {
        'file_path': 'data/fails_rel_1.csv',
        'focus_agent': 3,
        'title': "Scene 1 \n \n Failed Relay Robot",
        'key_steps': [10, 220, 400],
        'labels': ["distributed-search", "target-response", "communication-relay"]
    },
    {
        'file_path': 'data/fails_resp_1.csv',
        'focus_agent': 4,
        'title': "Scene 2 \n \n Failed Responder Robot",
        'key_steps': [30, 250, 450],
        'labels': ["distributed-search", "communication-relay", "target-response"]
    },
    {
        'file_path': 'data/fails_resp_rel_2.csv',
        'focus_agent': 6,
        'title': "Scene 3 \n \n Failed Relay & Responder Robots",
        'key_steps': [20, 200, 400],
        'labels': ["distributed-search", "communication-relay", "target-response"]
    }
]

FAIL_STEP = 300        
TRANSITION_FRAMES = 60 
INTRO_FRAMES = 120      
SIM_FRAME_STEP = 3     

X_LIMIT = 1.5 
Y_LIMIT = 0.5
GRID_RES_X = 50
GRID_RES_Y = 30
X, Y = np.meshgrid(np.linspace(-X_LIMIT, X_LIMIT, GRID_RES_X), 
                   np.linspace(-Y_LIMIT, Y_LIMIT, GRID_RES_Y))
grid_coords = np.vstack([X.ravel(), Y.ravel()])

PARAM_SMOOTH_WINDOW = 21

def smooth_parameter_by_role(df_agent, column, window=21):
    values = df_agent[column].astype(float).copy()
    if 'role' not in df_agent.columns:
        return values.rolling(window=window, center=True, min_periods=1).mean()

    role_block = df_agent['role'].ne(df_agent['role'].shift()).cumsum()
    return df_agent.groupby(role_block, group_keys=False)[column].transform(lambda x: x.astype(float).rolling(window=window, center=True, min_periods=1).mean())

def smooth_data(data, window=5):
    if len(data) < window: return data
    return np.convolve(data, np.ones(window)/window, mode='same')

def calc_rho_real(agents_pos):
    """实际 KDE 密度估计"""
    N = len(agents_pos)
    if N == 0: return np.zeros_like(X) 
    bw = 0.60 if N <= 3 else 0.45 if N <= 5 else 0.30
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
    rho = np.ones_like(X)
    return rho / np.sum(rho)

def calc_rho_respond_th(target_pos, w_target, k_diff):
    dist_sq = (X - target_pos[0])**2 + (Y - target_pos[1])**2
    rho = np.exp(- (w_target / max(k_diff, 0.01)) * dist_sq)
    return rho / (np.sum(rho) + 1e-8)

def calc_rho_relay_th(base_pos, target_pos, w_center, k_diff):
    line_vec = target_pos - base_pos
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-6: return np.ones_like(X) / X.size
    line_dir = line_vec / line_len
    vec_to_base_x, vec_to_base_y = X - base_pos[0], Y - base_pos[1]
    proj_len = vec_to_base_x * line_dir[0] + vec_to_base_y * line_dir[1]
    perp_dist_sq = np.maximum((vec_to_base_x**2 + vec_to_base_y**2) - proj_len**2, 0)
    mask = (proj_len >= 0) & (proj_len <= line_len)
    rho = np.zeros_like(X)
    rho[mask] = np.exp(- (w_center / (max(k_diff, 0.01) + 0.1)) * perp_dist_sq[mask])
    return rho / (np.sum(rho) + 1e-8)

def find_connected_chains(curr_df, base_pos, target_positions, comm_view=0.6, catch_distance=0.15):
    if 'is_alive' in curr_df.columns:
        routing_df = curr_df[(curr_df['is_alive'] == 1) & (curr_df['role'].isin([1, 2]))].copy()
    else:
        routing_df = curr_df[curr_df['role'].isin([1, 2])].copy()

    if routing_df.empty:
        return []

    agents = routing_df[['agent_id', 'pos_x', 'pos_y', 'role']].to_dict('records')
    all_chains = []

    for target_pos in target_positions:
        target_pos = np.asarray(target_pos, dtype=float)
        base_pos_array = np.asarray(base_pos, dtype=float)

        near_responder_ids = []

        for agent in agents:
            agent_pos = np.array([agent['pos_x'], agent['pos_y']], dtype=float)
            distance_to_target = np.linalg.norm(agent_pos - target_pos)

            if int(agent['role']) == 1 and distance_to_target < catch_distance:
                near_responder_ids.append(int(agent['agent_id']))

        if len(near_responder_ids) < 3:
            continue

        base_id = 'Base'
        target_id = 'Target'
        adjacency = collections.defaultdict(list)
        id_to_pos = {base_id: base_pos_array, target_id: target_pos}

        for agent in agents:
            agent_id = int(agent['agent_id'])
            id_to_pos[agent_id] = np.array([agent['pos_x'], agent['pos_y']], dtype=float)

        agent_ids = [int(agent['agent_id']) for agent in agents]

        for agent_id in agent_ids:
            distance_to_base = np.linalg.norm(id_to_pos[agent_id] - base_pos_array)

            if distance_to_base < comm_view:
                adjacency[base_id].append(agent_id)
                adjacency[agent_id].append(base_id)

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                id_i = agent_ids[i]
                id_j = agent_ids[j]
                distance = np.linalg.norm(id_to_pos[id_i] - id_to_pos[id_j])

                if distance < comm_view:
                    adjacency[id_i].append(id_j)
                    adjacency[id_j].append(id_i)

        for agent_id in agent_ids:
            distance_to_target = np.linalg.norm(id_to_pos[agent_id] - target_pos)

            if distance_to_target < comm_view:
                adjacency[agent_id].append(target_id)
                adjacency[target_id].append(agent_id)

        queue = collections.deque([(base_id, [base_id])])
        visited = {base_id}
        successful_path = None

        while queue:
            current_id, current_path = queue.popleft()

            if current_id == target_id:
                successful_path = current_path
                break

            for neighbor_id in adjacency[current_id]:
                if neighbor_id in visited:
                    continue

                visited.add(neighbor_id)
                queue.append((neighbor_id, current_path + [neighbor_id]))

        if successful_path is None:
            continue

        coordinate_path = [tuple(id_to_pos[node_id]) for node_id in successful_path]
        all_chains.append(coordinate_path)

    return all_chains

all_datasets = []
max_agents_global = 0  

for idx, config in enumerate(datasets_config):
    f_path = config['file_path']
    if not os.path.exists(f_path):
        print(f"Error: {f_path} not found. Please ensure it is in the data/ folder.")
        exit()
        
    print(f"[{idx+1}/{len(datasets_config)}] Loading and pre-computing {f_path} ...")
    df = pd.read_csv(f_path)
    
    total_steps = df['step'].max()
    num_agents = df['agent_id'].nunique()
    max_agents_global = max(max_agents_global, num_agents)
    
    base_x = df.iloc[0]['base_x'] if 'base_x' in df.columns else -1.2
    base_y = df.iloc[0]['base_y'] if 'base_y' in df.columns else -0.2
    target_x = df.iloc[0]['target_0_x'] if 'target_0_x' in df.columns else 1.2
    target_y = df.iloc[0]['target_0_y'] if 'target_0_y' in df.columns else 0.2
    
    focus_id = config['focus_agent']
    df_agent = df[df['agent_id'] == focus_id].sort_values('step').reset_index(drop=True)

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
        
    if 'is_alive' not in df.columns:
        df['is_alive'] = 1
        
    e_adr_list = []
    sorted_steps = sorted(df['step'].unique())
    for s in sorted_steps:
        group = df[df['step'] == s]
        if len(group) == 0: 
            e_adr_list.append(0.0)
            continue
        
        b_pos = np.array([base_x, base_y])
        t_pos = np.array([target_x, target_y])
        w_target_mean = group['w_target'].mean()
        w_center_mean = group['w_center'].mean()
        k_diff_mean = group['k_diff'].mean()
        
        searchers = group[(group['role'] == 0) & (group['is_alive'] == 1)][['pos_x', 'pos_y']].values
        responders = group[(group['role'] == 1) & (group['is_alive'] == 1)][['pos_x', 'pos_y']].values
        relays = group[(group['role'] == 2) & (group['is_alive'] == 1)][['pos_x', 'pos_y']].values
        
        err_s, err_resp, err_rel = 0.0, 0.0, 0.0
        if len(searchers) > 0:
            err_s = np.mean((calc_rho_real(searchers) - calc_rho_search_th())**2)
        if len(responders) > 0:
            err_resp = np.mean((calc_rho_real(responders) - calc_rho_respond_th(t_pos, w_target_mean, k_diff_mean))**2)
        if len(relays) > 0:
            err_rel = np.mean((calc_rho_real(relays) - calc_rho_relay_th(b_pos, t_pos, w_center_mean, k_diff_mean))**2)
            
        N = len(group[group['is_alive'] == 1])
        if N > 0:
            weighted_err = ((len(searchers)/N) * err_s + 
                            (len(responders)/N) * err_resp + 
                            (len(relays)/N) * err_rel)
            e_adr_list.append(weighted_err * 10000)
        else:
            e_adr_list.append(0.0)
            
    df_macro = df.groupby('step').agg({
        'w_target': 'mean', 'w_center': 'mean', 'w_rand': 'mean', 'k_diff': 'mean'
    }).reset_index()
    
    df_macro['e_adr'] = smooth_data(e_adr_list, window=7)[:len(df_macro)]
    max_err = df_macro['e_adr'].max()
    
    agent_paths = {i: {
        'x': df[df['agent_id']==i]['pos_x'].values, 
        'y': df[df['agent_id']==i]['pos_y'].values,
        'alive': df[df['agent_id']==i]['is_alive'].values,
        'role': df[df['agent_id']==i]['role'].values
    } for i in range(num_agents)}
    
    all_datasets.append({
        'df': df, 
        'macro': df_macro,
        'paths': agent_paths, 
        'df_agent': df_agent,
        'total_steps': total_steps, 
        'num_agents': num_agents,
        'focus_agent': focus_id,
        'title': config['title'],
        'base_x': base_x,       
        'base_y': base_y,
        'target_x': target_x,
        'target_y': target_y,
        'max_err': max_err
    })

print(f"All pre-computations finished! Max agents: {max_agents_global}")

frames_schedule = []
for _ in range(INTRO_FRAMES):
    frames_schedule.append({'mode': 'intro', 'd_idx': 0, 'step': 1})

for d_idx, data in enumerate(all_datasets):
    for _ in range(TRANSITION_FRAMES):
        frames_schedule.append({'mode': 'transition', 'd_idx': d_idx, 'step': 1})
    for step in range(1, data['total_steps'] + 1, SIM_FRAME_STEP):
        frames_schedule.append({'mode': 'sim', 'd_idx': d_idx, 'step': step})


fig = plt.figure(figsize=(14, 11)) 
gs = gridspec.GridSpec(3, 1, left=0.08, right=0.90, top=0.94, bottom=0.06, hspace=0.35, height_ratios=[2.2, 0.9, 0.9])

ax_map = fig.add_subplot(gs[0, 0]) 
ax_metric = fig.add_subplot(gs[1, 0]) 
ax_param = fig.add_subplot(gs[2, 0])  

# --- 地图设置 ---
ax_map.set_xlim(-1.6, 1.6); ax_map.set_ylim(-0.6, 0.6)
ax_map.set_aspect('equal'); ax_map.set_facecolor('#fdfdfd')
ax_map.set_xlabel("x position (m)", fontweight='bold'); ax_map.set_ylabel("y position (m)", fontweight='bold')

scat_base_halo = ax_map.scatter([], [], s=1200, c='green', alpha=0.4, edgecolors='green', lw=1, zorder=1)
scat_target_halo = ax_map.scatter([], [], s=1200, c='red', alpha=0.4, edgecolors='darkred', lw=1, zorder=1)

# 初始化轨迹笔刷
trail_cols = []
for _ in range(max_agents_global):
    lc = LineCollection([], linewidths=1.5, alpha=0.4, zorder=3)
    ax_map.add_collection(lc)
    trail_cols.append(lc)

chain_col = LineCollection([], colors='#FFD700', linewidths=2.0, linestyles='--', zorder=16)
ax_map.add_collection(chain_col)

scat_search = ax_map.scatter([],[], s=80, c='dodgerblue', edgecolors='k', zorder=10)
scat_respond = ax_map.scatter([],[], s=80, c='red', edgecolors='k', zorder=10)
scat_relay = ax_map.scatter([],[], s=80, c='purple', edgecolors='k', zorder=10)
scat_tracked = ax_map.scatter([],[], s=200, facecolors='none', edgecolors='gold', lw=3, zorder=50, label='Tracked Agent')
scat_dead = ax_map.scatter([],[], s=100, c='crimson', edgecolors='k', zorder=11, label='Failed')
scat_dead_cross = ax_map.scatter([],[], s=40, color='white', marker='x', linewidths=2, zorder=12)

legend_elements = [
    mlines.Line2D([], [], color='none', marker='o', markerfacecolor='dodgerblue', markeredgecolor='k', markersize=8.5, label='Searcher'),
    mlines.Line2D([], [], color='none', marker='o', markerfacecolor='purple', markeredgecolor='k', markersize=8.5, label='Relay'),
    mlines.Line2D([], [], color='none', marker='o', markerfacecolor='red', markeredgecolor='k', markersize=8.5, label='Responder')
]
ax_map.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9, edgecolor='#CCCCCC')

# --- 第二张子图：流形散度曲线 ---
ax_metric.set_ylabel(r"Divergence ($E_{ADR}$)", fontweight='bold')
line_e, = ax_metric.plot([],[], color='#1a2a6c', lw=2.5, label='ADR Divergence')
fail_line_e = ax_metric.axvline(FAIL_STEP, color='red', linestyle='--', alpha=0.7)
fail_text_e = ax_metric.text(FAIL_STEP+5, 0.5, "Failure Injection", color='red', fontweight='bold', zorder=10)
ax_metric.grid(True, alpha=0.3)

# --- 第三张子图：参数图 ---
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
ax_param_twin.set_ylabel(r"Coefficient ($D$)",  fontweight='bold', fontsize=11, labelpad=15)
ax_param_twin.tick_params(axis='y')
ax_param_twin.grid(False) 

line_k_diff, = ax_param_twin.plot([],[], color='#ff9a51', lw=2.5, label=r'$D$')

lines_all = [line_w_target_norm, line_w_center_norm, line_w_rand_norm, line_k_diff, line_l_anchor, line_l_release]
ax_param.legend(lines_all, [l.get_label() for l in lines_all], loc='lower center', 
                bbox_to_anchor=(0.5, 1.02), ncol=6, fontsize=9, framealpha=0.9)

overlay_rect = plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, color='black', alpha=0.85, zorder=100)
fig.patches.append(overlay_rect)
overlay_text = fig.text(0.5, 0.5, '', color='white', fontsize=26, fontweight='bold', ha='center', va='center', zorder=101)

def update(frame_info):
    mode = frame_info['mode']
    d_idx = frame_info['d_idx']
    step = frame_info['step']
    data = all_datasets[d_idx]
    
    if mode == 'intro':
        overlay_rect.set_visible(True)
        overlay_text.set_visible(True)
        overlay_text.set_text("Fault-tolerance verification in simulation\n \n \n For Role-Adaptive Swarm Search and Rescue")
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

    curr_df = data['df'][data['df']['step'] == step]
    if curr_df.empty: 
        return [scat_search, scat_respond, scat_relay, scat_base_halo, scat_target_halo,
                scat_dead, scat_dead_cross, scat_tracked, line_e, line_w_target_norm, line_w_center_norm,
                 chain_col] + trail_cols
    
    alive = curr_df[curr_df['is_alive'] == 1]
    dead = curr_df[curr_df['is_alive'] == 0]
    
    curr_base_coords = np.array([[data['base_x'], data['base_y']]])
    curr_target_coords = np.array([[data['target_x'], data['target_y']]])
    
    scat_base_halo.set_offsets(curr_base_coords)
    scat_target_halo.set_offsets(curr_target_coords)
    
    scat_search.set_offsets(alive[alive['role'] == 0][['pos_x', 'pos_y']])
    scat_respond.set_offsets(alive[alive['role'] == 1][['pos_x', 'pos_y']])
    scat_relay.set_offsets(alive[alive['role'] == 2][['pos_x', 'pos_y']])
    
    base_pos = (data['base_x'], data['base_y'])
    target_positions = [(data['target_x'], data['target_y'])]
    
    chains = find_connected_chains(curr_df, base_pos, target_positions, comm_view=0.6)
    
    unique_edges = set()
    for chain in chains:
        for k in range(len(chain) - 1):
            p1, p2 = chain[k], chain[k+1]
            edge = tuple(sorted([p1, p2]))
            unique_edges.add(edge)
            
    segments = [[edge[0], edge[1]] for edge in unique_edges]
    chain_col.set_segments(segments)
    
    tracked_agent_id = data['focus_agent']
    tracked_data = curr_df[curr_df['agent_id'] == tracked_agent_id]
    if not tracked_data.empty:
        scat_tracked.set_offsets(tracked_data[['pos_x', 'pos_y']].values)
    else:
        scat_tracked.set_offsets(np.empty((0, 2)))
        
    dead_pos = dead[['pos_x', 'pos_y']]
    scat_dead.set_offsets(dead_pos)
    scat_dead_cross.set_offsets(dead_pos)

    for i in range(max_agents_global):
        if i < data['num_agents']:
            steps_arr = data['df'][data['df']['agent_id'] == i]['step'].values
            idx = np.searchsorted(steps_arr, step, side='right')
            if idx < 2: 
                trail_cols[i].set_segments([])
                continue
            
            x, y = data['paths'][i]['x'][:idx], data['paths'][i]['y'][:idx]
            roles = data['paths'][i]['role'][:idx]
            is_alive_status = data['paths'][i]['alive'][idx-1]
            
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments_t = np.concatenate([points[:-1], points[1:]], axis=1)
            colors = ['gray' if r == 0 else 'red' if r == 1 else 'purple' for r in roles[:-1]]
            
            trail_cols[i].set_alpha(0.05 if not is_alive_status else 0.4)
            trail_cols[i].set_segments(segments_t)
            trail_cols[i].set_colors(colors)
        else:
            trail_cols[i].set_segments([])

    hist_macro = data['macro'][data['macro']['step'] <= step]
    line_e.set_data(hist_macro['step'], hist_macro['e_adr'])

    df_agent_curr = data['df_agent']
    hist_agent = df_agent_curr[df_agent_curr['step'] <= step]
    
    line_w_target_norm.set_data(hist_agent['step'], hist_agent['w_target_smooth'])
    line_w_center_norm.set_data(hist_agent['step'], hist_agent['w_center_smooth'])
    line_w_rand_norm.set_data(hist_agent['step'], hist_agent['w_rand_smooth'])
    line_l_anchor.set_data(hist_agent['step'], hist_agent['lambda_anchor_smooth'])
    line_l_release.set_data(hist_agent['step'], hist_agent['lambda_release_smooth'])
    line_k_diff.set_data(hist_agent['step'], hist_agent['k_diff_smooth'])

    return [scat_search, scat_respond, scat_relay, scat_base_halo, scat_target_halo,
            scat_dead, scat_dead_cross, scat_tracked, line_e, 
            line_w_target_norm, line_w_center_norm, line_w_rand_norm, line_l_anchor, line_l_release, line_k_diff,
            chain_col] + trail_cols

print("Starting video rendering engine...")
ani = animation.FuncAnimation(fig, update, frames=frames_schedule, interval=30, blit=False)
plt.show()
out_name = 'elasticity_video/res_relay_compilation.mp4'
os.makedirs('elasticity_video', exist_ok=True)
print(f"Saving compilation video to {out_name} ...")

ani.save(out_name, 
         writer='ffmpeg', 
         fps=30,       
         dpi=150,      
         bitrate=-1,   
         extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '15', '-preset', 'fast'])
print("Masterpiece Rendering Done! Enjoy your video.")