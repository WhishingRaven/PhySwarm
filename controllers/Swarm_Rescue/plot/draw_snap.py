import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
from matplotlib.legend_handler import HandlerTuple

WIDTH_CM = 12.0     # 目标宽度 (厘米)
HEIGHT_CM = 10    # 目标高度 (厘米)
FIG_WIDTH = WIDTH_CM / 2.54
FIG_HEIGHT = HEIGHT_CM / 2.54

plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.linewidth'] = 1.2


file_path = 'data/fails_0_9.csv' 
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: Could not find {file_path}.")
    exit()

num_agents = df['agent_id'].nunique()
steps_array = sorted(df['step'].unique())

success_steps = []
if 'rescued_count' in df.columns:
    rescued_array = df.groupby('step')['rescued_count'].first()
    diffs = rescued_array.diff()
    success_steps_raw = diffs[diffs > 0].index.tolist()
    for s in success_steps_raw:
        idx = steps_array.index(s)
        if idx > 0: 
            success_steps.append(steps_array[idx-1]) # 取目标移动前的最后一帧
else:
    prev_tx = df[df['step'] == steps_array[0]]['target_0_x'].iloc[0]
    prev_ty = df[df['step'] == steps_array[0]]['target_0_y'].iloc[0]
    for s in steps_array:
        current_tx = df[df['step'] == s]['target_0_x'].iloc[0]
        current_ty = df[df['step'] == s]['target_0_y'].iloc[0]
        dist = np.hypot(current_tx - prev_tx, current_ty - prev_ty)
        if dist > 0.02: 
            idx = steps_array.index(s)
            if idx > 0: success_steps.append(steps_array[idx-1]) 
            prev_tx = current_tx
            prev_ty = current_ty

if not success_steps:
    print("❌ 未在数据中侦测到救援成功事件！将默认绘制最后一步。")
    TARGET_STEP = steps_array[-1]
else:
    # 获取第一次成功的 Step
    TARGET_STEP = success_steps[1]
    print(f"✅ 成功截获首次救援链条形成的瞬间: Step {TARGET_STEP}")

curr_df = df[df['step'] == TARGET_STEP]

# --- 2. 创建专属画布 ---
fig, ax_map = plt.subplots(figsize=(8, 4), dpi=300)

ax_map.set_xlim(-1.5, 1.0)
ax_map.set_ylim(-0.6, 0.6)

ax_map.set_aspect('equal')
ax_map.set_facecolor('#fdfdfd')
ax_map.set_xlabel("x position (m)", fontweight='bold', fontsize=10)
ax_map.set_ylabel("y position (m)", fontweight='bold', fontsize=10)

ax_map.tick_params(axis='both', labelsize=10)

base_x = curr_df.iloc[0]['base_x']
base_y = curr_df.iloc[0]['base_y']
base_circle = plt.Circle((base_x, base_y), 0.1, color='green', alpha=0.4, zorder=5)
ax_map.add_patch(base_circle)

target_cols = [c for c in curr_df.columns if c.startswith('target_') and c.endswith('_x')]
target_positions = []
for i, t_col in enumerate(target_cols):
    t_idx = t_col.split('_')[1]
    tx = curr_df.iloc[0][t_col]
    ty = curr_df.iloc[0][f'target_{t_idx}_y']
    target_positions.append((tx, ty))
    
    target_circle = plt.Circle((tx, ty), 0.1, color='red', alpha=0.6, zorder=5)
    ax_map.add_patch(target_circle)

for i in range(num_agents):
    sub_df = df[(df['agent_id'] == i) & (df['step'] <= TARGET_STEP)].sort_values('step')
    if len(sub_df) == 0: continue
    
    # 检测该智能体到这一帧是否依然存活
    is_alive_at_end = sub_df.iloc[-1]['is_alive'] == 1
    
    x = sub_df['pos_x'].values
    y = sub_df['pos_y'].values
    roles = sub_df['role'].values
    
    if is_alive_at_end:
        change_indices = np.where(roles[:-1] != roles[1:])[0] + 1
        splits = np.concatenate(([0], change_indices, [len(roles)]))
        
        for j in range(len(splits) - 1):
            start_idx = splits[j]
            end_idx = splits[j+1]
            plot_end_idx = end_idx + 1 if end_idx < len(roles) else end_idx
            
            role = roles[start_idx]
            
            if role == 1:   # Responder (红色, 较粗)
                ax_map.plot(x[start_idx:plot_end_idx], y[start_idx:plot_end_idx], 
                            lw=1.5, color='crimson', alpha=0.4, zorder=2)
            elif role == 2: # Relay (紫色, 最粗)
                ax_map.plot(x[start_idx:plot_end_idx], y[start_idx:plot_end_idx], 
                            lw=2.0, color='purple', alpha=0.5, zorder=3)
            else:           # Searcher (灰色, 较细)
                ax_map.plot(x[start_idx:plot_end_idx], y[start_idx:plot_end_idx], 
                            lw=0.8, color='gray', alpha=0.3, zorder=1)
    else:
        ax_map.plot(x, y, lw=0.8, color='gray', alpha=0.15, zorder=1)

import collections

def find_connected_chains(curr_df, base_pos, target_positions, comm_view=0.6):
    """
    通过 BFS 搜索，找出从 Base 到各个 Target 的真实通信路径
    """
    alive_df = curr_df[curr_df['is_alive'] == 1]
    agents = alive_df[['agent_id', 'pos_x', 'pos_y', 'role']].to_dict('records')
    
    # 构建邻接表
    adj = collections.defaultdict(list)
    nodes_list = [{'id': 'Base', 'pos': base_pos}] + \
                 [{'id': a['agent_id'], 'pos': (a['pos_x'], a['pos_y']), 'role': a['role']} for a in agents]
    
    # 建立连接关系 
    n = len(nodes_list)
    for i in range(n):
        for j in range(i+1, n):
            p1, p2 = nodes_list[i]['pos'], nodes_list[j]['pos']
            if np.hypot(p1[0]-p2[0], p1[1]-p2[1]) < comm_view:
                adj[nodes_list[i]['id']].append(nodes_list[j]['id'])
                adj[nodes_list[j]['id']].append(nodes_list[i]['id'])
                
    # 针对每一个 Target 寻找连通路径
    all_chains = []
    for t_idx, t_pos in enumerate(target_positions):
        # 寻找靠近该 target 且存活的机器人作为终点源
        near_agents = []
        for a in agents:
            if np.hypot(a['pos_x']-t_pos[0], a['pos_y']-t_pos[1]) < comm_view:
                near_agents.append(a['agent_id'])
                
        if not near_agents:
            continue
            
        # BFS 寻找从 Base 到该 Target 附近机器人的最短路径
        queue = collections.deque([('Base', ['Base'])])
        visited = {'Base'}
        success_path = None
        
        while queue:
            curr, path = queue.popleft()
            if curr in near_agents:
                success_path = path
                break
            for nbr in adj[curr]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, path + [nbr]))
                    
        if success_path:
            # 将路径节点 ID 转换回坐标
            coord_path = []
            id_to_pos = {nd['id']: nd['pos'] for nd in nodes_list}
            for node_id in success_path:
                coord_path.append(id_to_pos[node_id])
            coord_path.append(t_pos) # 最后连向该目标点
            all_chains.append(coord_path)
            
    return all_chains

# 提取目标点的坐标
t0 = (curr_df.iloc[0]['target_0_x'], curr_df.iloc[0]['target_0_y'])
# t1 = (curr_df.iloc[0]['target_1_x'], curr_df.iloc[0]['target_1_y'])
# t2 = (curr_df.iloc[0]['target_2_x'], curr_df.iloc[0]['target_2_y'])
# target_positions = [t0,t1,t2]
target_positions = [t0]
base_pos = (base_x, base_y)

# 1. 寻找所有连通的链条路径
chains = find_connected_chains(curr_df, base_pos, target_positions, comm_view=0.6)

# 2. 将所有链条分解为唯一的“边（Edge）”
unique_edges = set()
for chain in chains:
    for k in range(len(chain) - 1):
        p1, p2 = chain[k], chain[k+1]
        # 对两个端点坐标进行排序，确保无视方向（A->B 和 B->A 视作同一条边）
        edge = tuple(sorted([p1, p2]))
        unique_edges.add(edge)

# 3. 逐条绘制唯一的边
for edge in unique_edges:
    xs = [edge[0][0], edge[1][0]]
    ys = [edge[0][1], edge[1][1]]

    ax_map.plot(xs, ys, color='#FFD700', lw=2, linestyle='--', zorder=16)

alive_df = curr_df[curr_df['is_alive'] == 1]
dead_df = curr_df[curr_df['is_alive'] == 0]

if len(alive_df) > 0:
    alive_colors = []
    for _, row in alive_df.iterrows():
        role = row['role']
        if role == 1: alive_colors.append('red')       # Responder
        elif role == 2: alive_colors.append('purple')  # Relay
        else: alive_colors.append('dodgerblue')        # Searcher
    
    ax_map.scatter(alive_df['pos_x'], alive_df['pos_y'], 
                   s=100, c=alive_colors, edgecolors='k', lw=1.5, zorder=30)

if len(dead_df) > 0:
    ax_map.scatter(dead_df['pos_x'], dead_df['pos_y'], 
                   s=150, c='crimson', edgecolors='k', lw=1.5, zorder=31)
    ax_map.scatter(dead_df['pos_x'], dead_df['pos_y'], 
                   s=55, color='white', marker='x', linewidths=2.2, zorder=32)

failed_circle = mlines.Line2D(
    [], [],
    linestyle='None',
    marker='o',
    markersize=10,
    markerfacecolor='crimson',
    markeredgecolor='black',
    markeredgewidth=1.2
)

failed_cross = mlines.Line2D(
    [], [],
    linestyle='None',
    marker='x',
    markersize=6.5,
    color='white',
    markeredgewidth=2.0
)

failed_robot_handle = (
    failed_circle,
    failed_cross
)

legend_handles = [
    mlines.Line2D(
        [], [],
        marker='o',
        linestyle='None',
        markersize=10,
        markerfacecolor='green',
        markeredgecolor='none',
        alpha=0.4
    ),

    mlines.Line2D(
        [], [],
        marker='o',
        linestyle='None',
        markersize=10,
        markerfacecolor='red',
        markeredgecolor='none',
        alpha=0.6
    ),

    failed_robot_handle,

    mlines.Line2D(
        [], [],
        marker='o',
        linestyle='None',
        markersize=8,
        markerfacecolor='dodgerblue',
        markeredgecolor='black'
    ),

    mlines.Line2D(
        [], [],
        marker='o',
        linestyle='None',
        markersize=8,
        markerfacecolor='red',
        markeredgecolor='black'
    ),

    mlines.Line2D(
        [], [],
        marker='o',
        linestyle='None',
        markersize=8,
        markerfacecolor='purple',
        markeredgecolor='black'
    ),
]

legend_labels = [
    'Base Station',
    'Rescue Target',
    'Failed Robot',
    'Searcher',
    'Responder',
    'Relay'
]

# ax_map.legend(
#     handles=legend_handles,
#     labels=legend_labels,
#     loc='upper left',
#     framealpha=0.95,
#     edgecolor='black',
#     ncol=2,
#     fontsize=8,

#     # ndivide=1 表示将圆圈和叉号绘制在同一个位置
#     handler_map={
#         tuple: HandlerTuple(
#             ndivide=1,
#             pad=0.0
#         )
#     }
# )

plt.tight_layout()

# --- 8. 导出 PDF ---
# os.makedirs("Fails_Figures", exist_ok=True)
# output_filename = f"Fails_Figures/rescue_fails_0.pdf"

os.makedirs("Paper_Figures", exist_ok=True)
output_filename = f"Paper_Figures/snap.pdf"

plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
print(f"✅ 成功! 高清科研快照已保存至: {output_filename}")

# plt.show()