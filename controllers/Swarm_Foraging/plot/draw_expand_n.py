import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.lines as mlines  # 🌟 引入物理线条代理库
from matplotlib.collections import LineCollection 
import glob
import os
import random  
import re

# ==========================================
#  视频转场与批量配置
# ==========================================
file_paths = sorted(
    glob.glob('data/expand_3_*.csv'),
    key=lambda x: int(
        re.search(
            r'expand_3_(\d+)\.csv$',
            os.path.basename(x)
        ).group(1)
    )
)
file_paths = [f for f in file_paths if os.path.basename(f) != 'expand_8.csv']
if not file_paths:
    print("Error: No expand_*.csv files found in data/ folder.")
    exit()

TRANSITION_TITLES = [
    "Scene 1\n \nSwarm Size: 16 Robots",
    "Scene 2\n \nSwarm Size: 24 Robots",
    "Scene 3\n \nSwarm Size: 32 Robots",
    "Scene 4\n \nSwarm Size: 40 Robots",
    "Scene 5\n \nSwarm Size: 48 Robots"
]

TRANSITION_FRAMES = 60 
SIM_FRAME_STEP = 3     
INTRO_FRAMES = 120    # 4秒开场介绍(30fps)

PARAM_SMOOTH_WINDOW = 21

def smooth_parameter_by_role(df_agent, column, window=21):
    values = df_agent[column].astype(float).copy()
    if 'role' not in df_agent.columns:
        return values.rolling(window=window, center=True, min_periods=1).mean()

    role_block = df_agent['role'].ne(df_agent['role'].shift()).cumsum()
    return df_agent.groupby(role_block, group_keys=False)[column].transform(lambda x: x.astype(float).rolling(window=window, center=True, min_periods=1).mean())


FOOD_POS = np.array([
    [-1.25, -0.5],[-1.25, 0.5], [1.25, -0.5]
])
NEST_POS = np.array([0.0, 0.0])


all_datasets = []
max_agents_global = 0  

for idx, f_path in enumerate(file_paths):
    if not os.path.exists(f_path):
        print(f"Warning: {f_path} not found. Skipping...")
        continue
        
    print(f"[{idx+1}/{len(file_paths)}] Loading and pre-computing {f_path} ...")
    df = pd.read_csv(f_path)
    
    total_steps = df['step'].max()
    num_agents = df['agent_id'].nunique()
    max_agents_global = max(max_agents_global, num_agents)
    
    local_max_food_cap = 5.0 
    cap_columns = [col for col in df.columns if 'cap' in col]
    if cap_columns:
        local_max_food_cap = max([df[col].max() for col in cap_columns])
    
    selected_agent_id = int(random.choice(df['agent_id'].unique()))
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
    
    # 提取个体路径
    agent_paths = {}
    for i in range(num_agents):
        sub_df = df[df['agent_id'] == i].sort_values('step')
        agent_paths[i] = {
            'x': sub_df['pos_x'].values,
            'y': sub_df['pos_y'].values,
            'steps': sub_df['step'].values,
            'is_carrying': sub_df['is_carrying'].values > 0, 
            'is_alive': sub_df['is_alive'].values
        }
        
    all_datasets.append({
        'df': df, 
        'df_agent': df_agent,               
        'tracked_id': selected_agent_id,    
        'paths': agent_paths, 
        'total_steps': total_steps, 
        'num_agents': num_agents,
        'local_max_food_cap': local_max_food_cap, 
        'title': TRANSITION_TITLES[idx]
    })

if not all_datasets:
    print("Error: No valid datasets loaded. Exiting.")
    exit()

print(f"All pre-computations finished! Max agents: {max_agents_global}")


frames_schedule = []
for _ in range(INTRO_FRAMES):
    frames_schedule.append({'mode': 'intro', 'd_idx': 0, 'step': 1})
for d_idx, data in enumerate(all_datasets):
    for _ in range(TRANSITION_FRAMES):
        frames_schedule.append({'mode': 'transition', 'd_idx': d_idx, 'step': 1})
    for step in range(1, data['total_steps'] + 1, SIM_FRAME_STEP):
        frames_schedule.append({'mode': 'sim', 'd_idx': d_idx, 'step': step})

fig = plt.figure(figsize=(15, 9.5), dpi=120)

gs_top = gridspec.GridSpec(1, 1, left=0.06, right=0.90, top=0.92, bottom=0.45) 
gs_bottom = gridspec.GridSpec(1, 1, left=0.14, right=0.82, top=0.32, bottom=0.08) 

ax_map = fig.add_subplot(gs_top[0, 0]) 
ax_param = fig.add_subplot(gs_bottom[0, 0])  

# --- 地图设置 ---
ax_map.set_xlim(-2.5, 2.5) 
ax_map.set_ylim(-1.1, 1.1)
ax_map.set_aspect('equal')
ax_map.set_facecolor('#f4f4f4') 
ax_map.set_xlabel("x position (m)", fontweight='bold', fontsize=12)
ax_map.set_ylabel("y position (m)", fontweight='bold', fontsize=12)

# 巢穴与食物
ax_map.scatter(0, 0, s=2300, c='green', alpha=0.4, edgecolors='green', lw=1, zorder=1)
scat_nest = ax_map.scatter([],[], s=80, c='green', edgecolors='green', lw=1, zorder=10)

food_circles = []
for fp in FOOD_POS:
    circle = patches.Circle((fp[0], fp[1]), 0.25, color='red', alpha=0.4, ec='darkred', lw=1, zorder=2)
    ax_map.add_patch(circle)
    food_circles.append(circle)

# 动态轨迹笔刷池
trail_lines_base = []
trail_lines_carry = [] 
for _ in range(max_agents_global):
    line_base, = ax_map.plot([],[], lw=1.2, alpha=0.4, color='gray', zorder=3)
    trail_lines_base.append(line_base)
    line_carry, = ax_map.plot([],[], lw=2.5, color='crimson', alpha=0.8, zorder=4)
    trail_lines_carry.append(line_carry)

agents_scat = ax_map.scatter([],[], s=70, edgecolors='k', linewidths=1.5, zorder=30)
scat_dead = ax_map.scatter([],[], s=70, c='black', alpha=0.3, edgecolors='k', linewidths=1.5, zorder=11)
scat_dead_cross = ax_map.scatter([],[], s=30, color='white', marker='x', linewidths=1.5, zorder=12)

scat_tracked = ax_map.scatter([],[], s=100, facecolors='none', edgecolors='gold', lw=3, zorder=50, label='Tracked Agent')

status_text = ax_map.text(0.02, 0.92, '', transform=ax_map.transAxes, 
                          fontsize=12, fontweight='bold', bbox=dict(boxstyle="round", fc="white", alpha=0.8, ec="gray"), zorder=40)

searching_handle = mlines.Line2D([], [], color='dodgerblue', marker='o', linestyle='None',
                                 markersize=8, markeredgecolor='k', label='Searching')
carrying_handle = mlines.Line2D([], [], color='crimson', marker='o', linestyle='None',
                                markersize=8, markeredgecolor='k', label='Carrying')

ax_map.legend(handles=[searching_handle, carrying_handle], loc='upper right', fontsize=10.5, framealpha=0.9, edgecolor='#EAEAEA')

# --- 参数演化图设置 ---
ax_param.set_xlabel("Simulation Steps", fontweight='bold', fontsize=12)

ax_param.set_ylim(-0.05, 1.1) 
ax_param.set_ylabel(r"Coefficient ($\tilde{w\omega}, \lambda$)", fontweight='bold', fontsize=12)

line_w_rand_norm, = ax_param.plot([],[], color='#ff2200', lw=2.5, label=r'$\tilde{\omega}_{rand}$')
line_w_info_norm, = ax_param.plot([],[], color='#8000ff', lw=2.5, label=r'$\tilde{\omega}_{info}$')
line_w_food_norm, = ax_param.plot([],[], color='#16c41c', lw=2.5, label=r'$\tilde{\omega}_{food}$')
line_w_nest_norm, = ax_param.plot([],[], color='#3dd5e9', lw=2.5, label=r'$\tilde{\omega}_{nest}$')
line_l_pick, = ax_param.plot([],[], color='#4b6cff', lw=2.5, label=r'$\lambda_{pick}$')
line_l_drop, = ax_param.plot([],[], color='#1697f3', lw=2.5, label=r'$\lambda_{drop}$')
ax_param.grid(True, alpha=0.3)

# 【右侧轴：D】
ax_param_twin = ax_param.twinx()
ax_param_twin.set_ylim(-0.01, 0.09) 
ax_param_twin.set_ylabel(r"Coefficient ($D$)", fontweight='bold', fontsize=10, labelpad=15)
ax_param_twin.tick_params(axis='y')
ax_param_twin.grid(False) 
ax_param_twin.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

line_k_diff, = ax_param_twin.plot([],[], color='#ff9a51', lw=2.5, label=r'$D$')

lines_p = [line_w_food_norm, line_w_nest_norm, line_w_info_norm, line_w_rand_norm, line_k_diff, line_l_pick, line_l_drop]
ax_param.legend(lines_p, [l.get_label() for l in lines_p], loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=7, framealpha=0.9, fontsize=9.5)


overlay_rect = plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, color='black', alpha=0.85, zorder=100)
fig.patches.append(overlay_rect)
overlay_text = fig.text(0.5, 0.5, '', color='white', fontsize=24, fontweight='bold', ha='center', va='center', zorder=101)


def update(frame_info):
    mode = frame_info['mode']
    d_idx = frame_info['d_idx']
    step = frame_info['step']
    data = all_datasets[d_idx]
    
    if mode == 'intro':
        overlay_rect.set_visible(True)
        overlay_text.set_visible(True)
        overlay_text.set_text("Scalability verification in simulation\n \n \n For Trail-Guided Swarm Foraging")
    
    # --- 转场控制 ---
    elif mode == 'transition':
        overlay_rect.set_visible(True)
        overlay_text.set_visible(True)
        overlay_text.set_text(data['title']) 
    else:
        overlay_rect.set_visible(False)
        overlay_text.set_visible(False)

    # --- 动态调整图表属性 ---
    ax_map.set_title(f"Scalability Verification", fontsize=14, fontweight='bold')
    ax_param.set_xlim(0, data['total_steps'])
    
    curr_df = data['df'][data['df']['step'] == step]
    if curr_df.empty: 
        return (agents_scat, scat_dead, scat_dead_cross, scat_tracked, status_text, 
                line_w_food_norm, line_w_nest_norm, line_w_info_norm, line_w_rand_norm, 
                line_k_diff, line_l_pick, line_l_drop, *trail_lines_base, *trail_lines_carry)

    alive = curr_df[curr_df['is_alive'] == 1]
    dead = curr_df[curr_df['is_alive'] == 0]

    # 食物圆圈大小缩放
    current_max_food_cap = data['local_max_food_cap']
    for i in range(len(FOOD_POS)):
        cap = curr_df[f'target_{i}_cap'].iloc[0] if f'target_{i}_cap' in curr_df.columns else 0
        food_circles[i].set_radius(np.sqrt(max(cap, 0) / (current_max_food_cap + 1e-6)) * 0.15)

    # 轨迹更新
    for i in range(max_agents_global):
        if i < data['num_agents']:
            current_idx = np.searchsorted(data['paths'][i]['steps'], step, side='right')
            if current_idx < 2: 
                trail_lines_base[i].set_data([], [])
                trail_lines_carry[i].set_data([],[])
                continue
                
            hist_x = data['paths'][i]['x'][:current_idx].astype(float).copy()
            hist_y = data['paths'][i]['y'][:current_idx].astype(float).copy()
            hist_carry = data['paths'][i]['is_carrying'][:current_idx]
            is_alive = data['paths'][i]['is_alive'][current_idx-1]
            
            trail_lines_base[i].set_data(hist_x, hist_y)
            carry_x, carry_y = hist_x.copy(), hist_y.copy()
            carry_x[~hist_carry] = np.nan
            carry_y[~hist_carry] = np.nan
            trail_lines_carry[i].set_data(carry_x, carry_y)
            trail_lines_base[i].set_alpha(0.4 if is_alive else 0.1)
            trail_lines_carry[i].set_alpha(0.8 if is_alive else 0.1)
        else:
            trail_lines_base[i].set_data([], [])
            trail_lines_carry[i].set_data([],[])

    # 散点更新
    if not alive.empty:
        positions = alive[['pos_x', 'pos_y']].values
        face_colors = np.where(alive['is_carrying'].values, 'crimson', 'dodgerblue')
        agents_scat.set_offsets(positions)
        agents_scat.set_facecolors(face_colors)
        
        # 更新被追踪智能体的金色光圈坐标
        tracked_agent_data = alive[alive['agent_id'] == data['tracked_id']]
        if not tracked_agent_data.empty:
            scat_tracked.set_offsets(tracked_agent_data[['pos_x', 'pos_y']].values)
        else:
            scat_tracked.set_offsets(np.empty((0, 2)))
    else:
        agents_scat.set_offsets(np.empty((0,2)))
        scat_tracked.set_offsets(np.empty((0,2)))
        
    if not dead.empty:
        dead_pos = dead[['pos_x', 'pos_y']].values
        scat_dead.set_offsets(dead_pos)
        scat_dead_cross.set_offsets(dead_pos)
    else:
        scat_dead.set_offsets(np.empty((0,2)))
        scat_dead_cross.set_offsets(np.empty((0,2)))

    # 数据源更新，左轴与右轴分离渲染
    hist_agent = data['df_agent'][data['df_agent']['step'] <= step]
    
    # 左轴更新
    line_w_food_norm.set_data(hist_agent['step'], hist_agent['w_food_smooth'])
    line_w_nest_norm.set_data(hist_agent['step'], hist_agent['w_nest_smooth'])
    line_w_info_norm.set_data(hist_agent['step'], hist_agent['w_info_smooth'])
    line_w_rand_norm.set_data(hist_agent['step'], hist_agent['w_rand_smooth'])
    line_l_pick.set_data(hist_agent['step'], hist_agent['lambda_pick_smooth'])
    line_l_drop.set_data(hist_agent['step'], hist_agent['lambda_drop_smooth'])
    
    # 右轴更新
    line_k_diff.set_data(hist_agent['step'], hist_agent['k_diff_smooth'])

    return (agents_scat, scat_dead, scat_dead_cross, scat_tracked, status_text, 
            line_w_food_norm, line_w_nest_norm, line_w_info_norm, line_w_rand_norm, 
            line_k_diff, line_l_pick, line_l_drop, *trail_lines_base, *trail_lines_carry)

print("\nStarting video rendering engine...")
ani = animation.FuncAnimation(fig, update, frames=frames_schedule, blit=False, interval=30)
# plt.show()
os.makedirs('expand_video', exist_ok=True)
out_name = 'expand_video/expand_compilation.mp4'
print(f"Saving compilation video to {out_name} (This may take a few minutes)...")

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

print("Masterpiece Rendering Done! Check the expand_video folder.")
