import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import os


DATASET_CONFIGS = [
    {
        'path': 'data/expand_0.3_16.csv',
        'beta_max': 5.0,
    },
    {
        'path': 'data/expand_0.4_24.csv',
        'beta_max': 5.0,
    },
    {
        'path': 'data/expand_0.5_32.csv',
        'beta_max': 5.0,
    },
    {
        'path': 'data/expand_0.6_40.csv',
        'beta_max': 8.0,
    },
    {
        'path': 'data/expand_0.6_48.csv',
        'beta_max': 8.0,
    },
]

TRANSITION_TITLES = [
    "Scene 1\n \nSwarm Size: 16 Robots",
    "Scene 2\n \nSwarm Size: 24 Robots",
    "Scene 3\n \nSwarm Size: 32 Robots",
    "Scene 4\n \nSwarm Size: 40 Robots",
    "Scene 5\n \nSwarm Size: 48 Robots"
]

TRANSITION_FRAMES = 60 # 每幕开始前的转场时间: 2秒(30fps)
SIM_FRAME_STEP = 3     # 仿真播放步长(加速播放用)
INTRO_FRAMES = 120    # 4秒开场介绍(30fps)

PARAM_SMOOTH_WINDOW = 21

def smooth_parameter_by_role(df_agent, column, window=21):
    values = df_agent[column].astype(float).copy()
    if 'role' not in df_agent.columns:
        return values.rolling(window=window, center=True, min_periods=1).mean()

    role_block = df_agent['role'].ne(df_agent['role'].shift()).cumsum()
    return df_agent.groupby(role_block, group_keys=False)[column].transform(lambda x: x.astype(float).rolling(window=window, center=True, min_periods=1).mean())

all_datasets = []
max_agents_global = 0

for idx, config in enumerate(DATASET_CONFIGS):
    f_path = config['path']
    beta_max = config['beta_max']

    if not os.path.exists(f_path):
        print(
            f"Warning: {f_path} not found. Skipping..."
        )
        continue

    print(
        f"[{idx + 1}/{len(DATASET_CONFIGS)}] "
        f"Loading and pre-computing {f_path} ..."
    )

    df = pd.read_csv(f_path)

    total_steps = df['step'].max()
    num_agents = df['agent_id'].nunique()

    max_agents_global = max(
        max_agents_global,
        num_agents,
    )

    df_macro = (
        df.groupby('step')
        .agg({
            'w_flow': 'mean',
            'w_shape': 'mean',
            'k_diff': 'mean',
            'beta': 'mean',
        })
        .reset_index()
    )

    df_macro['w_sum'] = (
        df_macro['w_flow']
        + df_macro['w_shape']
        + 1e-8
    )

    BETA_MIN = 1.0

    df_macro['lambda_val'] = np.clip(
        (df_macro['beta'] - BETA_MIN)
        / (beta_max - BETA_MIN),
        0.0,
        1.0,
    )

    df_macro['w_flow_norm'] = (
        df_macro['w_flow']
        / df_macro['w_sum']
    )

    df_macro['w_shape_norm'] = (
        df_macro['w_shape']
        / df_macro['w_sum']
    )

    df_macro['w_flow_smooth'] = smooth_parameter_by_role(df_macro, 'w_flow_norm', PARAM_SMOOTH_WINDOW)
    df_macro['w_shape_smooth'] = smooth_parameter_by_role(df_macro, 'w_shape_norm', PARAM_SMOOTH_WINDOW)
    df_macro['lambda_smooth'] = smooth_parameter_by_role(df_macro, 'lambda_val', PARAM_SMOOTH_WINDOW)
    df_macro['k_diff_smooth'] = smooth_parameter_by_role(df_macro, 'k_diff', PARAM_SMOOTH_WINDOW)

    smooth_w_sum = df_macro['w_flow_smooth'] + df_macro['w_shape_smooth'] + 1e-8
    df_macro['w_flow_smooth'] = df_macro['w_flow_smooth'] / smooth_w_sum
    df_macro['w_shape_smooth'] = df_macro['w_shape_smooth'] / smooth_w_sum

    df_macro['lambda_smooth'] = np.clip(df_macro['lambda_smooth'], 0.0, 1.0)
    df_macro['k_diff_smooth'] = np.clip(df_macro['k_diff_smooth'], 0.0, None)

    agent_paths = {}

    for i in range(num_agents):
        sub_df = (
            df[df['agent_id'] == i]
            .sort_values('step')
        )

        agent_paths[i] = {
            'x': sub_df['pos_x'].values,
            'y': sub_df['pos_y'].values,
            'steps': sub_df['step'].values,
        }

    all_datasets.append({
        'df': df,
        'macro': df_macro,
        'paths': agent_paths,
        'total_steps': total_steps,
        'num_agents': num_agents,
        'beta_max': beta_max,
        'title': TRANSITION_TITLES[idx],
    })

if not all_datasets:
    print("Error: No valid datasets loaded. Exiting.")
    exit()

print(f"All pre-computations finished! Max agents across all sets: {max_agents_global}")

frames_schedule = []
for _ in range(INTRO_FRAMES):
    frames_schedule.append({'mode': 'intro', 'd_idx': 0, 'step': 1})
for d_idx, data in enumerate(all_datasets):
    for _ in range(TRANSITION_FRAMES):
        frames_schedule.append({'mode': 'transition', 'd_idx': d_idx, 'step': 1})
    for step in range(1, data['total_steps'] + 1, SIM_FRAME_STEP):
        frames_schedule.append({'mode': 'sim', 'd_idx': d_idx, 'step': step})


fig = plt.figure(figsize=(15, 9.5), dpi=120)

gs_top = gridspec.GridSpec(1, 1, left=0.08, right=0.90, top=0.92, bottom=0.45) 
gs_bottom = gridspec.GridSpec(1, 1, left=0.08, right=0.90, top=0.35, bottom=0.08) 

ax_map = fig.add_subplot(gs_top[0, 0]) 
ax_param = fig.add_subplot(gs_bottom[0, 0])  

# --- 地图设置 ---
ax_map.set_xlim(-3.1, 3.1)
ax_map.set_ylim(-1.2, 1.2)
ax_map.set_aspect('equal')
ax_map.set_facecolor('#f4f4f4') 
ax_map.set_xlabel("x position (m)", fontweight='bold', fontsize=12)
ax_map.set_ylabel("y position (m)", fontweight='bold', fontsize=12)

walls_x = [
    -3.1,
    -1.7,
    -0.5,
    0.9,
    1.5,
    3.1,
]

walls_y_upper = [
    1.0,
    1.0,
    0.38,
    0.38,
    1.0,
    1.0,
]

walls_y_lower = [
    -1.0,
    -1.0,
    -0.38,
    -0.38,
    -1.0,
    -1.0,
]
ax_map.plot(walls_x, walls_y_upper, color='#444444', lw=3, label='Wall')
ax_map.plot(walls_x, walls_y_lower, color='#444444', lw=3)
ax_map.fill_between(walls_x, walls_y_upper, 1.2, color='gray', alpha=0.2)
ax_map.fill_between(walls_x, -1.2, walls_y_lower, color='gray', alpha=0.2)

goal_x, goal_y = 2.0, 0.0
ax_map.plot(goal_x, goal_y, 'y*', markersize=22, markeredgecolor='orange', label='Goal', zorder=20)

# 动态轨迹笔刷池
trail_lines_base = []
for _ in range(max_agents_global):
    line_base, = ax_map.plot([],[], lw=1.2, alpha=0.3, color='gray', zorder=1)
    trail_lines_base.append(line_base)

agents_scat = ax_map.scatter([],[], s=70, color='#1F77B4', edgecolors='k', linewidths=1.5, zorder=30)

# --- 参数演化图设置 ---
ax_param.set_xlabel("Simulation Steps", fontweight='bold', fontsize=12)

# 【左侧轴：归一化比值与反应项】
ax_param.set_ylim(-0.05, 1.1) 
ax_param.set_ylabel(r"Coefficient ($\tilde{\omega}, \lambda$)", fontweight='bold', fontsize=10)
line_w_norm, = ax_param.plot([],[], color='#16c41c', lw=2.5, label=r'$\tilde{\omega}_{flow}$')
line_w_shape_norm, = ax_param.plot([],[], color='#4d4dfb', lw=2.5, label=r'$\tilde{\omega}_{shape}$')
line_lambda, = ax_param.plot([],[], color='#1697f3', lw=2.5, label=r'$\lambda$')
ax_param.grid(True, alpha=0.3)

# 【右侧轴：微观扩散项 D】
ax_param_twin = ax_param.twinx()
ax_param_twin.set_ylim(-0.01, 0.04) 
ax_param_twin.set_ylabel(r"Coefficient ($D$)", fontweight='bold', fontsize=10, labelpad=15)
ax_param_twin.tick_params(axis='y')
ax_param_twin.grid(False) 

line_k_diff, = ax_param_twin.plot([],[], color='#ff9a51', lw=2.5, label=r'$D$')

# 合并图例并将其平铺在底部子图上方
lines_p = [line_w_norm, line_w_shape_norm, line_lambda, line_k_diff]
ax_param.legend(lines_p, [l.get_label() for l in lines_p], loc='lower center', 
                bbox_to_anchor=(0.5, 1.02), ncol=4, framealpha=0.9, fontsize=9.5)


overlay_rect = plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, color='black', alpha=0.85, zorder=100)
fig.patches.append(overlay_rect)

overlay_text = fig.text(0.5, 0.5, '', color='white', fontsize=24, fontweight='bold', ha='center', va='center', zorder=101)

def update(frame_info):
    mode = frame_info['mode']
    d_idx = frame_info['d_idx']
    step = frame_info['step']
    data = all_datasets[d_idx]
    
    # --- 转场逻辑 ---
    if mode == 'intro':
        overlay_rect.set_visible(True)
        overlay_text.set_visible(True)
        overlay_text.set_text("Scalability verification in simulation\n \n \n For Formation-Reconfigurable Swarm Navigation")
    elif mode == 'transition':
        overlay_rect.set_visible(True)
        overlay_text.set_visible(True)
        overlay_text.set_text(data['title']) 
    else:
        overlay_rect.set_visible(False)
        overlay_text.set_visible(False)

    # 动态调整标题
    ax_map.set_title(f"Scalability verification", fontsize=14, fontweight='bold')
    ax_param.set_xlim(0, data['total_steps'])
    
    # 提取当前帧数据
    curr_df = data['df'][data['df']['step'] == step]
    if curr_df.empty: 
        return (agents_scat, line_w_norm, line_w_shape_norm, line_k_diff, line_lambda, *trail_lines_base)

    positions = curr_df[['pos_x', 'pos_y']].values
    
    # 更新基础轨迹
    for i in range(max_agents_global):
        if i < data['num_agents']:
            current_idx = np.searchsorted(data['paths'][i]['steps'], step, side='right')
            hist_x = data['paths'][i]['x'][:current_idx].astype(float).copy()
            hist_y = data['paths'][i]['y'][:current_idx].astype(float).copy()
            
            trail_lines_base[i].set_data(hist_x, hist_y)
        else:
            trail_lines_base[i].set_data([], [])

    agents_scat.set_offsets(positions)

    hist_macro = data['macro'][data['macro']['step'] <= step]
    
    # 左轴线更新
    line_w_norm.set_data(hist_macro['step'], hist_macro['w_flow_smooth'])
    line_w_shape_norm.set_data(hist_macro['step'], hist_macro['w_shape_smooth'])
    line_lambda.set_data(hist_macro['step'], hist_macro['lambda_smooth'])
    
    # 右轴线更新
    line_k_diff.set_data(hist_macro['step'], hist_macro['k_diff_smooth'])

    return (agents_scat, line_w_norm, line_w_shape_norm, line_k_diff, line_lambda, *trail_lines_base)

print("\nStarting video rendering engine...")
ani = animation.FuncAnimation(fig, update, frames=frames_schedule, blit=False, interval=30)
# plt.show()
os.makedirs('expand_video', exist_ok=True)
out_name = 'expand_video/expand_compilation_all.mp4'
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
