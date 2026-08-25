import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import glob
import os

file_paths = sorted(glob.glob('data/fails_*_0.csv'))
file_paths = [f for f in file_paths if os.path.basename(f) != 'fails_0.csv']
if not file_paths:
    print("Error: No fails_*.csv files found in data/ folder.")
    exit()

TRANSITION_TITLES = [
    f"Scene 1\n \n12.5% Failed Robots",
    f"Scene 2\n \n25.0% Failed Robots",
    f"Scene 3\n \n37.5% Failed Robots",
    f"Scene 4\n \n50.0% Failed Robots"
]

FAIL_STEP = 150        
MAX_STEPS = 400        # 严格限制绘制的最大步数为前 400 步
TRANSITION_FRAMES = 60 # 2秒转场(30fps)
INTRO_FRAMES = 120    # 4秒开场介绍(30fps)

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
TARGET_X = 1.15
TARGET_Y = 0.0

GRID_RES = 60  
XY_RANGE = 0.6
x_grid = np.linspace(-XY_RANGE, XY_RANGE, GRID_RES)
y_grid = np.linspace(-XY_RANGE, XY_RANGE, GRID_RES)
X, Y = np.meshgrid(x_grid, y_grid)

def get_ideal_beta(cx):
    if cx < -0.7: hw = 0.5
    elif -0.7 <= cx < -0.1: hw = 0.5 + (cx - (-0.7)) / (-0.1 - (-0.7)) * (0.12 - 0.5)
    elif -0.1 <= cx < 0.6: hw = 0.12
    elif 0.6 <= cx < 0.9: hw = 0.12 + (cx - 0.6) / (0.9 - 0.6) * (0.5 - 0.12)
    else: hw = 0.5
    norm_width = np.clip(hw / 0.5, 0.24, 1.0)
    return np.clip(1.0 + (1.0 - norm_width) * 5.0, 1.0, 3.0)

def calc_rho_ideal(beta_ideal):
    C = SHAPE_GAIN * W_SHAPE_IDEAL
    dist_warped = np.sqrt(X**2 + beta_ideal * Y**2)
    U = 0.5 * C * (dist_warped - R0)**2
    rho_th = np.exp(-U / K_DIFF_IDEAL)
    return rho_th / (np.sum(rho_th) + 1e-8)

def calc_rho_real(x_rel, y_rel):
    from scipy.stats import gaussian_kde
    positions = np.vstack([x_rel, y_rel])
    positions += np.random.normal(0, 1e-4, positions.shape)
    try:
        kernel = gaussian_kde(positions, bw_method=0.3)
        grid_coords = np.vstack([X.ravel(), Y.ravel()])
        rho_act = kernel(grid_coords).reshape(X.shape)
        return rho_act / (np.sum(rho_act) + 1e-8)
    except:
        return np.zeros_like(X)

def smooth(data, window=5):
    return np.convolve(data, np.ones(window)/window, mode='same')

all_datasets = []
max_agents = 0

for idx, f_path in enumerate(file_paths):
    print(f"[{idx+1}/{len(file_paths)}] Loading and pre-computing {f_path} ...")
    df = pd.read_csv(f_path)
    
    df = df[df['step'] <= MAX_STEPS].copy()
    
    num_agents = df['agent_id'].nunique()
    max_agents = max(max_agents, num_agents)
    total_steps = df['step'].max()
    
    df_macro = df.groupby('step').agg({'w_flow':'mean', 'w_shape':'mean', 'k_diff':'mean', 'beta':'mean'}).reset_index()
    
    df_macro['w_sum'] = df_macro['w_flow'] + df_macro['w_shape'] + 1e-8
    # 归一化宏观参数
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
    
    e_adr_list = []
    for s in df_macro['step']:
        group = df[(df['step'] == s) & (df['is_alive'] == 1)] 
        if len(group) < 2:
            e_adr_list.append(0)
            continue
            
        cx, cy = group['pos_x'].mean(), group['pos_y'].mean()
        if cx > 0.6:
            alpha = np.clip((cx - 0.6) / 0.4, 0.0, 1.0) 
            ref_x, ref_y = (1-alpha)*cx + alpha*TARGET_X, (1-alpha)*cy + alpha*TARGET_Y
        else:
            ref_x, ref_y = cx, cy
            
        x_rel, y_rel = group['pos_x'].values - ref_x, group['pos_y'].values - ref_y
        mse = np.mean((calc_rho_real(x_rel, y_rel) - calc_rho_ideal(get_ideal_beta(cx)))**2) * 10000 
        e_adr_list.append(mse)
        
    df_macro['e_adr'] = smooth(e_adr_list)
    max_err = df_macro['e_adr'].max()
    
    # 提取个体路径
    agent_paths = {}
    for i in range(num_agents):
        sub_df = df[df['agent_id'] == i].sort_values('step')
        agent_paths[i] = {
            'x': sub_df['pos_x'].values, 
            'y': sub_df['pos_y'].values, 
            'alive': sub_df['is_alive'].values, 
            'steps': sub_df['step'].values
        }
        
    all_datasets.append({
        'df': df, 'macro': df_macro, 'paths': agent_paths, 
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
    for step in range(1, data['total_steps'] + 1, 2):
        frames_schedule.append({'mode': 'sim', 'd_idx': d_idx, 'step': step})


fig = plt.figure(figsize=(14, 11)) 

gs = gridspec.GridSpec(3, 1, left=0.08, right=0.90, top=0.94, bottom=0.08, hspace=0.35, height_ratios=[2.5, 0.9, 0.9])

ax_map = fig.add_subplot(gs[0, 0]) 
ax_metric = fig.add_subplot(gs[1, 0]) 
ax_param = fig.add_subplot(gs[2, 0])  

# --- 地图背景 ---
ax_map.set_xlim(-1.6, 1.6); ax_map.set_ylim(-0.6, 0.6)
ax_map.set_aspect('equal'); ax_map.set_facecolor('#fdfdfd')
ax_map.set_xlabel("x position (m)", fontweight='bold'); ax_map.set_ylabel("y position (m)", fontweight='bold')

walls_x = [-1.5, -0.7, -0.1, 0.6, 0.9, 1.5]
ax_map.plot(walls_x,[0.6, 0.6, 0.12, 0.12, 0.6, 0.6], color='#444444', lw=3, label='Wall')
ax_map.plot(walls_x,[-0.6, -0.6, -0.12, -0.12, -0.6, -0.6], color='#444444', lw=3)
ax_map.fill_between(walls_x,[0.6, 0.6, 0.12, 0.12, 0.6, 0.6], 0.6, color='gray', alpha=0.2)
ax_map.fill_between(walls_x, -0.6,[-0.6, -0.6, -0.12, -0.12, -0.6, -0.6], color='gray', alpha=0.2)
ax_map.scatter([1.15], [0.0], marker='*', s=400, color='gold', edgecolors='k', lw=1.5, zorder=9, label='Target')

trail_lines = [
    ax_map.plot(
        [],
        [],
        lw=1.5,
        alpha=0.35,
        color='gray',
        zorder=2,
    )[0]
    for _ in range(max_agents)
]
scat_alive = ax_map.scatter([],[], s=100, c='dodgerblue', edgecolors='k', zorder=10, label='Benign Robot')
scat_dead = ax_map.scatter([],[], s=100, c='crimson', edgecolors='k', zorder=11, label='Failed Robot')
scat_dead_cross = ax_map.scatter([],[], s=40, color='white', marker='x', linewidths=2, zorder=12)

# --- 第二张子图：流形散度曲线 ---
ax_metric.set_ylabel(r'Divergence ($E_{ADR}$)', fontweight='bold', fontsize=12)
ax_metric.set_xlabel("Simulation Steps", fontweight='bold')
line_e, = ax_metric.plot([],[], color='#1a2a6c', lw=2.5)
fail_line_e = ax_metric.axvline(FAIL_STEP, color='red', linestyle='--', alpha=0.7)
fail_text_e = ax_metric.text(FAIL_STEP+5, 0, "Failure Injection", color='red', fontweight='bold')
ax_metric.grid(True, alpha=0.3)

# --- 第三张子图：双轴参数 ---
ax_param.set_xlabel("Simulation Steps", fontweight='bold')

# 【左侧轴：归一化 ratios (0-1) 与 lambda】
ax_param.set_ylim(-0.05, 1.1) 
ax_param.set_ylabel(r"Coefficient ($\tilde{\omega}, \lambda$)", fontweight='bold', fontsize=10)
line_w_norm, = ax_param.plot([],[], color='#16c41c', lw=2.5, label=r'$\tilde{\omega}_{flow}$')
line_w_shape_norm, = ax_param.plot([],[], color='#4d4dfb', lw=2.5, label=r'$\tilde{\omega}_{shape}$')
line_lambda, = ax_param.plot([],[], color='#1697f3', lw=2.5, label=r'$\lambda$')
fail_line_p = ax_param.axvline(FAIL_STEP, color='red', linestyle='--', alpha=0.7)
ax_param.grid(True, alpha=0.3)

# 【右侧轴：D】
ax_param_twin = ax_param.twinx()
ax_param_twin.set_ylim(-0.01, 0.04) 
ax_param_twin.set_ylabel(r"Coefficient ($D$)", fontweight='bold', fontsize=10, labelpad=15)
ax_param_twin.tick_params(axis='y')
ax_param_twin.grid(False) 

line_k_diff, = ax_param_twin.plot([],[], color='#ff9a51', lw=2.5, label=r'$D$')

lines_p = [line_w_norm, line_w_shape_norm, line_lambda, line_k_diff]
ax_param.legend(lines_p, [l.get_label() for l in lines_p], loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, framealpha=0.9, fontsize=9.5)


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
        overlay_text.set_text("Fault-tolerance verification in simulation\n \n \n For Formation-Reconfigurable Swarm Navigation")
    
    # --- 转场控制 ---
    elif mode == 'transition':
        overlay_rect.set_visible(True)
        overlay_text.set_visible(True)
        overlay_text.set_text(data['title']) 
    else:
        overlay_rect.set_visible(False)
        overlay_text.set_visible(False)

    ax_map.set_title(f"Fault-Tolerance Verification", fontsize=14, fontweight='bold')
    ax_metric.set_xlim(0, MAX_STEPS)
    ax_param.set_xlim(0, MAX_STEPS)
    
    max_e = data['max_err']
    y_lim_top = max_e * 1.1 if max_e > 0 else 0.05
    ax_metric.set_ylim(0, y_lim_top)
    fail_text_e.set_position((FAIL_STEP + 5, y_lim_top * 0.8))

    show_failure_marks = (d_idx >= 0)
    fail_line_e.set_visible(show_failure_marks)
    fail_line_p.set_visible(show_failure_marks)
    fail_text_e.set_visible(show_failure_marks)

    curr_df = data['df'][data['df']['step'] == step]
    if curr_df.empty: 
        return [scat_alive, scat_dead, scat_dead_cross, line_e, line_lambda, line_w_norm, line_w_shape_norm, line_k_diff] + trail_lines
    
    alive_mask = curr_df['is_alive'] == 1
    dead_mask = ~alive_mask
    dead_coords = curr_df[dead_mask][['pos_x', 'pos_y']] if dead_mask.any() else np.empty((0, 2))
    
    scat_alive.set_offsets(curr_df[alive_mask][['pos_x', 'pos_y']] if alive_mask.any() else np.empty((0, 2)))
    scat_dead.set_offsets(dead_coords)
    scat_dead_cross.set_offsets(dead_coords)

    for i in range(max_agents):
        if i in data['paths']:
            idx = np.searchsorted(
                data['paths'][i]['steps'],
                step,
                side='right',
            )

            trail_lines[i].set_data(
                data['paths'][i]['x'][:idx],
                data['paths'][i]['y'][:idx],
            )

            # 所有机器人轨迹统一为灰色
            trail_lines[i].set_color('gray')

            # 失败机器人轨迹稍微更淡
            if (
                idx > 0
                and data['paths'][i]['alive'][idx - 1] == 0
            ):
                trail_lines[i].set_alpha(0.18)
            else:
                trail_lines[i].set_alpha(0.35)

        else:
            trail_lines[i].set_data([], [])

    hist_macro = data['macro'][data['macro']['step'] <= step]
    line_e.set_data(hist_macro['step'], hist_macro['e_adr'])
    line_w_norm.set_data(hist_macro['step'], hist_macro['w_flow_smooth'])
    line_w_shape_norm.set_data(hist_macro['step'], hist_macro['w_shape_smooth'])
    line_lambda.set_data(hist_macro['step'], hist_macro['lambda_smooth'])
    line_k_diff.set_data(hist_macro['step'], hist_macro['k_diff_smooth'])

    return [scat_alive, scat_dead, scat_dead_cross, line_e, line_lambda, line_w_norm, line_w_shape_norm, line_k_diff] + trail_lines

print("Starting render engine. Setting title to size 24 and path colors to uniform blue...")
ani = animation.FuncAnimation(fig, update, frames=frames_schedule, interval=30, blit=False)
plt.show()
os.makedirs('elasticity_video', exist_ok=True)
out_name = 'elasticity_video/fails_compilation.mp4'
print(f"Saving compilation video to {out_name} (Rendering optimized layout)...")
ani.save(out_name, 
         writer='ffmpeg', 
         fps=30,       
         dpi=150,      
         bitrate=-1,   
         extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '15', '-preset', 'fast'])
print("Masterpiece Rendering Done! Optimized video generated successfully.")
