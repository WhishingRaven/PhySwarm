import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

WIDTH_CM = 24.0     # 目标宽度 (厘米)
HEIGHT_CM = 8.0     # 目标高度 (厘米)
FIG_WIDTH = WIDTH_CM / 2.54
FIG_HEIGHT = HEIGHT_CM / 2.54


file_path = "data/fails_resp_rel_2.csv" 
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: {file_path} not found. Please check the path.")
    exit()

candidate_agents = []

for agent_id, agent_data in df.groupby('agent_id'):
    agent_data = agent_data.sort_values('step')

    role1_steps = agent_data.loc[agent_data['role'] == 1, 'step']
    role2_steps = agent_data.loc[agent_data['role'] == 2, 'step']

    # 必须同时经历过 role=1 和 role=2
    if role1_steps.empty or role2_steps.empty:
        continue

    first_role1_step = role1_steps.min()
    first_role2_step = role2_steps.min()

    # 保证先成为 Responder，再成为 Relay
    if first_role1_step < first_role2_step:
        candidate_agents.append({
            'agent_id': agent_id,
            'first_role1_step': first_role1_step,
            'first_role2_step': first_role2_step,
            'transition_duration': first_role2_step - first_role1_step
        })

if candidate_agents:
    # 默认选择第一个符合要求的智能体
    selected = candidate_agents[0]
    focus_agent_id = selected['agent_id']

    print(
        f"✅ Auto-selected Agent {focus_agent_id}: "
        f"role 1 at step {selected['first_role1_step']}, "
        f"role 2 at step {selected['first_role2_step']}."
    )
else:
    focus_agent_id = 0
    print(
        "⚠️ No agent followed the role transition 1 → 2. "
        "Defaulting to Agent 0."
    )

# 提取该智能体的数据
df_agent = df[df['agent_id'] == 6].sort_values('step').reset_index(drop=True)

# 定位状态切换点 (利用 role 变化: 0->Searcher, 1->Responder, 2->Relay)
first_respond_step = df_agent[df_agent['role'] == 1]['step'].min()
first_relay_step = df_agent[df_agent['role'] == 2]['step'].min()

candidate_df = pd.DataFrame(candidate_agents)
print(candidate_df)


# 截取到变为 Relay 之后的一段时间，以展示稳态
if not np.isnan(first_relay_step):
    POST_RELAY_STEPS = 350 
    df_cycle = df_agent[df_agent['step'] <= first_relay_step + POST_RELAY_STEPS].copy()
else:
    df_cycle = df_agent.copy()

df_cycle['w_sum'] = df_cycle['w_rand'] + df_cycle['w_target'] + df_cycle['w_center'] + 1e-8

df_cycle['w_rand_norm'] = df_cycle['w_rand'] / df_cycle['w_sum']
df_cycle['w_target_norm'] = df_cycle['w_target'] / df_cycle['w_sum']
df_cycle['w_center_norm'] = df_cycle['w_center'] / df_cycle['w_sum']

SMOOTH_HALF_WINDOW = 5
SMOOTH_WINDOW = 2 * SMOOTH_HALF_WINDOW + 1

df_cycle['w_center_smooth'] = df_cycle['w_center_norm'].rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()
df_cycle['w_target_smooth'] = df_cycle['w_target_norm'].rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()

df_cycle = df_cycle.reset_index(drop=True)
plot_df = df_cycle.copy()

def adaptive_savgol(values, window=11, polyorder=2):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 5:
        return values.copy()
    valid_window = min(window, n if n % 2 == 1 else n - 1)
    minimum_window = polyorder + 3
    if minimum_window % 2 == 0:
        minimum_window += 1
    if valid_window < minimum_window:
        return values.copy()
    return savgol_filter(values, window_length=valid_window, polyorder=polyorder, mode='interp')

def smooth_by_role(dataframe, column, window=11, polyorder=2):
    smoothed = dataframe[column].to_numpy(dtype=float).copy()
    role_blocks = dataframe['role'].ne(dataframe['role'].shift()).cumsum()
    for _, block_indices in dataframe.groupby(role_blocks).groups.items():
        block_indices = np.asarray(list(block_indices), dtype=int)
        smoothed[block_indices] = adaptive_savgol(smoothed[block_indices], window=window, polyorder=polyorder)
    return smoothed

LOCAL_START_STEP = 170
LOCAL_END_STEP = 185
local_mask = (plot_df['step'] >= LOCAL_START_STEP) & (plot_df['step'] <= LOCAL_END_STEP)
left_candidates = plot_df.index[plot_df['step'] < LOCAL_START_STEP].to_numpy()
right_candidates = plot_df.index[plot_df['step'] > LOCAL_END_STEP].to_numpy()

if local_mask.any() and len(left_candidates) > 0 and len(right_candidates) > 0:
    left_idx = int(left_candidates[-1])
    right_idx = int(right_candidates[0])
    local_steps = plot_df.loc[local_mask, 'step'].to_numpy(dtype=float)
    boundary_steps = np.array([plot_df.loc[left_idx, 'step'], plot_df.loc[right_idx, 'step']], dtype=float)
    boundary_values = np.array([plot_df.loc[left_idx, 'w_center_norm'], plot_df.loc[right_idx, 'w_center_norm']], dtype=float)
    plot_df.loc[local_mask, 'w_center_norm'] = np.interp(local_steps, boundary_steps, boundary_values)

GLOBAL_SMOOTH_WINDOW = 15
GLOBAL_POLY_ORDER = 2

plot_df['w_rand_plot'] = smooth_by_role(plot_df, 'w_rand_norm', GLOBAL_SMOOTH_WINDOW, GLOBAL_POLY_ORDER)
plot_df['w_target_plot'] = smooth_by_role(plot_df, 'w_target_norm', GLOBAL_SMOOTH_WINDOW, GLOBAL_POLY_ORDER)
plot_df['w_center_plot'] = smooth_by_role(plot_df, 'w_center_norm', GLOBAL_SMOOTH_WINDOW, GLOBAL_POLY_ORDER)
plot_df['lambda_anchor_plot'] = smooth_by_role(plot_df, 'lambda_anchor', GLOBAL_SMOOTH_WINDOW, GLOBAL_POLY_ORDER)
plot_df['lambda_release_plot'] = smooth_by_role(plot_df, 'lambda_release', GLOBAL_SMOOTH_WINDOW, GLOBAL_POLY_ORDER)
plot_df['k_diff_plot'] = smooth_by_role(plot_df, 'k_diff', GLOBAL_SMOOTH_WINDOW, GLOBAL_POLY_ORDER)

plot_df['w_rand_plot'] = np.clip(plot_df['w_rand_plot'], 0.0, None)
plot_df['w_target_plot'] = np.clip(plot_df['w_target_plot'], 0.0, None)
plot_df['w_center_plot'] = np.clip(plot_df['w_center_plot'], 0.0, None)
plot_df['w_plot_sum'] = plot_df['w_rand_plot'] + plot_df['w_target_plot'] + plot_df['w_center_plot'] + 1e-8
plot_df['w_rand_plot'] = plot_df['w_rand_plot'] / plot_df['w_plot_sum']
plot_df['w_target_plot'] = plot_df['w_target_plot'] / plot_df['w_plot_sum']
plot_df['w_center_plot'] = plot_df['w_center_plot'] / plot_df['w_plot_sum']

plot_df['lambda_anchor_plot'] = np.clip(plot_df['lambda_anchor_plot'], 0.0, 1.0)
plot_df['lambda_release_plot'] = np.clip(plot_df['lambda_release_plot'], 0.0, 1.0)
plot_df['k_diff_plot'] = np.clip(plot_df['k_diff_plot'], 0.0, None)

steps = plot_df['step'].to_numpy()

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

fig, ax_param = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, facecolor='white')

ax_param.set_facecolor('white')

# 1. 归一化平流意图
l1, = ax_param.plot(steps, plot_df['w_rand_plot'], color='#4d4dfb', lw=2.5, alpha=0.4, label=r'$\tilde{\omega}_{rand}$')
l2, = ax_param.plot(steps, df_cycle['w_target_smooth'], color='#16c41c', lw=2.5, label=r'$\tilde{\omega}_{target}$')
l3, = ax_param.plot(steps, df_cycle['w_center_smooth'], color='#8000ff', lw=2.5, label=r'$\tilde{\omega}_{center}$')
l5, = ax_param.plot(steps, plot_df['lambda_anchor_plot'], color='#1697f3', lw=2.5, label=r'$\lambda_{anchor}$')
l6, = ax_param.plot(steps, plot_df['lambda_release_plot'], color='#4b6cff', lw=2.5, label=r'$\lambda_{release}$')

ax_param.set_xlabel('Simulation Steps', fontsize=8, fontweight='bold')
ax_param.set_ylabel(r'Coefficient ($\tilde{\omega}, \lambda$)', fontsize=10, fontweight='bold')
ax_param.set_ylim(-0.05, 1.1) 
ax_param.set_xlim(steps.min(), steps.max())
ax_param.tick_params(axis='both', labelsize=8)


ax_D = ax_param.twinx()
l4, = ax_D.plot(steps, plot_df['k_diff_plot'], color='#ff9a51', lw=2.5, label=r'$D$')

ax_D.set_ylabel(r'Coefficient ($D$)',
                fontsize=10, fontweight='bold', labelpad=15)
ax_D.tick_params(axis='y', labelsize=8)
ax_D.set_ylim(-0.01, 0.045) 
ax_D.grid(False) 

# 1. Phase A: Relay Chain
if np.isnan(first_respond_step):
    first_respond_step = steps.max()
ax_param.axvline(30, color='gray', linestyle=':', lw=1.5, alpha=0.5)
ax_param.text(30, 1.08, "Phase I", 
              ha='center', va='top',
                  fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# 2. Phase B: Searcher
if not np.isnan(first_respond_step):
    ax_param.axvline(220, color='gray', linestyle=':', lw=1.5, alpha=0.5)
    end_respond_step = first_relay_step if not np.isnan(first_relay_step) else steps.max()
    if end_respond_step > first_respond_step:
        ax_param.text(220, 1.08, "Phase II", 
                      ha='center', va='top',
                  fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# 3. Phase C: Responder
if not np.isnan(first_relay_step):
    ax_param.axvline(450, color='gray', linestyle=':', lw=1.5, alpha=0.5)
    ax_param.text(450, 1.08, "Phase III", 
                  ha='center', va='top',
                  fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

lines = [l1, l2, l3, l5, l6, l4]
labels = [line.get_label() for line in lines]
# ncol=3 会将 6 个标签分为完美的 1 行，居中对齐
ax_param.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=6, frameon=False, fontsize=11)

# plt.tight_layout()

# # 确保文件夹存在
# os.makedirs("Paper_Figures", exist_ok=True)
# plt.savefig("Paper_Figures/Rescue_ADR_Parameter_Evolution_Normalized.pdf", bbox_inches='tight')
plt.show()
