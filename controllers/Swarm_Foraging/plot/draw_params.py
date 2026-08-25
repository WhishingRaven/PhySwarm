import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import savgol_filter

WIDTH_CM = 24.0     # 目标宽度 (厘米)
HEIGHT_CM = 9.0     # 目标高度 (厘米)
FIG_WIDTH = WIDTH_CM / 2.54
FIG_HEIGHT = HEIGHT_CM / 2.54

file_path = "data/fails_0_4.csv" 
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: {file_path} not found. Please check the path.")
    exit()

df_agent = df[df['agent_id'] == 5].sort_values('step').reset_index(drop=True)

state_diff = df_agent['is_carrying'].diff()
pickup_steps = df_agent[state_diff == 1]['step'].values
drop_steps = df_agent[state_diff == -1]['step'].values


POST_DROP_STEPS = 50

# 截取第一个完整工作周期 (出发 -> 捡起 -> 放下 -> 重新出发一小段)
if len(pickup_steps) > 0 and len(drop_steps) > 0:
    first_pickup = pickup_steps[0]
    valid_drops = drop_steps[drop_steps > first_pickup]
    if len(valid_drops) > 0:
        first_drop = valid_drops[0]
        df_cycle = df_agent[(df_agent['step'] <= first_drop + POST_DROP_STEPS)].copy()
    else:
        df_cycle = df_agent.copy()
else:
    df_cycle = df_agent.copy()

df_cycle['w_sum'] = df_cycle['w_rand'] + df_cycle['w_info'] + df_cycle['w_food'] + df_cycle['w_nest'] + 1e-8

df_cycle['w_rand_norm'] = df_cycle['w_rand'] / df_cycle['w_sum']
df_cycle['w_info_norm'] = df_cycle['w_info'] / df_cycle['w_sum']
df_cycle['w_food_norm'] = df_cycle['w_food'] / df_cycle['w_sum']
df_cycle['w_nest_norm'] = df_cycle['w_nest'] / df_cycle['w_sum']

steps = df_cycle['step'].values

omega_columns = [
    'w_rand_norm',
    'w_info_norm',
    'w_food_norm',
    'w_nest_norm',
]

# 原始归一化权重矩阵，形状为 [num_steps, 4]
omega_raw = df_cycle[omega_columns].to_numpy(dtype=float)

# 绘图使用的副本
omega_plot = omega_raw.copy()

SMOOTH_END_STEP = 120
SMOOTH_WINDOW = 31      # 可改为 11、15、21
SMOOTH_POLYORDER = 2
BLEND_POINTS = 10   

smooth_mask = steps <= SMOOTH_END_STEP
smooth_count = np.sum(smooth_mask)

if smooth_count >= 7:
    omega_segment = omega_raw[smooth_mask].copy()

    # Savitzky-Golay 窗口必须为奇数
    window_length = min(
        SMOOTH_WINDOW,
        smooth_count if smooth_count % 2 == 1 else smooth_count - 1,
    )

    if window_length <= SMOOTH_POLYORDER:
        window_length = SMOOTH_POLYORDER + 3

        if window_length % 2 == 0:
            window_length += 1

    omega_smooth = savgol_filter(
        omega_segment,
        window_length=window_length,
        polyorder=SMOOTH_POLYORDER,
        axis=0,
        mode='interp',
    )

    # 防止滤波产生负值
    omega_smooth = np.clip(
        omega_smooth,
        0.0,
        None,
    )

    omega_smooth_sum = omega_smooth.sum(
        axis=1,
        keepdims=True,
    )

    omega_smooth = (
        omega_smooth
        / np.maximum(omega_smooth_sum, 1e-8)
    )

    blend_points = min(
        BLEND_POINTS,
        len(omega_smooth),
    )

    blend_ratio = np.linspace(
        0.0,
        1.0,
        blend_points,
    )[:, np.newaxis]

    omega_smooth[-blend_points:] = (
        (1.0 - blend_ratio)
        * omega_smooth[-blend_points:]
        + blend_ratio
        * omega_segment[-blend_points:]
    )

    omega_smooth = np.clip(
        omega_smooth,
        0.0,
        None,
    )

    omega_smooth /= np.maximum(
        omega_smooth.sum(axis=1, keepdims=True),
        1e-8,
    )

    omega_plot[smooth_mask] = omega_smooth

# 分别提取四条用于绘图的曲线
w_rand_values = omega_plot[:, 0]
w_info_values = omega_plot[:, 1]
w_food_values = omega_plot[:, 2]
w_nest_values = omega_plot[:, 3]

# 检查归一化约束
print(
    "Maximum omega-sum error:",
    np.max(
        np.abs(
            omega_plot.sum(axis=1) - 1.0
        )
    )
)


plt.rcParams['pdf.fonttype'] = 42

# 强制全局和局部的背景全部为纯白
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

fig, ax_param = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, facecolor='white')
ax_param.set_facecolor('white')

l1, = ax_param.plot(
    steps,
    w_rand_values,
    color='#ff2200',
    lw=2.5,
    label=r'$\tilde{\omega}_{rand}$',
    zorder=4,
)

l2, = ax_param.plot(
    steps,
    w_info_values,
    color='#8000ff',
    lw=2.5,
    label=r'$\tilde{\omega}_{info}$',
    zorder=3,
)

l3, = ax_param.plot(
    steps,
    w_food_values,
    color='#16c41c',
    lw=2.5,
    label=r'$\tilde{\omega}_{food}$',
    zorder=3,
)

l4, = ax_param.plot(
    steps,
    w_nest_values,
    color='#3dd5e9',
    lw=2.5,
    label=r'$\tilde{\omega}_{nest}$',
    zorder=3,
)

# 绘制反应项
l6, = ax_param.plot(steps, df_cycle['lambda_pick'], color='#4b6cff', lw=2.5, label=r'$\lambda_{pick}$', zorder=3)
l7, = ax_param.plot(steps, df_cycle['lambda_drop'], color='#1697f3', lw=2.5, label=r'$\lambda_{drop}$', zorder=3)

ax_param.set_xlabel('Simulation Steps', fontsize=8, fontweight='bold')
ax_param.set_ylabel(r'Coefficient ($\tilde{\omega}, \lambda$)', fontsize=10, fontweight='bold')
ax_param.set_ylim(-0.05, 1.1) 
ax_param.set_xlim(steps.min(), steps.max())
ax_param.tick_params(axis='both', labelsize=8)

# ==========================================
# 右轴：单独绘制扩散项 D (实线)
# ==========================================
ax_D = ax_param.twinx()
l5, = ax_D.plot(steps, df_cycle['k_diff'], color='#ff9a51', lw=2.5, label=r'$D$')

# 同步将右轴刻度、标签文字颜色与曲线颜色统一，保证图面严谨
ax_D.set_ylabel(r'Coefficient ($D$)', fontsize=10, fontweight='bold', labelpad=15)
ax_D.tick_params(axis='y', labelsize=8)
ax_D.set_ylim(-0.01, 0.045) # 维持 D 真实的微小物理量纲
ax_D.grid(False) 

# 强制右侧 Y 轴（D轴）刻度保留两位小数
ax_D.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# ==========================================
# 3. 物理相位精准标注
# ==========================================
if len(pickup_steps) > 0:
    ax_param.axvline(23, color='gray', linestyle=':', lw=1.5, alpha=0.5)
    ax_param.text(23, 1.08, "Phase I", ha='center', va='top',
                  fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

if 'first_drop' in locals():
    ax_param.axvline(140, color='gray', linestyle=':', lw=1.5, alpha=0.5)
    ax_param.text(140, 1.08, "Phase II", ha='center', va='top',
                  fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax_param.axvline(185, color='gray', linestyle=':', lw=1.5, alpha=0.5)
    ax_param.text(185, 1.08, "Phase III", ha='center', va='top',
                  fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax_param.axvline(240, color='gray', linestyle=':', lw=1.5, alpha=0.5)
    ax_param.text(240, 1.08, "Phase IV", ha='center', va='top',
                  fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# ==========================================
# 4. 图例整理
# ==========================================
lines = [l1, l2, l3, l4, l6, l7, l5]
labels = [line.get_label() for line in lines]

ax_param.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=7, frameon=False, fontsize=11)

os.makedirs("Paper_Figures", exist_ok=True)
plt.tight_layout()
# plt.savefig("Paper_Figures/ADR_Parameter_Evolution.pdf", bbox_inches='tight')
plt.show()

print("✅ Segmented rendering plot generated successfully! w_rand underlays seamlessly when close to zero.")