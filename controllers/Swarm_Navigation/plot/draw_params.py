import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter

WIDTH_CM = 24.0     # 目标宽度 (厘米)
HEIGHT_CM = 8.0     # 目标高度 (厘米)
FIG_WIDTH = WIDTH_CM / 2.54
FIG_HEIGHT = HEIGHT_CM / 2.54

USE_SMOOTHING = True   

file_path = "data/parameter_variation.csv"
if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
    exit()

df = pd.read_csv(file_path)

df_mean = df.groupby('step').agg({
    'w_flow': 'mean', 
    'w_shape': 'mean', 
    'beta': 'mean', 
    'k_diff': 'mean',
    'pos_x': 'mean'
}).reset_index()

df_mean['w_sum'] = df_mean['w_flow'] + df_mean['w_shape'] + 1e-8
df_mean['w_flow_norm'] = df_mean['w_flow'] / df_mean['w_sum']
df_mean['w_shape_norm'] = df_mean['w_shape'] / df_mean['w_sum']

df_mean['lambda_val'] = (df_mean['beta'] - 1.0) / 4.0

step_gather = df_mean['step'].iloc[10]
step_squeeze = df_mean[df_mean['pos_x'] > -0.3].iloc[0]['step']
step_recovery = df_mean[df_mean['pos_x'] > 0.65].iloc[0]['step']
step_goal = df_mean['step'].iloc[-20]

key_steps = [int(step_gather) + 80, int(step_squeeze), int(step_recovery) + 30, int(step_goal)- 30]
phase_labels = ["Phase I", "Phase II", 
                "Phase III", "Phase IV"]

if USE_SMOOTHING:
    cols_to_smooth = ['w_flow_norm', 'w_shape_norm', 'lambda_val', 'k_diff']
    for col in cols_to_smooth:
        smooth_light = df_mean[col].rolling(window=15, min_periods=1, center=True).mean()
        
        smooth_heavy = df_mean[col].rolling(window=15, min_periods=1, center=True).mean()

        df_mean[col] = np.where(
            (df_mean['step'] > 150) & (df_mean['step'] < 250),
            smooth_heavy,
            smooth_light
        )

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

fig, ax_param = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, facecolor='white')

steps = df_mean['step']

l1, = ax_param.plot(steps, df_mean['w_flow_norm'], color='#16c41c', lw=2.5, 
                   label=r'$\tilde{\omega}_{flow}$ ')
l2, = ax_param.plot(steps, df_mean['w_shape_norm'], color='#4d4dfb', lw=2.5,
                   label=r'$\tilde{\omega}_{shape}$ ')
l3, = ax_param.plot(steps, df_mean['lambda_val'], color='#1697f3', lw=2.5, 
                    label=r'$\lambda$ ')

ax_param.set_xlabel('Simulation Steps', fontsize=8, fontweight='bold')
ax_param.set_ylabel(r'Coefficient ($\tilde{\omega}, \lambda$)', fontsize=10, fontweight='bold')
ax_param.set_ylim(-0.05, 1.1) 
ax_param.set_xlim(steps.min(), steps.max())
ax_param.tick_params(axis='both', labelsize=8)

# ==========================================
# 右轴：单独绘制扩散项 D
# ==========================================
ax_D = ax_param.twinx()
l4, = ax_D.plot(steps, df_mean['k_diff'], color='#ff9a51', lw=2.5,
                label=r'$D$')

# 同步将右轴刻度、标签文字颜色与曲线颜色统一，保证图面严谨
ax_D.set_ylabel(r'Coefficient ($D$)', 
                fontsize=10, fontweight='bold', labelpad=15)
ax_D.tick_params(axis='y', labelsize=8)
ax_D.set_ylim(-0.01, 0.045) 
ax_D.grid(False) 

ax_D.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


for i, target_step in enumerate(key_steps):
    ax_param.axvline(target_step, color='gray', linestyle=':', lw=1.5, alpha=0.5)
    ax_param.text(target_step, 1.08, phase_labels[i], ha='center', va='top',
                  fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

ax_param.legend(handles=[l1, l2, l3, l4], loc='lower center', 
                bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=11)

plt.tight_layout()
os.makedirs("Paper_Figures", exist_ok=True)
plt.savefig("Paper_Figures/ADR_Parameter_Evolution_Normalized.pdf", bbox_inches='tight')
# plt.show()

print(f"✅ Parameter evolution plot saved as PDF with custom size: {FIG_WIDTH:.2f} x {FIG_HEIGHT:.2f} inches.")