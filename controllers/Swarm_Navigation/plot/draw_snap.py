import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

WIDTH_CM = 12.0     # 目标宽度 (厘米)
HEIGHT_CM = 10    # 目标高度 (厘米)
FIG_WIDTH = WIDTH_CM / 2.54
FIG_HEIGHT = HEIGHT_CM / 2.54

plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.linewidth'] = 1.2

file_path = 'data/fails_0_3.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: Could not find {file_path}.")
    exit()

TARGET_STEP = 480 
num_agents = df['agent_id'].nunique()

fig, ax_map = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)

ax_map.set_xlim(-1.5, 1.5)
ax_map.set_ylim(-0.6, 0.6)
ax_map.set_aspect('equal')
ax_map.set_facecolor('#fdfdfd')
ax_map.set_xlabel("x position (m)", fontweight='bold', fontsize=10)
ax_map.set_ylabel("y position (m)", fontweight='bold', fontsize=10)

ax_map.tick_params(axis='both', labelsize=8)

walls_x =[-0.7, -0.1, 0.6, 0.9]
walls_y_upper =[0.6, 0.12, 0.12, 0.6]
walls_y_lower =[-0.6, -0.12, -0.12, -0.6]

ax_map.plot(walls_x, walls_y_upper, color='#444444', lw=3)
ax_map.plot(walls_x, walls_y_lower, color='#444444', lw=3)
ax_map.fill_between(walls_x, walls_y_upper, 0.6, color='gray', alpha=0.2)
ax_map.fill_between(walls_x, -0.6, walls_y_lower, color='gray', alpha=0.2)

ax_map.scatter([1.05], [0.0], marker='*', s=200, color='gold', edgecolors='k', lw=1.0, zorder=9, label='Navigation Target')

cmap = plt.get_cmap("tab10") 

for i in range(num_agents):
    sub_df = df[(df['agent_id'] == i) & (df['step'] <= TARGET_STEP)].sort_values('step')
    if len(sub_df) == 0: continue
    
    is_alive_at_end = sub_df.iloc[-1]['is_alive'] == 1
    alpha_val = 0.5 if is_alive_at_end else 0.2
    color_val = cmap(i % 10) if is_alive_at_end else 'gray'
    
    ax_map.plot(sub_df['pos_x'], sub_df['pos_y'], lw=1.5, alpha=alpha_val, color=color_val, zorder=2)

curr_df = df[df['step'] == TARGET_STEP]
alive_mask = curr_df['is_alive'] == 1
dead_mask = ~alive_mask

if alive_mask.any():
    ax_map.scatter(
        curr_df[alive_mask]['pos_x'],
        curr_df[alive_mask]['pos_y'],
        s=80,
        c='dodgerblue',
        edgecolors='k',
        zorder=10,
        label='Benign Robot'
    )

if dead_mask.any():
    ax_map.scatter(
        curr_df[dead_mask]['pos_x'],
        curr_df[dead_mask]['pos_y'],
        s=100,
        c='crimson',
        edgecolors='k',
        zorder=11,
        label='Failed Robot'
    )

    ax_map.scatter(
        curr_df[dead_mask]['pos_x'],
        curr_df[dead_mask]['pos_y'],
        s=40,
        color='white',
        marker='x',
        linewidths=2,
        zorder=12
    )

else:
    ax_map.scatter(
        [],
        [],
        s=100,
        c='crimson',
        edgecolors='k',
        label='Failed Robot'
    )

target_legend = Line2D(
    [],
    [],
    linestyle="None",
    marker="*",
    markersize=12,
    markerfacecolor="gold",
    markeredgecolor="black",
    markeredgewidth=1.0,
)

benign_legend = Line2D(
    [],
    [],
    linestyle="None",
    marker="o",
    markersize=8,
    markerfacecolor="dodgerblue",
    markeredgecolor="black",
    markeredgewidth=1.0,
)

failed_circle_legend = Line2D(
    [],
    [],
    linestyle="None",
    marker="o",
    markersize=9,
    markerfacecolor="crimson",
    markeredgecolor="black",
    markeredgewidth=1.0,
)

failed_cross_legend = Line2D(
    [],
    [],
    linestyle="None",
    marker="x",
    markersize=5,
    color="white",
    markeredgewidth=1.8,
)

failed_legend = (
    failed_circle_legend,
    failed_cross_legend,
)

# ax_map.legend(
#     handles=[
#         target_legend,
#         benign_legend,
#         failed_legend,
#     ],
#     labels=[
#         "Navigation Target",
#         "Benign Robot",
#         "Failed Robot",
#     ],
#     handler_map={
#         tuple: HandlerTuple(
#             ndivide=1,  # 让两个标记重叠，而不是并排
#             pad=0.0,
#         )
#     },
#     loc="upper left",
#     bbox_to_anchor=(0.01, 0.96),
#     framealpha=0.9,
#     edgecolor="black",
#     fontsize=10,
# )

plt.tight_layout()

# 确保输出文件夹存在
# os.makedirs("Fails_Figures", exist_ok=True)
# output_filename = f"Fails_Figures/fails_1.pdf"

os.makedirs("Paper_Figures", exist_ok=True)
output_filename = f"Paper_Figures/snap.pdf"

plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
print(f"✅ Success! High-quality PDF saved to: {output_filename}")

# plt.show()
