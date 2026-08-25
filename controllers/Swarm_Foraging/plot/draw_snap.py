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

# --- 1. 读取数据与目标步数 ---
file_path = 'data/fails_0_1.csv'  
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: Could not find {file_path}.")
    exit()

TARGET_STEP = 1000  # 指定要截取的步数
num_agents = df['agent_id'].nunique()

# --- 2. 创建专属画布 (适配[2, 1] 的空间比例) ---
figsize=(8, 4) # 完美对应 2:1 的物理长宽比
# fig, ax_map = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
fig, ax_map = plt.subplots(figsize=figsize, dpi=300)

# 设置无障碍空间的 2x1m 坐标范围 (假设以原点为中心)
ax_map.set_xlim(-1.0, 1.0)
ax_map.set_ylim(-0.52, 0.52)
ax_map.set_aspect('equal')
ax_map.set_facecolor('#fdfdfd')
ax_map.set_xlabel("x position (m)", fontweight='bold', fontsize=10)
ax_map.set_ylabel("y position (m)", fontweight='bold', fontsize=10)
ax_map.tick_params(axis='both', labelsize=10)

history_df = df[
    df['step'] <= TARGET_STEP
].copy()

# 对每个机器人取目标步之前的最后一条记录
curr_df = (
    history_df
    .sort_values(['agent_id', 'step'])
    .groupby('agent_id', as_index=False)
    .tail(1)
    .copy()
)

print(
    curr_df[
        ['agent_id', 'step', 'is_alive']
    ].sort_values('agent_id')
)

# --- 3. 绘制静态物理环境 (巢穴与食物) ---
if len(curr_df) > 0:
    # 绘制巢穴
    if 'nest_x' in curr_df.columns:
        nest_x = curr_df.iloc[0]['nest_x']
        nest_y = curr_df.iloc[0]['nest_y']
        ax_map.scatter([nest_x], [nest_y], marker='s', s=350, color='mediumpurple', edgecolors='k', lw=1.5, zorder=8, label='Nest')
    
    # 自动检索并绘制所有食物点
    target_cols = [c for c in curr_df.columns if c.startswith('target_') and c.endswith('_x')]
    for i, t_col in enumerate(target_cols):
        t_idx = t_col.split('_')[1]
        tx = curr_df.iloc[0][t_col]
        ty = curr_df.iloc[0][f'target_{t_idx}_y']
        
        # 仅标注第一个图例，避免重复
        label = 'Resource' if i == 0 else ""
        ax_map.scatter([tx], [ty], marker='*', s=400, color='gold', edgecolors='k', lw=1.5, zorder=9, label=label)

# --- 4.分段绘制轨迹 (融合存活状态检测) ---
for i in range(num_agents):
    sub_df = df[(df['agent_id'] == i) & (df['step'] <= TARGET_STEP)].sort_values('step')
    if len(sub_df) == 0: continue
    
    # 检测该智能体在当前截断步时是否存活
    is_alive_at_end = sub_df.iloc[-1]['is_alive'] == 1
    
    x = sub_df['pos_x'].values
    y = sub_df['pos_y'].values
    is_carry = sub_df['is_carrying'].values
    
    if is_alive_at_end:
        # 活着：按照搬运状态(crimson)和探索状态(gray)交替分段绘制轨迹
        change_indices = np.where(is_carry[:-1] != is_carry[1:])[0] + 1
        splits = np.concatenate(([0], change_indices, [len(is_carry)]))
        
        for j in range(len(splits) - 1):
            start_idx = splits[j]
            end_idx = splits[j+1]
            
            # 为了防止不同状态的轨迹之间出现视觉断层(Gap)，向后多连一个点
            plot_end_idx = end_idx + 1 if end_idx < len(is_carry) else end_idx
            state = is_carry[start_idx]
            
            if state == 1:  # 搬运状态 (Return)
                ax_map.plot(x[start_idx:plot_end_idx], y[start_idx:plot_end_idx], 
                            lw=1.5, color='crimson', alpha=0.4, zorder=2)
            else:           # 探索状态 (Search)
                ax_map.plot(x[start_idx:plot_end_idx], y[start_idx:plot_end_idx], 
                            lw=0.8, color='gray', alpha=0.25, zorder=1)
    else:
        ax_map.plot(x, y, lw=0.8, color='gray', alpha=0.15, zorder=1)

# --- 5. 绘制智能体在目标步数的最终位置点 ---
if len(curr_df) > 0:
    alive_mask = curr_df['is_alive'] == 1
    dead_mask = ~alive_mask

    # 绘制存活的智能体
    if alive_mask.any():
        ax_map.scatter(curr_df[alive_mask]['pos_x'], curr_df[alive_mask]['pos_y'], 
                       s=100, c='dodgerblue', edgecolors='k', zorder=10, label='Benign Robot')
        
    # 绘制故障/断电的智能体
    if dead_mask.any():
        ax_map.scatter(curr_df[dead_mask]['pos_x'], curr_df[dead_mask]['pos_y'], 
                       s=100, c='crimson', edgecolors='k', zorder=11, label='Failed Robot')
        ax_map.scatter(curr_df[dead_mask]['pos_x'], curr_df[dead_mask]['pos_y'], 
                       s=40, color='white', marker='x', linewidths=2, zorder=12)
        
# --- 6. 整理图例与导出 PDF ---
handles, labels = ax_map.get_legend_handles_labels()

if 'Failed Robot' not in labels:
    failed_circle = Line2D(
        [],
        [],
        linestyle='None',
        marker='o',
        markersize=9,
        markerfacecolor='crimson',
        markeredgecolor='black',
        markeredgewidth=1.0,
    )

    failed_cross = Line2D(
        [],
        [],
        linestyle='None',
        marker='x',
        markersize=6,
        color='white',
        markeredgewidth=1.8,
    )

    # 使用元组将圆圈和叉叠加为一个图例符号
    failed_robot_handle = (
        failed_circle,
        failed_cross,
    )

    handles.append(failed_robot_handle)
    labels.append('Failed Robot')

# ax_map.legend(
#     handles=handles,
#     labels=labels,
#     handler_map={
#         tuple: HandlerTuple(
#             ndivide=1,
#             pad=0,
#         )
#     },
#     loc='upper left',
#     bbox_to_anchor=(0.01, 0.3),
#     framealpha=0.9,
#     edgecolor='black',
#     labelspacing=1.0,
#     fontsize=10,
#     markerscale=0.8,
# )

plt.tight_layout()

# 确保输出文件夹存在
# os.makedirs("Fails_Figures", exist_ok=True)
# output_filename = f"Fails_Figures/fails_4.pdf"

os.makedirs("Paper_Figures", exist_ok=True)
output_filename = f"Paper_Figures/snap.pdf"

plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
print(f"✅ Success! High-quality PDF saved to: {output_filename}")

# plt.show()








# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # ==========================================
# # 🌟 全局科研绘图规范设置 (保证PDF文字可编辑)
# # ==========================================
# plt.rcParams['pdf.fonttype'] = 42 
# plt.rcParams['ps.fonttype'] = 42
# plt.rcParams['axes.linewidth'] = 1.2

# # --- 1. 读取数据与目标步数 ---
# file_path = 'data/expand_3_16.csv'  # 替换为你的觅食数据文件路径
# try:
#     df = pd.read_csv(file_path)
# except FileNotFoundError:
#     print(f"Error: Could not find {file_path}.")
#     exit()

# TARGET_STEP = 900  # 指定要截取的步数
# num_agents = df['agent_id'].nunique()

# # --- 2. 创建专属画布 (适配[2, 1] 的空间比例) ---
# # figsize=(10, 5) 完美对应 2:1 的物理长宽比
# fig, ax_map = plt.subplots(figsize=(10, 5), dpi=300)

# # 设置无障碍空间的 2x1m 坐标范围 (假设以原点为中心)
# ax_map.set_xlim(-2.0, 2.0)
# ax_map.set_ylim(-1, 1)
# ax_map.set_aspect('equal')
# ax_map.set_facecolor('#fdfdfd')
# ax_map.set_xlabel("x position (m)", fontweight='bold', fontsize=12)
# ax_map.set_ylabel("y position (m)", fontweight='bold', fontsize=12)

# curr_df = df[df['step'] == TARGET_STEP]

# # --- 3. 绘制静态物理环境 (巢穴与食物) ---
# if len(curr_df) > 0:
#     # 绘制巢穴
#     if 'nest_x' in curr_df.columns:
#         nest_x = curr_df.iloc[0]['nest_x']
#         nest_y = curr_df.iloc[0]['nest_y']
#         ax_map.scatter([nest_x], [nest_y], marker='s', s=350, color='mediumpurple', edgecolors='k', lw=1.5, zorder=8, label='Nest')
    
#     # 自动检索并绘制所有食物点
#     target_cols = [c for c in curr_df.columns if c.startswith('target_') and c.endswith('_x')]
#     for i, t_col in enumerate(target_cols):
#         t_idx = t_col.split('_')[1]
#         tx = curr_df.iloc[0][t_col]
#         ty = curr_df.iloc[0][f'target_{t_idx}_y']
        
#         # 仅标注第一个图例，避免重复
#         label = 'Food Target' if i == 0 else ""
#         ax_map.scatter([tx], [ty], marker='*', s=400, color='gold', edgecolors='k', lw=1.5, zorder=9, label=label)

# # --- 4. 🌟 分段绘制轨迹 (融合存活状态检测) ---
# for i in range(num_agents):
#     sub_df = df[(df['agent_id'] == i) & (df['step'] <= TARGET_STEP)].sort_values('step')
#     if len(sub_df) == 0: continue
    
#     # 🌟 检测该智能体在当前截断步时是否存活
#     is_alive_at_end = sub_df.iloc[-1]['is_alive'] == 1
    
#     x = sub_df['pos_x'].values
#     y = sub_df['pos_y'].values
#     is_carry = sub_df['is_carrying'].values
    
#     if is_alive_at_end:
#         # 活着：按照搬运状态(crimson)和探索状态(gray)交替分段绘制轨迹
#         change_indices = np.where(is_carry[:-1] != is_carry[1:])[0] + 1
#         splits = np.concatenate(([0], change_indices, [len(is_carry)]))
        
#         for j in range(len(splits) - 1):
#             start_idx = splits[j]
#             end_idx = splits[j+1]
            
#             # 为了防止不同状态的轨迹之间出现视觉断层(Gap)，向后多连一个点
#             plot_end_idx = end_idx + 1 if end_idx < len(is_carry) else end_idx
#             state = is_carry[start_idx]
            
#             if state == 1:  # 搬运状态 (Return)
#                 ax_map.plot(x[start_idx:plot_end_idx], y[start_idx:plot_end_idx], 
#                             lw=1.5, color='crimson', alpha=0.4, zorder=2)
#             else:           # 探索状态 (Search)
#                 ax_map.plot(x[start_idx:plot_end_idx], y[start_idx:plot_end_idx], 
#                             lw=0.8, color='gray', alpha=0.25, zorder=1)
#     else:
#         # 🌟 死了：不进行状态切割，直接绘制整条褪色灰色轨迹，代表该智能体已失效
#         ax_map.plot(x, y, lw=0.8, color='gray', alpha=0.15, zorder=1)

# # --- 5. 🌟 绘制智能体在目标步数的最终位置点 (分活/死进行可视化) ---
# if len(curr_df) > 0:
#     alive_mask = curr_df['is_alive'] == 1
#     dead_mask = ~alive_mask

#     # 绘制存活的智能体
#     if alive_mask.any():
#         ax_map.scatter(curr_df[alive_mask]['pos_x'], curr_df[alive_mask]['pos_y'], 
#                        s=100, c='dodgerblue', edgecolors='k', zorder=10, label='Benign Robot')
        
#     # 绘制故障/断电的智能体 (外圈猩红，内含白色 X)
#     if dead_mask.any():
#         ax_map.scatter(curr_df[dead_mask]['pos_x'], curr_df[dead_mask]['pos_y'], 
#                        s=100, c='crimson', edgecolors='k', zorder=11, label='Failed Robot')
#         ax_map.scatter(curr_df[dead_mask]['pos_x'], curr_df[dead_mask]['pos_y'], 
#                        s=40, color='white', marker='x', linewidths=2, zorder=12)

# # --- 6. 整理图例与导出 PDF ---
# # 如果你想在图中直接展示这些分类图例，可以取消下面的注释
# ax_map.legend(
#     loc='upper left', 
#     bbox_to_anchor=(0.01, 1.0),  # 🌟 1. 精确微调图例位置，使其从顶格向下移动一部分
#     framealpha=0.9, 
#     edgecolor='black', 
#     labelspacing=1.0,     # 同步调整
#     fontsize=10,
#     markerscale=0.8               # 🌟 2. 整体缩放图例中所有图标的尺寸（例如将450的大星标缩到接近正常圆形的大小）
# )

# plt.tight_layout()

# # 确保输出文件夹存在
# os.makedirs("Expand_Figures", exist_ok=True)
# output_filename = f"Expand_Figures/expand_16.pdf"

# # bbox_inches='tight' 能够完美切除白边
# plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
# print(f"✅ Success! High-quality PDF saved to: {output_filename}")

# plt.show()