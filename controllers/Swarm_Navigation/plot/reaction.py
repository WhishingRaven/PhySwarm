import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'


R0 = 0.2

def get_corridor_geometry(px):
    if px < -0.7: 
        return 0.5
    elif -0.7 <= px < -0.1: 
        return 0.5 + (px + 0.7)/0.6 * (0.12 - 0.5)
    elif -0.1 <= px < 0.6: 
        return 0.12
    elif 0.6 <= px < 0.9:
        return 0.12 + (px - 0.6)/0.3 * (0.5 - 0.12)
    else: 
        return 0.5

def compute_reaction_advection_field(X, Y, task_idx, phase_idx):

    if task_idx == 0: # 任务一：群体觅食
        r_nest = np.array([-0.4, 0.0])
        r_food = np.array([0.4, 0.0])
        
        if phase_idx == 0: 
            U = -0.08 * Y + 0.03 * np.cos(3*X)
            V = 0.08 * (X - r_nest[0]) + 0.03 * np.sin(3*Y)
            rho = np.exp(-5 * ((X - r_nest[0])**2 + (Y - r_nest[1])**2))
        elif phase_idx == 1: 
            dx, dy = r_nest[0] - X, r_nest[1] - Y
            dist = np.sqrt(dx**2 + dy**2) + 1e-6
            U = 0.12 * (dx / dist)
            V = 0.12 * (dy / dist)
            rho = np.exp(-5 * ((X - r_nest[0])**2 + (Y - r_nest[1])**2))
        else: 
            dx, dy = r_food[0] - X, r_food[1] - Y
            dist = np.sqrt(dx**2 + dy**2) + 1e-6
            U = 0.12 * (dx / dist)
            V = 0.12 * (dy / dist)
            rho = np.exp(-5 * ((X - r_food[0])**2 + (Y - r_food[1])**2))
            
    elif task_idx == 1: # 任务二：自适应编队
        centers = [[-0.5, 0.0], [-0.30, 0.0], [0.35, 0.0]]
        w_shapes = [0.12, 0.08, 0.04]
        w_flows = [0.01, 0.05, 0.12]
        betas = [1.0, 2.5, 5.0]
        
        centroid = centers[phase_idx]
        ws, wf, bt = w_shapes[phase_idx], w_flows[phase_idx], betas[phase_idx]
        
        dx, dy = X - centroid[0], Y - centroid[1]
        dw = np.sqrt(dx**2 + bt * dy**2) + 1e-6
        U = -ws * ((dw - R0)/dw) * dx + wf
        V = -ws * ((dw - R0)/dw) * (bt * dy)
        rho = np.exp(-15 * ws * (dw - R0)**2)
        
    else: # 任务三：目标救援
        r_base = np.array([-0.45, -0.2])
        r_target = np.array([0.45, 0.2])
        
        if phase_idx == 0: 
            U = 0.05 * np.cos(4*Y) + 0.02
            V = 0.05 * np.sin(4*X) + 0.02
            rho = np.exp(-5 * ((X - r_base[0])**2 + (Y - r_base[1])**2))
        elif phase_idx == 1:
            u_x = 0.9 / np.sqrt(0.97)
            u_y = 0.4 / np.sqrt(0.97)
            X_rot = X * u_x + Y * u_y
            Y_rot = -X * u_y + Y * u_x
            
            U = -0.05 * (2.0 * X_rot * u_x - 30.0 * Y_rot * u_y)
            V = -0.05 * (2.0 * X_rot * u_y + 30.0 * Y_rot * u_x)
            
            rho = np.exp(-1.5 * X_rot**2 - 35.0 * Y_rot**2)
        else: 
            dx, dy = r_target[0] - X, r_target[1] - Y
            dist = np.sqrt(dx**2 + dy**2) + 1e-6
            U = 0.11 * (dx / dist) + 0.02 * np.cos(4*Y)
            V = 0.11 * (dy / dist) + 0.02 * np.sin(4*X)
            rho = np.exp(-5 * ((X - r_target[0])**2 + (Y - r_target[1])**2))
            
    return U, V, rho

def compute_diffusion_field(X, Y, agent_pos, task_idx, phase_idx):

    if task_idx == 0:
        kd = 0.12 if phase_idx == 0 else 0.03
    elif task_idx == 1:
        kd = 0.03
    else:
        kd = 0.12 if phase_idx == 1 else 0.04 
        
    u_d = np.zeros_like(X)
    v_d = np.zeros_like(Y)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            px, py = X[i,j], Y[i,j]
            
            if task_idx == 1:
                hw = get_corridor_geometry(px)
                dist_top = hw - py
                dist_bottom = py - (-hw)
                if 0 < dist_top < 0.15: v_d[i,j] -= (0.15 - dist_top) * 6.0
                if 0 < dist_bottom < 0.15: v_d[i,j] += (0.15 - dist_bottom) * 6.0
            else:
                dist_top = 0.55 - py
                dist_bottom = py - (-0.55)
                if 0 < dist_top < 0.15: v_d[i,j] -= (0.15 - dist_top) * 5.0
                if 0 < dist_bottom < 0.15: v_d[i,j] += (0.15 - dist_bottom) * 5.0
                    
            for ax_p, ay_p in agent_pos:
                dist_a = np.sqrt((px - ax_p)**2 + (py - ay_p)**2)
                if 0.02 < dist_a < 0.2:
                    u_d[i,j] += ((px - ax_p) / dist_a) * (0.2 - dist_a) * 4.0
                    v_d[i,j] += ((py - ay_p) / dist_a) * (0.2 - dist_a) * 4.0
                    
    return u_d * kd, v_d * kd

FORAGING_AGENTS_POS = np.array([
    [-0.45, 0.05], [-0.38, -0.08], [-0.25, 0.02], [-0.10, -0.05],
    [0.10, 0.05], [0.22, -0.02], [0.35, 0.06], [0.42, -0.04]
])
FORAGING_AGENTS_STATES = [0, 0, 0, 2, 1, 1, 1, 2] # 0: 探索(蓝), 1: 搬运(橙), 2: 靠近(紫)

RESCUE_AGENTS_POS = np.array([
    [-0.40, -0.22],  # 靠近 base 区域
    [-0.32, -0.05],  # 探索态
    [-0.22, -0.15],  # 中继态 1 (探测对象)
    [-0.08, 0.08],   # 响应态
    [0.05, -0.05],   # 中继态 2
    [0.18, 0.12],    # 响应态 (探测对象)
    [0.28, -0.08],   # 中继态 3
    [0.38, 0.18]     # 响应态
])
RESCUE_AGENTS_STATES = [0, 0, 1, 2, 1, 2, 1, 2] # 0: 搜索态(蓝圆), 1: 中继态(绿圆), 2: 响应态(红圆)

def get_perception_agents(task_idx, phase_idx):
    if task_idx == 0:
        probe_indices = [0, 5, 7] 
        return FORAGING_AGENTS_POS, FORAGING_AGENTS_STATES, probe_indices[phase_idx]
        
    elif task_idx == 1:
        cx_list = [-0.5, -0.30, 0.35]
        cx = cx_list[phase_idx]
        if phase_idx == 0:
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            pos = np.column_stack([cx + 0.2*np.cos(angles), 0.0 + 0.2*np.sin(angles)])
        elif phase_idx == 1:
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            pos = np.column_stack([cx + 0.22*np.cos(angles), 0.0 + (0.22/np.sqrt(2.5))*np.sin(angles)])
        else:
            pos = np.column_stack([np.linspace(cx-0.25, cx+0.25, 8), np.zeros(8)])
        return pos, [0]*8, -1 
        
    else:
        probe_indices = [1, 2, 5] 
        return RESCUE_AGENTS_POS, RESCUE_AGENTS_STATES, probe_indices[phase_idx]

def draw_3x3_perception_matrix():
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.15, wspace=0.11)

    
    colormaps = ['YlOrRd', 'Purples', 'Blues']
    stream_colors = [(0.6, 0.4, 0.1, 0.45), (1.0, 0.1, 0.1, 0.4), (0.1, 0.4, 0.8, 0.45)]
    
    agent_colors = [
        {0: '#1F77B4', 1: '#FF7F0E', 2: '#9400D3'}, 
        {0: '#8A2BE2', 1: '#8A2BE2'},               
        {0: '#1F77B4', 1: '#2CA02C', 2: '#D62728'}  
    ]
    
    X, Y = np.meshgrid(np.linspace(-0.8, 0.8, 100), np.linspace(-0.55, 0.55, 80))
    
    for task_idx in range(3): 
        for phase_idx in range(3): 
            ax = fig.add_subplot(gs[task_idx, phase_idx])
            
            # 1. 物理环境渲染
            if task_idx == 1: 
                path = [(-1.6, 0.5), (-0.7, 0.5), (-0.1, 0.12), (1.6, 0.12)]
                px, py = zip(*path)
                ax.fill_between(px, py, 1.0, color='#f2f2f2', zorder=1)
                ax.fill_between(px, [-y for y in py], -1.0, color='#f2f2f2', zorder=1)
                ax.plot(px, py, color='black', lw=1.5, zorder=2)
                ax.plot(px, [-y for y in py], color='black', lw=1.5, zorder=2)
            else: 
                ax.axhline(0.55, color='black', lw=1.5, zorder=2)
                ax.axhline(-0.55, color='black', lw=1.5, zorder=2)
                ax.fill_between([-1.0, 1.0], [0.55, 0.55], 1.0, color='#f8f8f8', zorder=1)
                ax.fill_between([-1.0, 1.0], [-0.55, -0.55], -1.0, color='#f8f8f8', zorder=1)
            
            # 2. 静态地标绘制
            if task_idx == 0: 
                ax.scatter(-0.4, 0.0, marker='s', color='#2CA02C', s=250, zorder=5, label='Nest' if phase_idx==0 else "")
                ax.scatter(0.4, 0.0, marker='*', color='#D4AF37', s=300, zorder=5, label='Resource' if phase_idx==0 else "")
            elif task_idx == 2: 
                ax.scatter(-0.45, -0.2, marker='p', color='#1F77B4', s=250, zorder=5, label='Base Station' if phase_idx==0 else "")
                ax.scatter(0.45, 0.2, marker='H', color='#FF3333', s=250, zorder=5, label='Target' if phase_idx==0 else "") 


            # 3. 提取去中心化智能体与主观探测个体
            agent_pos, states, probe_idx = get_perception_agents(task_idx, phase_idx)

            # 4. 根据当前探测个体的相位感知，渲染平流场与扩散避障矢量
            U, V, rho = compute_reaction_advection_field(X, Y, task_idx, phase_idx)
            u_dif, v_dif = compute_diffusion_field(X, Y, agent_pos, task_idx, phase_idx)

            # 5. 绘制感知势图与主观流线
            ax.contourf(X, Y, rho, levels=14, cmap=colormaps[task_idx], alpha=0.35, zorder=1)
            ax.streamplot(X, Y, U, V, color=stream_colors[task_idx], linewidth=0.8, density=0.8, arrowsize=0.8, zorder=3)
            
            # 6. 绘制主观扩散避障矢量
            mask = (np.arange(X.shape[0]) % 4 == 0)[:, None] & (np.arange(X.shape[1]) % 5 == 0)
            ax.quiver(X[mask], Y[mask], u_dif[mask], v_dif[mask], color='blue', scale=12, width=0.005, zorder=10)

            # 7. 绘制智能体集
            for i, (ax_p, ay_p) in enumerate(agent_pos):
                state_val = states[i]
                color_val = agent_colors[task_idx][state_val]
                marker_val = 'o' 
                size_val = 80
                
                ax.scatter(ax_p, ay_p, color=color_val, marker=marker_val, 
                           edgecolors='black', s=size_val, lw=1.2, zorder=11)
                
                if i == probe_idx:
                    ax.scatter(ax_p, ay_p, facecolors='none', edgecolors='#FF0000', 
                               s=size_val*4.5, lw=3.0, linestyle='-', zorder=12, label="Probe Robot")

            # 8. 视窗轴线设定
            ax.set_xlim(-0.75, 0.75) 
            ax.set_ylim(-0.52, 0.52)
            ax.set_aspect('equal')
            
            for spine in ax.spines.values():
                spine.set_color('#CCCCCC')
                spine.set_linewidth(1.0)
            ax.set_xticks([])
            ax.set_yticks([])

            # 9. 标题：聚焦于“探测个体主观相位”的描述形式
            if task_idx == 0:
                p_labels = [r"$\lambda_{pick}=0, \lambda_{drop}=0$",
                            r"$\lambda_{pick}=1, \lambda_{drop}=0$",
                            r"$\lambda_{pick}=0, \lambda_{drop}=1$"]
                ax.set_title(p_labels[phase_idx], fontsize=11, family='monospace', pad=6)
            elif task_idx == 1:
                p_labels = [r"$\lambda_{morph} \approx 0.00$",
                            r"$\lambda_{morph} \approx 0.38$",
                            r"$\lambda_{morph} \approx 1.00$"]
                ax.set_title(p_labels[phase_idx], fontsize=11, family='monospace', pad=6 )
            else:
                p_labels = [r"$\lambda_{anchor}=0, \lambda_{release}=0$ ",
                            r"$\lambda_{anchor}=1$ ",
                            r"$\lambda_{release}=1$"]
                ax.set_title(p_labels[phase_idx], fontsize=11, family='monospace', pad=6,)


            handles, labels_list = ax.get_legend_handles_labels()
            if len(handles) > 0:
                ax.legend(
                    loc='upper right', 
                    fontsize=10, 
                    framealpha=0.8, 
                    edgecolor='#EAEAEA', 
                    handletextpad=0.6,  
                    labelspacing=1.2,  
                    borderpad=0.6     
                )

    # 保存为 PDF
    os.makedirs("Paper_Figures", exist_ok=True)
    plt.savefig("Paper_Figures/Three_Tasks_Reaction_Fields.pdf", format='pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    draw_3x3_perception_matrix()