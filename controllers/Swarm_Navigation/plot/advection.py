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

def compute_foraging_field(X, Y, phase):
    """任务一：群体觅食平流场"""
    r_nest = np.array([-0.4, 0.0])
    r_food = np.array([0.4, 0.0])
    
    if phase == 1: 
        U = -0.08 * (Y) + 0.03 * np.cos(3*X)
        V = 0.08 * (X - r_nest[0]) + 0.03 * np.sin(3*Y)
        rho = np.exp(-5 * ((X - r_nest[0])**2 + (Y - r_nest[1])**2))
    elif phase == 2:
        dx, dy = r_food[0] - X, r_food[1] - Y
        dist = np.sqrt(dx**2 + dy**2) + 1e-6
        U = 0.12 * (dx / dist)
        V = 0.12 * (dy / dist)
        rho = np.exp(-5 * ((X - r_food[0])**2 + (Y - r_food[1])**2))
    else: 
        dx, dy = r_nest[0] - X, r_nest[1] - Y
        dist = np.sqrt(dx**2 + dy**2) + 1e-6
        U = 0.12 * (dx / dist)
        V = 0.12 * (dy / dist)
        rho = np.exp(-5 * ((X - r_nest[0])**2 + (Y - r_nest[1])**2))
        
    return U, V, rho

def compute_formation_field(X, Y, centroid, ws, wf, bt):
    """任务二：自适应编队平流场"""
    dx, dy = X - centroid[0], Y - centroid[1]
    dw = np.sqrt(dx**2 + bt * dy**2) + 1e-6
    U = -ws * ((dw - R0)/dw) * dx + wf
    V = -ws * ((dw - R0)/dw) * (bt * dy)
    rho = np.exp(-15 * ws * (dw - R0)**2)
    return U, V, rho

def compute_rescue_field(X, Y, phase):
    """任务三：目标救援平流场"""
    r_base = np.array([-0.45, -0.2])
    r_target = np.array([0.45, 0.2])
    r_center = (r_base + r_target) / 2.0
    
    if phase == 1: 
        U = 0.05 * np.cos(4*Y) + 0.02
        V = 0.05 * np.sin(4*X) + 0.02
        rho = np.exp(-5 * ((X - r_base[0])**2 + (Y - r_base[1])**2)) 
    elif phase == 2:
        dx_t, dy_t = r_target[0] - X, r_target[1] - Y
        dist_t = np.sqrt(dx_t**2 + dy_t**2) + 1e-6
        
        U = 0.04 * (dx_t / dist_t) - 0.10 * (X - r_center[0])
        V = 0.04 * (dy_t / dist_t) - 0.10 * (Y - r_center[1])
        
        dist_to_line = np.abs((r_target[1]-r_base[1])*X - (r_target[0]-r_base[0])*Y) / np.sqrt(0.9**2 + 0.4**2)
        rho = np.exp(-16 * dist_to_line**2) * (np.exp(-3 * ((X - r_center[0])**2 + (Y - r_center[1])**2)) + 0.3)
    else: 
        r_target_new = np.array([0.45, -0.2])
        dx, dy = r_target_new[0] - X, r_target_new[1] - Y
        dist = np.sqrt(dx**2 + dy**2) + 1e-6
        U = 0.11 * (dx / dist) + 0.02 * np.cos(4*Y)
        V = 0.11 * (dy / dist) + 0.02 * np.sin(4*X)
        rho = np.exp(-5 * ((X - r_target_new[0])**2 + (Y - r_target_new[1])**2))
        
    return U, V, rho

def get_synthetic_agents(task_idx, row_idx):
    if task_idx == 0: 
        if row_idx == 0: 
            return np.array([
                [-0.45, 0.05], [-0.38, -0.08], [-0.42, 0.12], [-0.32, 0.02],
                [-0.48, -0.04], [-0.33, 0.10], [-0.46, -0.10], [-0.31, -0.06]
            ])
        elif row_idx == 1: 
            return np.array([
                [-0.30, 0.03], [-0.20, -0.01], [-0.10, 0.02], [0.00, -0.02],
                [0.10, 0.01], [0.20, -0.03], [0.30, 0.02], [0.38, -0.01]
            ])
        else: 
            return np.array([
                [0.30, -0.02], [0.20, 0.01], [0.10, -0.03], [0.00, 0.02],
                [-0.10, -0.01], [-0.20, 0.03], [-0.30, -0.02], [-0.38, 0.01]
            ])
            
    elif task_idx == 1: 
        if row_idx == 0: 
            cx, cy = -0.5, 0.0
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            return np.column_stack([cx + 0.2*np.cos(angles), cy + 0.2*np.sin(angles)])
        elif row_idx == 1: 
            cx, cy = -0.30, 0.0 
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            return np.column_stack([cx + 0.22*np.cos(angles), cy + (0.22/np.sqrt(2.5))*np.sin(angles)])
        else: 
            cx, cy = 0.35, 0.0
            return np.column_stack([np.linspace(cx-0.25, cx+0.25, 8), np.zeros(8)])
            
    else: 
        r_base = np.array([-0.45, -0.2])
        r_target = np.array([0.45, 0.2])
        if row_idx == 0: 
            return np.array([
                [-0.45, -0.20], [-0.40, -0.17], [-0.35, -0.12], [-0.30, -0.08],
                [-0.25, -0.04], [-0.20, 0.00], [-0.38, -0.22], [-0.42, -0.15]
            ])
        elif row_idx == 1: 
            xs = np.linspace(-0.45, 0.45, 8)
            ys = np.linspace(-0.2, 0.2, 8)
            return np.column_stack([xs, ys])
        else: 
            xs = np.linspace(-0.45, 0.45, 8)
            ys = np.linspace(-0.2, -0.2, 8) + 0.1 * np.sin(np.linspace(0, np.pi, 8))
            return np.column_stack([xs, ys])

# ==========================================
# 绘图主程序
# ==========================================
def draw_3x3_advection_matrix():
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.15, wspace=0.11)
    
    colormaps = ['YlOrBr', 'Reds', 'Blues']
    stream_colors = [(0.6, 0.4, 0.1, 0.45), (1.0, 0.1, 0.1, 0.4), (0.1, 0.4, 0.8, 0.45)]
    agent_colors = ['#FFA500', '#FF3333', '#1F77B4']
    
    X, Y = np.meshgrid(np.linspace(-0.8, 0.8, 100), np.linspace(-0.55, 0.55, 80))
    
    for task_idx in range(3): 
        for phase_idx in range(3): 
            ax = fig.add_subplot(gs[task_idx, phase_idx])
            
            # 1. 环境边界绘制
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
            
            # 2. 绘制标志物
            if task_idx == 0: 
                ax.scatter(-0.4, 0.0, marker='s', color='#2CA02C', s=250, zorder=5, label='Nest' if phase_idx==0 else "")
                ax.scatter(0.4, 0.0, marker='*', color='#D4AF37', s=300, zorder=5, label='Resource' if phase_idx==0 else "")
            elif task_idx == 2: 
                ax.scatter(-0.45, -0.2, marker='p', color='#1F77B4', s=250, zorder=5, label='Base Station' if phase_idx==0 else "")
                if phase_idx == 2: 
                    ax.scatter(0.45, -0.2, marker='H', color='#FF3333', s=250, zorder=5, label='New Target' if phase_idx==2 else "")
                    ax.scatter(0.45, 0.2, marker='H', color='gray', s=150, alpha=0.3, zorder=4)
                else:
                    ax.scatter(0.45, 0.2, marker='H', color='#FF3333', s=250, zorder=5, label='Target' if phase_idx==0 else "")

            # 3. 计算物理场
            if task_idx == 0: 
                U, V, rho = compute_foraging_field(X, Y, phase_idx + 1)
            elif task_idx == 1: 
                centers = [[-0.5, 0.0], [-0.30, 0.0], [0.35, 0.0]] 
                w_shapes = [0.12, 0.08, 0.04]
                w_flows = [0.01, 0.05, 0.12]
                betas = [1.0, 2.5, 5.0]
                U, V, rho = compute_formation_field(X, Y, centers[phase_idx], w_shapes[phase_idx], w_flows[phase_idx], betas[phase_idx])
            else: 
                U, V, rho = compute_rescue_field(X, Y, phase_idx + 1)

            # 4. 绘制热力图与流线
            ax.contourf(X, Y, rho, levels=14, cmap=colormaps[task_idx], alpha=0.35, zorder=1)
            ax.streamplot(X, Y, U, V, color=stream_colors[task_idx], linewidth=0.8, density=0.8, arrowsize=0.8, zorder=3)

            # 5. 绘制智能体
            agent_pos = get_synthetic_agents(task_idx, phase_idx)
            ax.scatter(agent_pos[:, 0], agent_pos[:, 1], color=agent_colors[task_idx], 
                       edgecolors='black', s=85, lw=1.2, zorder=10)

            # 6. 视窗设定
            ax.set_xlim(-0.75, 0.75) 
            ax.set_ylim(-0.52, 0.52)
            ax.set_aspect('equal')
            
            for spine in ax.spines.values():
                spine.set_color('#CCCCCC')
                spine.set_linewidth(1.0)
            ax.set_xticks([])
            ax.set_yticks([])

            if task_idx == 0:
                p_labels = [r"$w_{rand}: 1.00 \mid w_{food}: 0.00 \mid w_{nest}: 0.00$",
                            r"$w_{rand}: 0.17 \mid w_{food}: 0.83 \mid w_{nest}: 0.00$",
                            r"$w_{rand}: 0.17 \mid w_{food}: 0.00 \mid w_{nest}: 0.83$"]
                ax.set_title(p_labels[phase_idx], fontsize=13.5, family='monospace', pad=6)
            elif task_idx == 1:
                # 归一化后的自适应编队参数
                w_shapes_norm = [0.92, 0.62, 0.25]
                w_flows_norm = [0.08, 0.38, 0.75]
                p_labels = [
                    rf"$w_{{shape}}: {w_shapes_norm[phase_idx]:.2f} \mid w_{{flow}}: {w_flows_norm[phase_idx]:.2f}$",
                    rf"$w_{{shape}}: {w_shapes_norm[phase_idx]:.2f} \mid w_{{flow}}: {w_flows_norm[phase_idx]:.2f}$",
                    rf"$w_{{shape}}: {w_shapes_norm[phase_idx]:.2f} \mid w_{{flow}}: {w_flows_norm[phase_idx]:.2f}$"
                ]
                ax.set_title(p_labels[phase_idx], fontsize=13.5, family='monospace', pad=6)
            else:
                p_labels = [r"$w_{rand}: 1.00 \mid w_{target}: 0.00 \mid w_{center}: 0.00$",
                            r"$w_{rand}: 0.12 \mid w_{target}: 0.25 \mid w_{center}: 0.63$",
                            r"$w_{rand}: 0.15 \mid w_{target}: 0.85 \mid w_{center}: 0.00$"]
                ax.set_title(p_labels[phase_idx], fontsize=13.5, family='monospace', pad=6)

            if phase_idx == 0 and task_idx != 1:
                ax.legend(
                    loc='upper right', 
                    fontsize=10, 
                    framealpha=0.8, 
                    edgecolor='#EAEAEA', 
                    handletextpad=0.6,   
                    labelspacing=1.5,     
                    borderpad=0.6       
                )
            elif phase_idx == 2 and task_idx == 2:
                ax.legend(
                    loc='upper right', 
                    fontsize=10, 
                    framealpha=0.8, 
                    edgecolor='#EAEAEA', 
                    handletextpad=0.6,   
                    labelspacing=1.5,    
                    borderpad=0.6      
                )

    # 保存为 PDF
    os.makedirs("Paper_Figures", exist_ok=True)
    plt.savefig("Paper_Figures/Three_Tasks_Advection_Fields.pdf", format='pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    draw_3x3_advection_matrix()