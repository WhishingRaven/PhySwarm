import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'

R0 = 0.2
WALL_THRESHOLD = 0.12   
AGENT_THRESHOLD = 0.15  

def get_corridor_geometry(px):
    if px < -0.7: 
        return 0.5, 0.0
    elif -0.7 <= px < -0.1: 
        return 0.5 + (px + 0.7)/0.6 * (0.12 - 0.5), -0.6333
    elif -0.1 <= px < 0.6: 
        return 0.12, 0.0
    elif 0.6 <= px < 0.9:
        return 0.12 + (px - 0.6)/0.3 * (0.5 - 0.12), 1.2666
    else: 
        return 0.5, 0.0

def get_corridor_half_width_array(X):
    hw = np.full_like(X, 0.5, dtype=float)
    mask = (X >= -0.7) & (X < -0.1)
    hw[mask] = 0.5 + (X[mask] + 0.7) / 0.6 * (0.12 - 0.5)
    mask = (X >= -0.1) & (X < 0.6)
    hw[mask] = 0.12
    mask = (X >= 0.6) & (X < 0.9)
    hw[mask] = 0.12 + (X[mask] - 0.6) / 0.3 * (0.5 - 0.12)
    return hw

def get_physics_diffusion_fields(X, Y, agent_pos, task_idx, D, eps_rho=0.05):
    """按照 V_diff=-D/(rho_norm+eps_rho)*grad(rho_norm) 计算扩散速度。"""
    sigma_agent, sigma_wall = 0.050, 0.040

    # 机器人密度场：每个机器人对应一个平滑高斯核
    rho_agent = np.zeros_like(X, dtype=float)
    for agent_x, agent_y in agent_pos:
        dist_sq = (X-agent_x)**2 + (Y-agent_y)**2
        rho_agent += np.exp(-dist_sq/(2.0*sigma_agent**2))

    # 墙壁密度场：越接近墙壁，密度值越高
    if task_idx == 1:
        half_width = get_corridor_half_width_array(X)
    else:
        half_width = np.full_like(X, 0.55, dtype=float)

    dist_top = np.abs(half_width-Y)
    dist_bottom = np.abs(Y+half_width)
    rho_wall = np.exp(-dist_top**2/(2.0*sigma_wall**2)) + np.exp(-dist_bottom**2/(2.0*sigma_wall**2))

    rho_raw = rho_agent + rho_wall
    rho_norm = rho_raw/(np.max(rho_raw)+1e-8)

    # 精确计算二维密度梯度
    dx = float(X[0, 1]-X[0, 0])
    dy = float(Y[1, 0]-Y[0, 0])
    grad_y, grad_x = np.gradient(rho_norm, dy, dx)

    # 新扩散定义
    diffusion_gain = D/(rho_norm+eps_rho)
    u_diff = -diffusion_gain*grad_x
    v_diff = -diffusion_gain*grad_y
    speed_diff = np.sqrt(u_diff**2+v_diff**2)

    # 清除环境外部区域的扩散箭头
    inside_mask = np.abs(Y) <= half_width
    u_diff = np.where(inside_mask, u_diff, 0.0)
    v_diff = np.where(inside_mask, v_diff, 0.0)
    speed_diff = np.where(inside_mask, speed_diff, 0.0)

    return u_diff, v_diff, speed_diff, rho_norm


def get_adaptive_agents(task_idx, diff_idx):
    if task_idx == 0: 
        if diff_idx == 0: 
            return np.array([[-0.45, 0.02], [-0.43, -0.01], [-0.44, 0.03], [-0.46, -0.02], 
                             [-0.42, 0.01], [-0.47, 0.0], [-0.45, -0.03], [-0.43, 0.015]])
        elif diff_idx == 1: 
            return np.array([[-0.38, 0.0], [-0.26, 0.02], [-0.14, -0.01], [-0.02, 0.01], 
                             [0.10, -0.02], [0.22, 0.0], [0.34, 0.03], [0.44, -0.01]])
        else: 
            return np.array([[-0.55, 0.32], [-0.35, -0.32], [-0.48, 0.38], [-0.22, 0.08], 
                             [-0.58, -0.12], [-0.26, 0.28], [-0.51, -0.3], [-0.18, -0.18]])
            
    elif task_idx == 1: 
        if diff_idx == 0: 
            cx, cy = -0.30, 0.0
            return np.array([[cx-0.03, cy+0.02], [cx+0.02, cy-0.03], [cx, cy+0.04], [cx-0.02, cy-0.01], 
                             [cx+0.03, cy+0.01], [cx-0.01, cy-0.04], [cx+0.04, cy-0.01], [cx-0.04, cy+0.03]])
        elif diff_idx == 1: 
            cx, cy = -0.30, 0.0
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            return np.column_stack([cx + 0.22*np.cos(angles), cy + (0.22/np.sqrt(2.5))*np.sin(angles)])
        else: 
            cx, cy = -0.38, 0.0
            a, b = 0.25, 0.15
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            return np.column_stack([cx + a*np.cos(angles), cy + b*np.sin(angles)])
            
    else: 
        if diff_idx == 0: 
            return np.array([[-0.1, -0.05], [-0.08, -0.03], [-0.12, -0.04], [-0.09, -0.06], 
                             [0.1, 0.05], [0.08, 0.03], [0.12, 0.04], [0.09, 0.06]])
        elif diff_idx == 1: 
            xs = np.linspace(-0.45, 0.45, 8)
            ys = np.linspace(-0.2, 0.2, 8)
            return np.column_stack([xs, ys])
        else: 
            return np.array([[-0.45, 0.1], [-0.3, -0.35], [-0.15, 0.32], [0.0, -0.28], 
                             [0.15, 0.35], [0.3, -0.3], [0.45, 0.1], [0.0, 0.1]])

# ==========================================
#  绘图主程序
# ==========================================
def draw_3x3_diffusion_matrix():
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.15, wspace=0.11)
    
    colormap = 'Blues'
    agent_color = '#2C3E50' 
    
    X, Y = np.meshgrid(np.linspace(-0.8, 0.8, 100), np.linspace(-0.55, 0.55, 80))
    D_levels = [0.0, 0.035, 0.085]

    field_cache = {}
    all_speed_values = []

    for task_idx in range(3):
        for diff_idx, D in enumerate(D_levels):
            agent_pos = get_adaptive_agents(task_idx, diff_idx)
            fields = get_physics_diffusion_fields(X, Y, agent_pos, task_idx, D, eps_rho=0.05)
            field_cache[(task_idx, diff_idx)] = fields
            all_speed_values.append(fields[2].ravel())

    all_speed_values = np.concatenate(all_speed_values)
    V_DIFF_MAX = np.percentile(all_speed_values, 99.0)
    V_DIFF_MAX = max(V_DIFF_MAX, 1e-8)
    
    for task_idx in range(3): 
        for diff_idx in range(3): 
            ax = fig.add_subplot(gs[task_idx, diff_idx])
            D = D_levels[diff_idx]
            
            if task_idx == 1: 
                path = [(-1.6, 0.5), (-0.7, 0.5), (-0.1, 0.12), (1.6, 0.12)]
                px, py = zip(*path)
                ax.fill_between(px, py, 1.0, color='#f0f0f0', zorder=1)
                ax.fill_between(px, [-y for y in py], -1.0, color='#f0f0f0', zorder=1)
                ax.plot(px, py, color='black', lw=2.0, zorder=2)
                ax.plot(px, [-y for y in py], color='black', lw=2.0, zorder=2)
            else: 
                ax.axhline(0.55, color='black', lw=2.0, zorder=2)
                ax.axhline(-0.55, color='black', lw=2.0, zorder=2)
                ax.fill_between([-1.0, 1.0], [0.55, 0.55], 1.0, color='#f0f0f0', zorder=1)
                ax.fill_between([-1.0, 1.0], [-0.55, -0.55], -1.0, color='#f0f0f0', zorder=1)
            
            if task_idx == 0: 
                ax.scatter(-0.4, 0.0, marker='s', color='#2CA02C', edgecolors='white', s=250, zorder=5, label='Nest' if diff_idx==0 else "")
                ax.scatter(0.4, 0.0, marker='*', color='#F1C40F', edgecolors='white', s=300, zorder=5, label='Resource' if diff_idx==0 else "")
            elif task_idx == 2: 
                ax.scatter(-0.45, -0.2, marker='p', color='#2980B9', edgecolors='white', s=250, zorder=5, label='Base Station' if diff_idx==0 else "")
                ax.scatter(0.45, 0.2, marker='H', color='#E74C3C', edgecolors='white', s=250, zorder=5, label='Target' if diff_idx==0 else "")

            agent_pos = get_adaptive_agents(task_idx, diff_idx)
            u_dif, v_dif, V_diff_mag, rho_norm = field_cache[(task_idx, diff_idx)]

            ax.contourf(
                X, Y, V_diff_mag,
                levels=16,
                cmap=colormap,
                vmin=0.0,
                vmax=V_DIFF_MAX,
                alpha=0.30,
                zorder=1
            )
            
            if D > 0.001:
                skip = (slice(None, None, 4), slice(None, None, 4))
                ax.quiver(
                    X[skip], Y[skip],
                    u_dif[skip], v_dif[skip],
                    color='blue',
                    angles='xy',
                    scale_units='xy',
                    scale=V_DIFF_MAX*15.0,
                    width=0.004,
                    alpha=0.75,
                    zorder=10
                )

            ax.scatter(agent_pos[:, 0], agent_pos[:, 1], color=agent_color, 
                       edgecolors='white', s=75, lw=1.2, zorder=15)

            ax.set_xlim(-0.75, 0.75) 
            ax.set_ylim(-0.52, 0.52)
            ax.set_aspect('equal')
            
            # 边框精致化
            for spine in ax.spines.values():
                spine.set_color('#BDC3C7')
                spine.set_linewidth(1.0)
            ax.set_xticks([])
            ax.set_yticks([])

            if diff_idx == 0 and task_idx != 1:
                ax.legend(
                    loc='upper right', 
                    fontsize=10, 
                    framealpha=0.8, 
                    edgecolor='#EAEAEA', 
                    handletextpad=0.6,   
                    labelspacing=1.5,   
                    borderpad=0.6       
                )

            ax.set_title(f"$D$: {D:.2f}", fontsize=11, family='monospace', pad=6)

    # 保存为科研 PDF 矢量格式
    os.makedirs("Paper_Figures", exist_ok=True)
    plt.savefig("Paper_Figures/Three_Tasks_Diffusion_Fields.pdf", format='pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    draw_3x3_diffusion_matrix()