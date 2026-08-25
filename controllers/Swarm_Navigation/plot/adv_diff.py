import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'

soft_reds_cmap = LinearSegmentedColormap.from_list(
    "SoftReds", 
    ["#FFFFFF", "#FDEDEC", "#F5B7B1", "#CD6155"]
)


R0 = 0.2
WALL_THRESHOLD = 0.12   

def get_corridor_geometry_array(X_arr):
    """
    高效 NumPy 矢量化通道几何函数（窄道半宽已加宽至 0.18）
    """
    hw = np.zeros_like(X_arr)
    narrow_hw = 0.18  
    
    mask1 = X_arr < -0.7
    hw[mask1] = 0.5
    
    mask2 = (X_arr >= -0.7) & (X_arr < -0.1)
    hw[mask2] = 0.5 + (X_arr[mask2] + 0.7)/0.6 * (narrow_hw - 0.5)
    
    mask3 = (X_arr >= -0.1) & (X_arr < 0.6)
    hw[mask3] = narrow_hw
    
    mask4 = (X_arr >= 0.6) & (X_arr < 0.9)
    hw[mask4] = narrow_hw + (X_arr[mask4] - 0.6)/0.3 * (0.5 - narrow_hw)
    
    mask5 = X_arr >= 0.9
    hw[mask5] = 0.5
    
    return hw

def compute_foraging_field(X, Y, phase):
    """任务一：群体觅食平流场"""
    r_nest = np.array([-0.4, 0.0])
    r_food = np.array([0.4, 0.0])
    
    if phase == 1: 
        base = np.exp(-0.4 * ((X - r_nest[0] - 0.15)**2 + 1.3 * Y**2))
        chaos_noise = 0.40 * np.cos(6.0 * X) * np.sin(6.5 * Y) + \
                      0.18 * np.sin(12.0 * X) * np.cos(10.5 * Y)
        rho = np.clip(base * (0.85 + chaos_noise) - 0.08, 0.0, 1.0)
    elif phase == 2: 
        rho = np.exp(-0.75 * ((X - r_food[0])**2 + (Y - r_food[1])**2))
    else: 
        rho = np.exp(-0.75 * ((X - r_nest[0])**2 + (Y - r_nest[1])**2))
        
    return rho

def compute_formation_field(X, Y, centroid, bt):
    """任务二：自适应编队平流场"""
    dx, dy = X - centroid[0], Y - centroid[1]
    dw = np.sqrt(dx**2 + bt * dy**2) + 1e-6
    rho = np.exp(-1.8 * (dw - R0)**2)
    return rho

def compute_rescue_field(X, Y, phase):
    """任务三：目标救援平流场"""
    r_base = np.array([-0.45, -0.2])
    r_target = np.array([0.45, 0.2])
    
    if phase == 1: 
        base = np.exp(-0.45 * ((X - r_base[0] - 0.15)**2 + 1.2 * (Y - r_base[1] - 0.1)**2))
        chaos_noise = 0.42 * np.cos(6.5 * X) * np.sin(6.0 * Y) + \
                      0.18 * np.sin(13.0 * X) * np.cos(11.0 * Y)
        rho = np.clip(base * (0.85 + chaos_noise) - 0.08, 0.0, 1.0)
    elif phase == 2: 
        dist_to_line = np.abs((r_target[1]-r_base[1])*X - (r_target[0]-r_base[0])*Y) / np.sqrt(0.9**2 + 0.4**2)
        lateral_decay = np.exp(-22.0 * dist_to_line**2)
        longitudinal_decay = np.exp(-1.5 * (X**2 + Y**2))
        rho = lateral_decay * longitudinal_decay
    else: 
        r_target_new = np.array([0.45, -0.2])
        rho = np.exp(-0.75 * ((X - r_target_new[0])**2 + (Y - r_target_new[1])**2))
        
    return rho

def get_density_normalized_diffusion_field(X, Y, agent_pos, task_idx, D, eps_rho=0.05):
    """按照 V_diff=-D/(rho_norm+eps_rho)*grad(rho_norm) 计算扩散速度场。"""
    sigma_agent, sigma_wall = 0.048, 0.045

    # 机器人局部密度场
    rho_agent = np.zeros_like(X, dtype=float)
    for ax_p, ay_p in agent_pos:
        d_sq = (X-ax_p)**2 + (Y-ay_p)**2
        rho_agent += np.exp(-d_sq/(2.0*sigma_agent**2))

    # 墙壁局部密度场
    half_width = get_corridor_geometry_array(X) if task_idx == 1 else np.full_like(X, 0.55)
    dist_top = np.abs(half_width-Y)
    dist_bottom = np.abs(Y+half_width)
    rho_wall = np.exp(-dist_top**2/(2.0*sigma_wall**2)) + np.exp(-dist_bottom**2/(2.0*sigma_wall**2))

    # 合成归一化密度
    rho_raw = rho_agent + rho_wall
    rho_norm = rho_raw/(np.max(rho_raw)+1e-8)

    # 解析网格梯度
    dx = float(X[0, 1]-X[0, 0])
    dy = float(Y[1, 0]-Y[0, 0])
    grad_y, grad_x = np.gradient(rho_norm, dy, dx)

    # 新扩散定义
    diffusion_gain = D/(rho_norm+eps_rho)
    u_diff = -diffusion_gain*grad_x
    v_diff = -diffusion_gain*grad_y

    # 屏蔽通道外部区域
    inside_mask = np.abs(Y) <= half_width
    u_diff = np.where(inside_mask, u_diff, 0.0)
    v_diff = np.where(inside_mask, v_diff, 0.0)

    V_diff_mag = np.sqrt(u_diff**2+v_diff**2)
    return u_diff, v_diff, V_diff_mag, rho_norm


def get_synthetic_agents(task_idx, row_idx):
    if task_idx == 0: 
        if row_idx == 0: 
            return np.array([
                [-0.60,  0.25], [-0.10, -0.22], [0.11,  0.28], [-0.40,  0.09],
                [-0.62, -0.12], [-0.22,  0.22], [0.12, -0.25], [-.32, -0.19]
            ])
        elif row_idx == 1: 
            return np.array([
                [-0.55,  0.36], [-0.39, -0.06], [-0.23,  0.45], [-0.07, -0.28],
                [ 0.29,  0.06], [ 0.21, -0.26], [ 0.41,  0.26], [ 0.62, -0.12]
            ])
        else: 
            return np.array([
                [ 0.55, -0.06], [ 0.39,  0.06], [ 0.23, -0.46], [ 0.07,  0.54],
                [-0.39, -0.26], [-0.52,  0.16], [-0.41, -0.34], [-0.55,  0.38]
            ])
            
    elif task_idx == 1: 
        if row_idx == 0: 
            cx, cy = -0.40, 0.0
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            return np.column_stack([cx + 0.26 * np.cos(angles), cy + 0.26 * np.sin(angles)])
        elif row_idx == 1: 
            cx, cy = -0.32, 0.0 
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            return np.column_stack([cx + 0.28 * np.cos(angles), cy + 0.18 * np.sin(angles)])
        else: 
            return np.column_stack([np.linspace(-0.01, 0.71, 8), np.zeros(8)])
            
    else: 
        if row_idx == 0: 
            return np.array([
                [-0.35, -0.40], [-0.25, -0.11], [-0.10,  0.15], [ 0.05,  0.28],
                [-0.58, -0.02], [0.28, -0.32], [0.08, -0.35], [-0.28, -0.38]
            ])
        elif row_idx == 1: 
            xs = np.linspace(-0.45, 0.45, 8)
            ys = np.linspace(-0.2, 0.2, 8)
            return np.column_stack([xs, ys])
        else: 
            xs = np.linspace(-0.45, 0.45, 8)
            ys = np.linspace(-0.2, -0.2, 8) + 0.18 * np.sin(np.linspace(0, np.pi, 8))
            return np.column_stack([xs, ys])


def draw_3x3_combined_adr_matrix():
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.15, wspace=0.11)
    
    colormaps = ['YlOrBr', 'Reds', 'Blues']
    agent_colors = ['#FFA500', '#FF3333', '#1F77B4']
    
    D_levels = [0.02, 0.05, 0.08] 
    
    X, Y = np.meshgrid(np.linspace(-0.8, 0.8, 400), np.linspace(-0.55, 0.55, 320))

    diffusion_cache = {}
    all_diffusion_magnitudes = []

    for task_idx in range(3):
        for phase_idx, D in enumerate(D_levels):
            agent_pos = get_synthetic_agents(task_idx, phase_idx)
            fields = get_density_normalized_diffusion_field(
                X, Y, agent_pos, task_idx, D, eps_rho=0.05
            )
            diffusion_cache[(task_idx, phase_idx)] = fields
            all_diffusion_magnitudes.append(fields[2].ravel())

    all_diffusion_magnitudes = np.concatenate(all_diffusion_magnitudes)
    V_DIFF_MAX = max(np.percentile(all_diffusion_magnitudes, 99.0), 1e-8)
    
    for task_idx in range(3): 
        for phase_idx in range(3): 
            ax = fig.add_subplot(gs[task_idx, phase_idx])
            D = D_levels[phase_idx]
            
            agent_pos = get_synthetic_agents(task_idx, phase_idx)

            # --- 计算平流渐变密度场 ---
            if task_idx == 0: 
                rho_adv = compute_foraging_field(X, Y, phase_idx + 1)
            elif task_idx == 1: 
                centers = [[-0.5, 0.0], [-0.30, 0.0], [0.35, 0.0]] 
                betas = [1.0, 2.5, 5.0]
                rho_adv = compute_formation_field(X, Y, centers[phase_idx], betas[phase_idx])
            else: 
                rho_adv = compute_rescue_field(X, Y, phase_idx + 1)

            # 计算平滑的高斯排斥势场
            # 读取密度归一化扩散速度场
            u_diff, v_diff, V_diff_mag, rho_norm = diffusion_cache[
                (task_idx, phase_idx)
            ]

            # 统一归一化，仅用于可视化
            V_diff_vis = np.clip(V_diff_mag/V_DIFF_MAX, 0.0, 1.0)

            # 平流背景与扩散作用的联合可视化
            DIFFUSION_VIS_GAIN = 0.35
            rho_total = np.clip(
                rho_adv*np.exp(-DIFFUSION_VIS_GAIN*V_diff_vis),
                0.0,
                1.0
            )

            # 绘制大渐变热力图
            contour = ax.contourf(X, Y, rho_total, levels=24, cmap=colormaps[task_idx], 
                                  alpha=0.45, antialiased=True, zorder=1)
            
            for collection in contour.collections:
                collection.set_edgecolor("face")
                collection.set_linewidth(0.5)

            if task_idx == 1: 
                path = [(-1.6, 0.5), (-0.7, 0.5), (-0.1, 0.18), (0.6, 0.18), (0.9, 0.5), (1.6, 0.5)]
                px, py = zip(*path)
                ax.fill_between(px, py, 1.0, color='#f2f2f2', zorder=2)
                ax.fill_between(px, [-y for y in py], -1.0, color='#f2f2f2', zorder=2)
                ax.plot(px, py, color='black', lw=1.5, zorder=3)
                ax.plot(px, [-y for y in py], color='black', lw=1.5, zorder=3)
            else: 
                ax.axhline(0.55, color='black', lw=1.5, zorder=3)
                ax.axhline(-0.55, color='black', lw=1.5, zorder=2)
                ax.fill_between([-1.0, 1.0], [0.55, 0.55], 1.0, color='#f8f8f8', zorder=2)
                ax.fill_between([-1.0, 1.0], [-0.55, -0.55], -1.0, color='#f8f8f8', zorder=2)
            
            if task_idx == 0: 
                ax.scatter(-0.4, 0.0, marker='s', color='#2CA02C', edgecolors='#1C2833', s=230, lw=0.8, zorder=5, label='Nest' if phase_idx==0 else "")
                ax.scatter(0.4, 0.0, marker='*', color='#D4AF37', edgecolors='#1C2833', s=280, lw=0.8, zorder=5, label='Resource' if phase_idx==0 else "")
            elif task_idx == 2: 
                ax.scatter(-0.45, -0.2, marker='p', color='#1F77B4', edgecolors='#1C2833', s=230, lw=0.8, zorder=5, label='Base Station' if phase_idx==0 else "")
                if phase_idx == 2: 
                    ax.scatter(0.45, -0.2, marker='H', color='#FF3333', edgecolors='#1C2833', s=230, lw=0.8, zorder=5, label='New Target' if phase_idx==2 else "")
                    ax.scatter(0.45, 0.2, marker='H', color='gray', s=130, alpha=0.3, zorder=4)
                else:
                    ax.scatter(0.45, 0.2, marker='H', color='#FF3333', edgecolors='#1C2833', s=230, lw=0.8, zorder=5, label='Target' if phase_idx==0 else "")

            # 智能体下方排斥光晕
            ax.scatter(agent_pos[:, 0], agent_pos[:, 1], color='white', alpha=0.6, s=110, zorder=14)

            # 智能体主体
            ax.scatter(agent_pos[:, 0], agent_pos[:, 1], color=agent_colors[task_idx], 
                       edgecolors='#1C2833', s=75, lw=0.8, zorder=15)

            # 布局参数
            ax.set_xlim(-0.75, 0.75) 
            ax.set_ylim(-0.52, 0.52)
            ax.set_aspect('equal')
            
            for spine in ax.spines.values():
                spine.set_color('#CCCCCC')
                spine.set_linewidth(1.0)
            ax.set_xticks([])
            ax.set_yticks([])

            if task_idx == 0:
                p_labels = [
                    rf"$D: {D:.2f} \mid w_{{rand}}: 1.00 \mid w_{{food}}: 0.00 \mid w_{{nest}}: 0.00$",
                    rf"$D: {D:.2f} \mid w_{{rand}}: 0.17 \mid w_{{food}}: 0.83 \mid w_{{nest}}: 0.00$",
                    rf"$D: {D:.2f} \mid w_{{rand}}: 0.17 \mid w_{{food}}: 0.00 \mid w_{{nest}}: 0.83$"
                ]
                ax.set_title(p_labels[phase_idx], fontsize=13.5, family='monospace', pad=6)
            elif task_idx == 1:
                w_shapes_norm = [0.92, 0.62, 0.25]
                w_flows_norm = [0.08, 0.38, 0.75]
                p_labels = [
                    rf"$D: {D:.2f} \mid w_{{shape}}: {w_shapes_norm[phase_idx]:.2f} \mid w_{{flow}}: {w_flows_norm[phase_idx]:.2f}$",
                    rf"$D: {D:.2f} \mid w_{{shape}}: {w_shapes_norm[phase_idx]:.2f} \mid w_{{flow}}: {w_flows_norm[phase_idx]:.2f}$",
                    rf"$D: {D:.2f} \mid w_{{shape}}: {w_shapes_norm[phase_idx]:.2f} \mid w_{{flow}}: {w_flows_norm[phase_idx]:.2f}$"
                ]
                ax.set_title(p_labels[phase_idx], fontsize=13.5, family='monospace', pad=6)
            else:
                p_labels = [
                    rf"$D: {D:.2f} \mid w_{{rand}}: 1.00 \mid w_{{target}}: 0.00 \mid w_{{center}}: 0.00$",
                    rf"$D: {D:.2f} \mid w_{{rand}}: 0.12 \mid w_{{target}}: 0.25 \mid w_{{center}}: 0.63$",
                    rf"$D: {D:.2f} \mid w_{{rand}}: 0.15 \mid w_{{target}}: 0.85 \mid w_{{center}}: 0.00$"
                ]
                ax.set_title(p_labels[phase_idx], fontsize=13.5, family='monospace', pad=6)

            # 标志物图例
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

            from matplotlib.colors import Normalize
            from matplotlib.cm import ScalarMappable

            if phase_idx == 2:
                pos = ax.get_position()
                cax = fig.add_axes([pos.x1 + 0.016, pos.y0, 0.012, pos.height])
                
                sm = ScalarMappable(
                    norm=Normalize(vmin=0.0, vmax=1.0), 
                    cmap=plt.get_cmap(colormaps[task_idx])
                )
                
                cbar = fig.colorbar(sm, cax=cax, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], alpha=0.45)
                cbar.ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10, family='sans-serif')
                cbar.ax.tick_params(labelsize=10, width=0.8, length=3)

    # 保存最终合并版本 PDF
    os.makedirs("Paper_Figures", exist_ok=True)
    plt.savefig("Paper_Figures/Three_Tasks_Unified_ADR_Heatmaps.pdf", format='pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    draw_3x3_combined_adr_matrix()