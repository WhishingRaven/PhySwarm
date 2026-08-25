import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os

plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'

x = np.linspace(-4, 4, 150)
y = np.linspace(-4, 4, 150)
X, Y = np.meshgrid(x, y)

fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=300)
axes = axes.flatten()

def format_ax(ax, title):
    ax.set_title(title, fontsize=15, fontweight='bold', pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')

DX = x[1] - x[0]
DY = y[1] - y[0]
EPS_RHO = 0.05

def calc_density_normalized_diffusion(rho, D=0.05, eps_rho=EPS_RHO):
    """V_diff = -D/(rho_norm+eps_rho) * grad(rho_norm)."""
    rho_norm = rho / (np.max(rho) + 1e-8)
    grad_y, grad_x = np.gradient(rho_norm, DY, DX)
    gain = D / (rho_norm + eps_rho)
    U = -gain * grad_x
    V = -gain * grad_y
    speed = np.sqrt(U**2 + V**2)
    return U, V, speed, rho_norm

agents = np.array([[-1.5, -0.5], [0.5, 1.0], [-0.2, 1.5], [1.2, -1.2], [0.0, -1.8]])

ax = axes[0]
# 智能体局部密度
rho_agent = np.zeros_like(X)
for ag in agents:
    rho_agent += np.exp(-((X - ag[0])**2 + (Y - ag[1])**2) / (2 * 0.8**2))

# 障碍物虚拟距离场
dist_wall = np.maximum(0, (3 - (X + Y)) / np.sqrt(2)) 
dist_circle = np.maximum(0, np.sqrt((X + 2)**2 + (Y - 2)**2) - 1.0)
min_dist = np.minimum(dist_wall, dist_circle)
rho_obs = np.exp(-min_dist / 0.8)

# 广义总密度
rho_total = 1.0 * rho_agent + 1.2 * rho_obs

D_TOTAL = 0.05
U_total, V_total, speed_total, rho_total_norm = calc_density_normalized_diffusion(
    rho_total, D=D_TOTAL
)

# 屏蔽障碍物内部
inside_obs = (X + Y >= 3) | ((X + 2)**2 + (Y - 2)**2 <= 1.0)
U_total[inside_obs] = np.nan
V_total[inside_obs] = np.nan

ax.contourf(X, Y, rho_total_norm, levels=40, cmap='Reds', alpha=0.85)
ax.streamplot(X, Y, U_total, V_total, color='black', linewidth=1.1, density=0.8, arrowsize=1.3)

# 绘制障碍物与智能体
ax.add_patch(Polygon([[3, 0], [4, -1], [4, 4], [-1, 4], [0, 3]], closed=True, color='dimgray', zorder=5))
ax.add_patch(plt.Circle((-2, 2), 1.0, color='dimgray', zorder=5))
ax.plot(agents[:, 0], agents[:, 1], 'co', markersize=8, markeredgecolor='black', zorder=6)

ax.text(-2, 2, 'Obs', color='white', fontweight='bold', fontsize=11, ha='center', va='center', zorder=6)
ax.text(2.6, 2.6, 'Obs', color='white', fontweight='bold', fontsize=11, ha='center', va='center', zorder=6)

format_ax(ax, r"(a) Generalized total-density field ($\hat{\rho}_{total}$)")

ax = axes[1]
rho_risk = 2.0 * np.exp(-(X**2 + Y**2)/2.0) + 1.5 * np.exp(-((X-2.3)**2 + (Y+2.3)**2)/1.2)

dy_risk, dx_risk = np.gradient(rho_risk)
U_risk = -dx_risk
V_risk = -dy_risk

ax.contourf(X, Y, rho_risk, levels=40, cmap='Oranges', alpha=0.85)
ax.streamplot(X, Y, U_risk, V_risk, color='black', linewidth=1.1, density=0.8, arrowsize=1.3)

ax.add_patch(plt.Circle((0, 0), 0.5, color='crimson', alpha=0.8, zorder=5))
ax.add_patch(plt.Circle((2.3, -2.3), 0.4, color='crimson', alpha=0.8, zorder=5))

ax.text(0, 0, 'Hazardous\nSource A', color='white', fontweight='bold', fontsize=8, ha='center', va='center', zorder=6)
ax.text(2.3, -2.3, 'Hazardous\nSource B', color='white', fontweight='bold', fontsize=8, ha='center', va='center', zorder=6)

format_ax(ax, r"(b) Hazard or risk field ($\hat{\rho}_{risk}$)")

ax = axes[2]
rho_comm = 2.5 * np.exp(-((X+1.8)**2 + (Y+1.8)**2)/1.0) + 2.5 * np.exp(-((X-1.8)**2 + (Y-1.8)**2)/1.0)

D_COMM = 0.05
U_comm, V_comm, speed_comm, rho_comm_norm = calc_density_normalized_diffusion(
    rho_comm, D=D_COMM
)
ax.contourf(X, Y, rho_comm_norm, levels=40, cmap='Purples', alpha=0.85)
ax.streamplot(X, Y, U_comm, V_comm, color='black', linewidth=1.1, density=0.8, arrowsize=1.3)

ax.scatter([-1.8, 1.8], [-1.8, 1.8], marker='^', s=150, color='indigo', edgecolors='black', zorder=6)
ax.text(-1.8, -2.4, 'AP Node 1', color='indigo', fontweight='bold', fontsize=10, ha='center', zorder=6)
ax.text(1.8, 2.2, 'AP Node 2', color='indigo', fontweight='bold', fontsize=10, ha='center', zorder=6)
format_ax(ax, r"(c) Communication-congestion field ($\hat{\rho}_{comm}$)")

ax = axes[3]
rho_explore = 3.0 / (1.0 + np.exp(1.5 * X)) 

D_EXPLORE = 0.05
U_exp, V_exp, speed_exp, rho_explore_norm = calc_density_normalized_diffusion(
    rho_explore, D=D_EXPLORE
)

ax.contourf(X, Y, rho_explore_norm, levels=40, cmap='Greens', alpha=0.8)
ax.streamplot(X, Y, U_exp, V_exp, color='black', linewidth=1.1, density=0.8, arrowsize=1.3)

bbox_style = dict(facecolor='white', alpha=0.85, edgecolor='darkgreen', boxstyle='round,pad=0.3', lw=1.2)
ax.text(-2.2, 3.0, 'Explored Zone\n(Low Demand)', color='darkgreen', fontweight='bold', fontsize=10, ha='center', bbox=bbox_style, zorder=6)
ax.text(2.2, -3.0, 'Unexplored Frontier\n(High Demand)', color='darkgreen', fontweight='bold', fontsize=10, ha='center', bbox=bbox_style, zorder=6)

format_ax(ax, r"(d) Exploration-density field ($\hat{\rho}_{explore}$)")

ax = axes[4]

theta = np.pi / 6
cos_t, sin_t = np.cos(theta), np.sin(theta)

U_rot = cos_t * X + sin_t * Y
V_rot = -sin_t * X + cos_t * Y

rho_aniso = np.exp(-(U_rot**2 / (2 * 1.8**2) + V_rot**2 / (2 * 0.7**2)))

rho_aniso_norm = rho_aniso / (np.max(rho_aniso) + 1e-8)
grad_y_ani, grad_x_ani = np.gradient(rho_aniso_norm, DY, DX)

g_u = cos_t * grad_x_ani + sin_t * grad_y_ani
g_v = -sin_t * grad_x_ani + cos_t * grad_y_ani

D_u, D_v = 2.0, 0.1
density_gain = 1.0 / (rho_aniso_norm + EPS_RHO)

v_u = -D_u * density_gain * g_u
v_v = -D_v * density_gain * g_v

U_aniso = cos_t * v_u - sin_t * v_v
V_aniso = sin_t * v_u + cos_t * v_v

ax.contourf(X, Y, rho_aniso_norm, levels=40, cmap='Blues', alpha=0.85)
ax.streamplot(X, Y, U_aniso, V_aniso, color='black', linewidth=1.1, density=0.8, arrowsize=1.3)

ax.arrow(0, 0, 2.5 * cos_t, 2.5 * sin_t, head_width=0.25, head_length=0.3, fc='darkblue', ec='darkblue', lw=2.5, zorder=6)
ax.arrow(0, 0, -1.0 * sin_t, 1.0 * cos_t, head_width=0.15, head_length=0.2, fc='crimson', ec='crimson', lw=2.0, zorder=6)

bbox_blue = dict(facecolor='white', alpha=0.85, edgecolor='darkblue', boxstyle='round,pad=0.2', lw=1.2)
bbox_red = dict(facecolor='white', alpha=0.85, edgecolor='crimson', boxstyle='round,pad=0.2', lw=1.2)
ax.text(1.3 * cos_t + 0.3, 1.3 * sin_t - 0.5, r'Major Axis ($D_1=2.0$)', color='darkblue', fontweight='bold', fontsize=9, bbox=bbox_blue, zorder=7)
ax.text(-1.0 * sin_t - 0.5, 1.0 * cos_t + 0.3, r'Minor Axis ($D_2=0.1$)', color='crimson', fontweight='bold', fontsize=9, bbox=bbox_red, zorder=7)

format_ax(ax, r"(e) Anisotropic spatial-diffusion field ($f_{diff}$)")

ax = axes[5]
rho_state = 0.5 + 0.5 * np.tanh(0.7 * X + 0.3 * Y)

dy_st, dx_st = np.gradient(rho_state)
U_state = -dx_st
V_state = -dy_st

ax.contourf(X, Y, rho_state, levels=40, cmap='coolwarm', alpha=0.7)
ax.streamplot(X, Y, U_state, V_state, color='black', linewidth=1.0, density=0.8, arrowsize=1.2)

net_nodes = np.array([
    [-2.5, -2.0, 0.15], [-2.0, 1.5, 0.30], [-0.5, -1.5, 0.42], [0.0, 2.0, 0.55],
    [1.5, -2.5, 0.68], [1.0, 0.5, 0.72], [2.5, -0.5, 0.85], [3.0, 2.0, 0.95]
])

edges = [(0,1), (0,2), (1,3), (2,4), (2,5), (3,5), (4,6), (5,6), (5,7), (6,7)]
for edge in edges:
    n1, n2 = net_nodes[edge[0]], net_nodes[edge[1]]
    ax.plot([n1[0], n2[0]], [n1[1], n2[1]], 'k--', lw=1.2, alpha=0.5, zorder=5)

sc = ax.scatter(net_nodes[:, 0], net_nodes[:, 1], c=net_nodes[:, 2], cmap='coolwarm', 
                s=120, edgecolors='black', lw=1.5, zorder=10, vmin=0.0, vmax=1.0)

for node in net_nodes:
    ax.text(node[0], node[1]+0.3, f"$\\theta$={node[2]:.2f}", fontsize=8, fontweight='bold', 
            ha='center', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'), zorder=11)

format_ax(ax, r"(f) Cooperative state-consensus field ($D_{\theta}$)")


plt.tight_layout()
os.makedirs('Paper_Figures', exist_ok=True)
plt.savefig('Paper_Figures/Diffusion_Density_Field_Corrected.pdf', bbox_inches='tight', dpi=300)
# plt.show()

print("✅ Great success! High-quality 2x3 grid of all expanded diffusion fields saved.")