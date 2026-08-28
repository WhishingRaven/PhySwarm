import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)

fig, axes = plt.subplots(3, 3, figsize=(16, 16.5), dpi=300)
axes = axes.flatten()

def format_ax(ax, title):
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')


ax = axes[0]
pc = np.array([1, 1])
Phi_point = 0.5 * ((X - pc[0])**2 + (Y - pc[1])**2)
U_point = -(X - pc[0])
V_point = -(Y - pc[1])

ax.contourf(X, Y, Phi_point, levels=30, cmap='Blues_r', alpha=0.8)
ax.streamplot(X, Y, U_point, V_point, color='black', linewidth=1.2, density=0.8, arrowsize=1.5)
ax.plot(pc[0], pc[1], 'r*', markersize=18, markeredgecolor='black', label=r'Target $\mathbf{p}_c$')
ax.legend(loc='upper right', framealpha=0.9, fontsize=11)
format_ax(ax, r"1. Point attraction/repulsion field ($\Phi_{point}$)")

ax = axes[1]
beta = 3.0  
R0 = 3.0
dist_warped = np.sqrt(X**2 + beta * Y**2)
Phi_shape = 0.5 * (dist_warped - R0)**2

dist_safe = np.where(dist_warped == 0, 1e-5, dist_warped)
error = dist_safe - R0
U_shape = -error * (X / dist_safe)
V_shape = -error * (beta * Y / dist_safe)

ax.contourf(X, Y, Phi_shape, levels=30, cmap='Purples_r', alpha=0.8)
ax.streamplot(X, Y, U_shape, V_shape, color='black', linewidth=1.2, density=0.9, arrowsize=1.5)
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(R0*np.cos(theta), (R0/np.sqrt(beta))*np.sin(theta), 'r--', lw=3, label=r'Manifold $\mathcal{M}_{\beta}$')
ax.legend(loc='upper right', framealpha=0.9, fontsize=11)
format_ax(ax, r"2. Manifold or morphology field ($\Phi_{shape}$)")

ax = axes[2]
v_dir = np.array([1.0, 0.4])
Phi_flow = -(v_dir[0] * X + v_dir[1] * Y)
U_flow = np.ones_like(X) * v_dir[0]
V_flow = np.ones_like(Y) * v_dir[1]

ax.contourf(X, Y, Phi_flow, levels=30, cmap='Greens_r', alpha=0.8)
ax.streamplot(X, Y, U_flow, V_flow, color='black', linewidth=1.5, density=0.7, arrowsize=2.0)
format_ax(ax, r"3. Uniform vector-flow field ($\Phi_{flow}$)")

ax = axes[3]
p_shared = np.array([-1.5, 0.5])
Phi_info_a = 0.5 * ((X - p_shared[0])**2 + (Y - p_shared[1])**2)
U_info_a = -(X - p_shared[0])
V_info_a = -(Y - p_shared[1])

ax.contourf(X, Y, Phi_info_a, levels=30, cmap='GnBu_r', alpha=0.8)
ax.streamplot(X, Y, U_info_a, V_info_a, color='gray', linewidth=1.2, density=0.8, arrowsize=1.5)
ax.plot(p_shared[0], p_shared[1], '*', markersize=16, markeredgecolor='black', label=r'$\mathbf{P}_{shared}$')
for r in [1, 2, 3]:
    ax.add_patch(Circle((p_shared[0], p_shared[1]), r, color='magenta', fill=False, linestyle=':', lw=2, alpha=0.6))
ax.legend(loc='upper right', framealpha=0.9, fontsize=11)
format_ax(ax, r"4a. Coordinate-anchored field ($\Phi_{info\_a}$)")

ax = axes[4]
sources = [[-2, -2], [1, 2],[3, -1], [-1, 3], [0, 0]]
Phi_info_b = np.zeros_like(X)
for src in sources:
    Phi_info_b -= 1.5 * np.exp(-0.6 * ((X - src[0])**2 + (Y - src[1])**2))

V_info_b, U_info_b = np.gradient(-Phi_info_b)
U_info_b, V_info_b = -U_info_b, -V_info_b

ax.contourf(X, Y, Phi_info_b, levels=30, cmap='Oranges_r', alpha=0.8)
ax.streamplot(X, Y, U_info_b, V_info_b, color='black', linewidth=1.2, density=1.0, arrowsize=1.5)
for src in sources:
    ax.plot(src[0], src[1], 'yo', markersize=8, markeredgecolor='black')
ax.plot([],[], 'yo', markeredgecolor='black', label='Pheromone Sources')
ax.legend(loc='upper right', framealpha=0.9, fontsize=11)
format_ax(ax, r"4b. Distribution-accumulated field ($\Phi_{info\_b}$)")

ax = axes[5]
agents_x = np.array([-1.5, -0.5, 0.8, 2.0, 1.2, -0.8])
agents_y = np.array([1.0, 2.5, 1.8, -0.5, -2.0, -1.5])
lc_x, lc_y = np.mean(agents_x), np.mean(agents_y)

Phi_soc = 0.5 * ((X - lc_x)**2 + (Y - lc_y)**2)
U_soc = -(X - lc_x)
V_soc = -(Y - lc_y)

ax.contourf(X, Y, Phi_soc, levels=30, cmap='Reds_r', alpha=0.8)

cohesion_boundary = Circle((lc_x, lc_y), 2.5, facecolor='none', edgecolor='green', linestyle='--', lw=2.0, alpha=0.8, label='Cohesion Range')
ax.add_patch(cohesion_boundary)

for ax_x, ax_y in zip(agents_x, agents_y):
    ax.plot([ax_x, lc_x], [ax_y, lc_y], 'k:', alpha=0.6, lw=1.5)

ax.quiver(agents_x, agents_y, -(agents_x - lc_x), -(agents_y - lc_y), 
          color='black', scale=8, width=0.008, headwidth=4, zorder=5)
ax.plot(agents_x, agents_y, 'co', markersize=10, markeredgecolor='black', zorder=6, label='Local Agents')
ax.plot(lc_x, lc_y, 'gD', markersize=12, markeredgecolor='black', zorder=6, label=r'Centroid $\mathbf{x}_{lc}$')
ax.legend(loc='upper right', framealpha=0.9, fontsize=11)
format_ax(ax, r"5. Local-centroid cohesion field ($\Phi_{soc}$)")

ax = axes[6]
pA = np.array([-2.5, -2.5]) 
pB = np.array([2.5, 2.5])   
kv, ku = 4.0, 0.4          

Phi_coup = 0.5 * kv * ((-X + Y)/np.sqrt(2))**2 + 0.5 * ku * ((X + Y)/np.sqrt(2))**2
U_coup = 0.5 * ((kv - ku) * Y - (kv + ku) * X)
V_coup = 0.5 * ((kv - ku) * X - (kv + ku) * Y)

ax.contourf(X, Y, Phi_coup, levels=30, cmap='YlOrRd_r', alpha=0.8)
ax.streamplot(X, Y, U_coup, V_coup, color='black', linewidth=1.2, density=0.8, arrowsize=1.5)

ax.plot(pA[0], pA[1], 'gP', markersize=12, markeredgecolor='black', label=r'Base $\mathbf{p}_A$')
ax.plot(pB[0], pB[1], 'r*', markersize=16, markeredgecolor='black', label=r'Target $\mathbf{p}_B$')
ax.plot(0, 0, 'cD', markersize=10, markeredgecolor='black', label=r'Relay $\mathbf{p}_{mid}$')
ax.plot([pA[0], pB[0]], [pA[1], pB[1]], 'k--', alpha=0.4, lw=1.5)
ax.legend(loc='upper right', framealpha=0.9, fontsize=11)
format_ax(ax, r"6. Bipolar relay or coupling field ($\Phi_{coup}$)")

ax = axes[7]
r = np.sqrt(X**2 + Y**2) + 1e-5
Phi_bar = 8.0 / r
U_bar = X / r**2
V_bar = Y / r**2

ax.contourf(X, Y, Phi_bar, levels=30, cmap='magma_r', alpha=0.8)
ax.streamplot(X, Y, U_bar, V_bar, color='black', linewidth=1.2, density=0.8, arrowsize=1.5)

hollow_mask = Circle((0, 0), 2.0, facecolor='white', fill=True, zorder=4)
ax.add_patch(hollow_mask)

threat_zone = Circle((0, 0), 2.0, facecolor='crimson', fill=True, hatch='//', alpha=0.15, edgecolor='crimson', lw=2, zorder=5, label=r'Exclusion Zone $\partial \Omega$')
ax.add_patch(threat_zone)
ax.legend(loc='upper right', framealpha=0.9, fontsize=11)
format_ax(ax, r"7. Regional safety barrier field ($\Phi_{bar}$)")

ax = axes[8]
sigma_x, sigma_y = 2.5, 1.0 
Phi_ani = 0.5 * (X**2 / sigma_x**2 + Y**2 / sigma_y**2)
U_ani = -X / sigma_x**2
V_ani = -Y / sigma_y**2

ax.contourf(X, Y, Phi_ani, levels=30, cmap='YlGnBu_r', alpha=0.8)

ax.streamplot(X, Y, U_ani, V_ani, color='black', linewidth=0.8, density=0.8, arrowsize=1.0)

R_ref = 3.0
angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
x_p = R_ref * np.cos(angles)
y_p = R_ref * np.sin(angles)
u_p = -x_p / sigma_x**2
v_p = -y_p / sigma_y**2

ref_circle = Circle((0, 0), R_ref, facecolor='none', edgecolor='gray', linestyle='--', lw=1.5, alpha=0.8, label='Equal-Distance Ring')
# ax.add_patch(ref_circle)

# 绘制 8 个智能体测试点 (金色)
ax.scatter(x_p, y_p, color='gold', edgecolor='black', s=90, zorder=11)

ax.quiver(x_p, y_p, u_p, v_p, color='crimson', angles='xy', scale_units='xy', scale=1.0, 
          width=0.015, headwidth=4, headlength=5, zorder=12, label='Motion Preference')

ellipse_t = np.linspace(0, 2*np.pi, 100)
ax.plot(0, 0, 'go', markersize=10, markeredgecolor='black', zorder=10, label=r'Center $\mathbf{p}_c$')
ax.legend(loc='upper right', framealpha=0.9, fontsize=11)
format_ax(ax, r"8. Anisotropic or direction-weighted field ($\Phi_{ani}$)")

plt.tight_layout()
plt.savefig('Paper_Figures/Potential_Field_Dictionary_Full.pdf', bbox_inches='tight', dpi=300)
# plt.show()
