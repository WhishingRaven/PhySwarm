import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import glob
import os
from matplotlib.ticker import FormatStrFormatter

WIDTH_CM = 12.0     # 目标宽度 (厘米)
HEIGHT_CM = 6.0     # 高度
FIG_WIDTH = WIDTH_CM / 2.54
FIG_HEIGHT = HEIGHT_CM / 2.54

plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42

# 全局调粗坐标轴线框及刻度线宽度
plt.rcParams['axes.linewidth'] = 1.2       
plt.rcParams['xtick.major.width'] = 1.2     
plt.rcParams['ytick.major.width'] = 1.2   

def smooth_data(data, window=5):
    return np.convolve(data, np.ones(window)/window, mode='same')

def plot_scientific_metric(steps, data_matrix, ylabel, save_filename, 
                           title=None, legend_label='Mean Value', 
                           highlight_zone=None, highlight_text='Narrow Passage', 
                           color_main='#1a2a6c', color_fill='#2c408b',
                           y_limit=None, apply_smooth=True, window_size=5):
    """
    统一的科研时序指标绘图函数
    """
    mean_val = np.mean(data_matrix, axis=0)
    lower_95 = np.percentile(data_matrix, 2.5, axis=0)
    upper_95 = np.percentile(data_matrix, 97.5, axis=0)
    lower_99 = np.percentile(data_matrix, 0.5, axis=0)
    upper_99 = np.percentile(data_matrix, 99.5, axis=0)

    if apply_smooth:
        mean_val = smooth_data(mean_val, window_size)
        lower_95, upper_95 = smooth_data(lower_95, window_size), smooth_data(upper_95, window_size)
        lower_99, upper_99 = smooth_data(lower_99, window_size), smooth_data(upper_99, window_size)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)

    for spine in ax.spines.values():
        spine.set_capstyle('projecting') 

    if highlight_zone:
        start, end = highlight_zone
        ax.axvspan(start, end, color='gray', alpha=0.12, label=highlight_text, edgecolor='none', zorder=0)

    ax.fill_between(steps, lower_99, upper_99, color=color_fill, alpha=0.15, label='99% CI', edgecolor='none', zorder=1)
    ax.fill_between(steps, lower_95, upper_95, color=color_fill, alpha=0.3, label='95% CI', edgecolor='none', zorder=2)
    ax.plot(steps, mean_val, color=color_main, lw=2, label=legend_label, zorder=3)

    ax.set_xlabel('Simulation Steps', fontsize=8, fontweight='bold', labelpad=4)
    ax.set_ylabel(ylabel, fontsize=8, fontweight='bold', labelpad=2)
    ax.tick_params(axis='both', which='major', labelsize=6, size=3, width=1.0, pad=2)
    
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=15) 
    
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=6, edgecolor='gray')
    
    ax.set_xlim(min(steps), max(steps))
    if y_limit:
        ax.set_ylim(y_limit)
    else:
        ax.set_ylim(0, np.max(upper_99) * 1.35)

    ax.tick_params(axis='both', which='major', labelsize=8, size=4, width=1.2)
    ax.set_ylim(0.0, 0.03)
    ax.set_yticks([0.00, 0.01, 0.02, 0.03])
    ax.set_xlim(0, 600)
    ax.set_xticks(np.arange(100, 600, 100))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.tight_layout()
    os.makedirs("Paper_Figures", exist_ok=True)
    out_path = os.path.join("Paper_Figures", save_filename)
    plt.savefig(out_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved vector plot with corrected borders and ticks to {out_path}")


R0 = 0.2
W_SHAPE_IDEAL = 0.12
K_DIFF_IDEAL = 0.05
SHAPE_GAIN = 50.0

GRID_RES = 60  
XY_RANGE = 0.6
X, Y = np.meshgrid(np.linspace(-XY_RANGE, XY_RANGE, GRID_RES), np.linspace(-XY_RANGE, XY_RANGE, GRID_RES))

def get_ideal_beta(cx):
    if cx < -0.7: hw = 0.5
    elif -0.7 <= cx < -0.1: hw = 0.5 + (cx - (-0.7)) / (-0.1 - (-0.7)) * (0.12 - 0.5)
    elif -0.1 <= cx < 0.6: hw = 0.12
    elif 0.6 <= cx < 0.9: hw = 0.12 + (cx - 0.6) / (0.9 - 0.6) * (0.5 - 0.12)
    else: hw = 0.5
    norm_width = np.clip(hw / 0.5, 0.24, 1.0)
    return np.clip(1.0 + (1.0 - norm_width) * 5.0, 1.0, 3.0)

def calc_rho_ideal(beta_ideal):
    C = SHAPE_GAIN * W_SHAPE_IDEAL
    dist_warped = np.sqrt(X**2 + beta_ideal * Y**2)
    U = 0.5 * C * (dist_warped - R0)**2
    rho_th = np.exp(-U / K_DIFF_IDEAL)
    rho_th /= (np.sum(rho_th) + 1e-8)
    return rho_th

def calc_rho_real(x_rel, y_rel):
    positions = np.vstack([x_rel, y_rel])
    positions += np.random.normal(0, 1e-4, positions.shape)
    try:
        kernel = gaussian_kde(positions, bw_method=0.3)
        grid_coords = np.vstack([X.ravel(), Y.ravel()])
        rho_act = kernel(grid_coords).reshape(X.shape)
        rho_act /= (np.sum(rho_act) + 1e-8)
        return rho_act
    except:
        return np.zeros_like(X)

file_pattern = "data/fails_0_*.csv"  
files = glob.glob(file_pattern)
print(f"Found {len(files)} trial files. Starting calculation...")

TARGET_X = 1.05
TARGET_Y = 0.0

all_trials_errors = []
common_steps = None

for file_path in files:
    df = pd.read_csv(file_path)
    steps = sorted(df['step'].unique())
    if common_steps is None: common_steps = steps
    
    trial_errors = []
    
    for s in steps:
        group = df[df['step'] == s]
        cx, cy = group['pos_x'].mean(), group['pos_y'].mean()
        
        if cx > 0.6:
            alpha = np.clip((cx - 0.6) / 0.4, 0.0, 1.0) 
            ref_x = (1 - alpha) * cx + alpha * TARGET_X
            ref_y = (1 - alpha) * cy + alpha * TARGET_Y
        else:
            ref_x = cx
            ref_y = cy
        
        x_rel = group['pos_x'].values - ref_x
        y_rel = group['pos_y'].values - ref_y
        
        beta_ideal = get_ideal_beta(cx)
        rho_ideal = calc_rho_ideal(beta_ideal)
        rho_real = calc_rho_real(x_rel, y_rel)
        
        mse = np.mean((rho_real - rho_ideal)**2) * 10000 
        trial_errors.append(mse)
        
    all_trials_errors.append(trial_errors)

error_matrix = np.array(all_trials_errors)

plot_scientific_metric(
    steps=common_steps, 
    data_matrix=error_matrix, 
    ylabel=r'Divergence ($E_{ADR}$)', 
    save_filename='ADR_Divergence.pdf',
    title=None,
    legend_label='Mean $E_{ADR}$',
    highlight_zone=(180, 300),           
    highlight_text='Narrow Passage',     
    color_main='#1a2a6c', 
    color_fill='#2c408b'
)