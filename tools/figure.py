import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.cm as cm
import pandas as pd
import pickle

plt.rc('font', family='Times New Roman')

def plot_trajectory_from_pkl(pkl_path, lat_min, lat_max, lon_min, lon_max, save_path=None):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    fig = plt.figure(figsize=(14, 12))
    ax = plt.gca()
    cmap = cm.get_cmap('coolwarm')
    for i, ship_data in enumerate(data):
        traj = ship_data['traj']
        lon = traj[:, 1]
        lat = traj[:, 0]
        # 反归一化
        lat_orig = lat_min + lat * (lat_max - lat_min)
        lon_orig = lon_min + lon * (lon_max - lon_min)
        color = cmap(1 - i / len(data))
        ax.plot(lon_orig, lat_orig, color=color, linewidth=1, alpha=0.65)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_cog_sog_comparison(cog_csv, sog_csv, num=1, save_dir=None):
    cog = pd.read_csv(cog_csv) * 360
    sog = pd.read_csv(sog_csv) * 30

    def map_360_to_pm180(x):
        return (x + 180) % 360 - 180

    cog['COG_true'] = map_360_to_pm180(cog['COG'])
    cog['COG_pred_STGT'] = map_360_to_pm180(cog['COG_STGT'])
    cog['COG_pred_Tr'] = map_360_to_pm180(cog['COG_Tr'])

    sog['SOG_true'] = sog['SOG']
    sog['SOG_pred_STGT'] = sog['SOG_STGT']
    sog['SOG_pred_Tr'] = sog['SOG_Tr']

    fz = 15
    fig1, ax1 = plt.subplots(figsize=(10, 3.5))
    ax1.plot(cog.index, cog['COG_true'], 'o-', markersize=3, label='Actual')
    ax1.plot(cog.index, cog['COG_pred_Tr'], '*-', markersize=3, label='TrAISformer')
    ax1.plot(cog.index, cog['COG_pred_STGT'], 'x-', markersize=3, color='red', label='DSTGT-Net')
    ax1.set_xlabel('Time Steps', fontsize=fz)
    ax1.set_ylabel('Course Over Ground (°)', fontsize=fz)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlim(-0.1, 60.1)
    ax1.legend(prop={'size': 12}, ncol=3, loc='best')
    ax1.grid(alpha=0.3)
    if save_dir:
        plt.savefig(f"{save_dir}/cog_{num}.png", dpi=660, bbox_inches='tight')

    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    ax2.plot(sog.index, sog['SOG_true'], 'o-', markersize=3, label='Actual')
    ax2.plot(sog.index, sog['SOG_pred_Tr'], '*-', markersize=3, label='TrAISformer')
    ax2.plot(sog.index, sog['SOG_pred_STGT'], 'x-', markersize=3, color='red', label='DSTGT-Net')
    ax2.set_xlabel('Time Steps', fontsize=fz)
    ax2.set_ylabel('Speed Over Ground (kt)', fontsize=fz)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlim(-0.1, 60.1)
    ax2.legend(prop={'size': 12}, ncol=3, loc='best')
    ax2.grid(alpha=0.3)
    if save_dir:
        plt.savefig(f"{save_dir}/sog_{num}.png", dpi=660, bbox_inches='tight')
    plt.show()

def plot_prob_distribution(csv_path, col_names, titles, xlim=(10,20), save_path=None):
    df = pd.read_csv(csv_path) * 30.0
    bins = np.linspace(0, 30.0, 30)
    labels = bins[:-1]
    fig, axes = plt.subplots(1, len(col_names), figsize=(15, 4), sharey=True)
    for ax, col, title in zip(axes, col_names, titles):
        counts, _ = np.histogram(df[col], bins=bins)
        probs = counts / counts.sum()
        ax.bar(labels, probs, width=0.1, color='steelblue', edgecolor='black')
        ax.set_title(title)
        ax.set_xlabel('Speed (m/s)')
        ax.set_xlim(*xlim)
        if ax == axes[0]:
            ax.set_ylabel('Probability')
    plt.suptitle('Probability distribution', y=1.02)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_error_bars(metric_data, algorithms, scenarios, ylabel, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(scenarios))
    width = 0.1
    colors = [(183/255,34/255,48/255), (220/255,109/255,87/255), (246/255,178/255,147/255),
              (182/255,215/255,232/255), (109/255,173/255,209/255), (49/255,124/255,183/255)]
    for i, (alg, values) in enumerate(metric_data.items()):
        offset = width * (i - (len(algorithms)-1)/2)
        ax.bar(x + offset, values, width, label=alg, color=colors[i], edgecolor='black', linewidth=0.8)
    ax.set_xlabel('Route', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.legend(fontsize=9, frameon=False, ncol=3, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='-')
    ax.set_axisbelow(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=330, bbox_inches='tight')
    plt.show()