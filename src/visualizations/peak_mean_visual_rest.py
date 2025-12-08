import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_ind 
from glob import glob
import numpy as np



plt.rcParams.update({
    "font.size": 16,             # Base font size
    "font.weight": "bold",       # Bold font globally
    "axes.labelweight": "bold",  # Bold axis labels
    "axes.titlesize": 18,        # Title font size
    "axes.titleweight": "bold",  # Title bold
    "xtick.labelsize": 16,       # X tick label size
    "ytick.labelsize": 16,       # Y tick label size
    "xtick.direction": "out",    # Ticks point outward
    "ytick.direction": "out",
    "xtick.major.width": 2,      # Tick line width
    "ytick.major.width": 2,
    "xtick.major.size": 10,       # Tick length
    "ytick.major.size": 6
})

def remove_outliers_iqr(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return data[(data >= lower) & (data <= upper)]

def get_significance_stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

def load_and_concatenate(files):
    """Load multiple npy files; each contains [peaks, means]."""
    data = [np.load(f, allow_pickle=True) for f in files]
    peaks = np.concatenate([d[0] for d in data])
    means = np.concatenate([d[1] for d in data])
    return remove_outliers_iqr(peaks), remove_outliers_iqr(means)

def make_long_df(values, condition):
    return pd.DataFrame({'value': values, 'condition': condition})

def plot_metric(ax, df, metric_name, p_value):
    category_order = ['Visual', 'No Visual Change']
    palette = ["#1f77b4", "#ff7f0e"]  # Modern colors

    # Boxplot
    sns.boxplot(
        data=df, x='condition', y='value', order=category_order, 
        palette=palette, width=0.5, showfliers=False,
        boxprops={'alpha': 0.9, 'linewidth': 2, 'edgecolor': '#444'},
        medianprops={'linewidth': 2.5, 'color': 'white'},
        whiskerprops={'linewidth':1.5},
        capprops={'linewidth':1.5},
        ax=ax
    )

    # Stripplot
    df_plot = df.sample(20000) if len(df) > 20000 else df
    sns.stripplot(
        data=df_plot, x='condition', y='value',
        order=category_order, color='black', size=4,
        jitter=0.25, alpha=0.35, ax=ax
    )

    # Plot mean markers
    means = df.groupby('condition')['value'].mean().reindex(category_order)
    ax.scatter(range(2), means.values, marker='D', s=100, color='white', edgecolor='black', zorder=6)

    # Curved significance line
    y_max, y_min = df['value'].max(), df['value'].min()
    y_range = y_max - y_min
    bar_y = y_max + y_range * 0.12
    curve_height = y_range * 0.03
    ax.plot([0, 0, 1, 1], [bar_y, bar_y+curve_height, bar_y+curve_height, bar_y], lw=1.8, c='black')

    # Significance stars
    ax.text(0.5, bar_y + curve_height + y_range*0.02, get_significance_stars(p_value),
            ha='center', fontsize=20, fontweight='bold')

    # Styling
    ax.set_title(f"P100 {metric_name.capitalize()} Amplitude")
    ax.set_ylabel(f"Amplitude (µV)")
    ax.set_xlabel("")
    ax.set_xticklabels(['Visual Change', 'No Visual Change'])
   
    sns.despine(ax=ax)


def log_stats(logger, name, arr):
    logger.info(
        f"{name}: n={len(arr)}, min={np.min(arr):.4f}, max={np.max(arr):.4f}, "
        f"mean={np.mean(arr):.4f}, std={np.std(arr):.4f}"
    )

def plot_peak_mean_visual_novisual(logger):
    logger.info('Plotting P100 peak and mean amplitudes')

    output_dir = Path('results/images/')
    output_dir.mkdir(parents=True, exist_ok=True)

    visual_files = sorted(glob('results/P100/*Visual.npy'))
    no_visual_files = sorted(glob('results/P100/*Change.npy'))

    if not visual_files:
        logger.warning("No files found.")
        return

    visual_peaks, visual_means = load_and_concatenate(visual_files)
    no_visual_peaks, no_visual_means = load_and_concatenate(no_visual_files)

    df_peak = pd.concat([
        make_long_df(visual_peaks, 'Visual'),
        make_long_df(no_visual_peaks, 'No Visual Change')
    ], ignore_index=True)

    df_mean = pd.concat([
        make_long_df(visual_means, 'Visual'),
        make_long_df(no_visual_means, 'No Visual Change')
    ], ignore_index=True)

    # T-tests
    logger.info('='*50)
    t_peak, p_peak = ttest_ind(visual_peaks, no_visual_peaks, equal_var=False)
    t_mean, p_mean = ttest_ind(visual_means, no_visual_means, equal_var=False)
    logger.info(f"T-test Peak: t={t_peak:.3f}, p={p_peak:.3e}")
    logger.info(f"T-test Mean: t={t_mean:.3f}, p={p_mean:.3e}")

    logger.info('='*50)
    logger.info('='*50)
    # Log stats
    logger.info("Summary statistics for Visual condition")
    log_stats(logger, "Visual Peaks", visual_peaks)
    log_stats(logger, "Visual Means", visual_means)
    logger.info('='*50)
    logger.info('='*50)
    logger.info("Summary statistics for No Visual Change condition")
    log_stats(logger, "No-Visual Peaks", no_visual_peaks)
    log_stats(logger, "No-Visual Means", no_visual_means)

    # Plotting
    sns.set(style="whitegrid", font_scale=1.3)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    plot_metric(axes[0], df_peak, 'peak', p_peak)
    plot_metric(axes[1], df_mean, 'mean', p_mean)

    plt.tight_layout()
    filepath = output_dir / "peakMeanAmplitudeVisualRest.pdf"
    fig.savefig(filepath, format='pdf', dpi=800, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Plot saved to {filepath}")
