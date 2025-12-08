import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from glob import glob
import numpy as np

plt.rcParams.update({
    "font.size": 16,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.major.size": 10,
    "ytick.major.size": 6
})


def remove_outliers_iqr(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return data[(data >= lower) & (data <= upper)]


def get_significance_stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


def make_long_df(values, condition):
    return pd.DataFrame({'value': values, 'condition': condition})


def load_and_concatenate(files):
    """Load multiple npy files; each contains [peaks, means]."""
    data = [np.load(f, allow_pickle=True) for f in files]
    peaks = np.concatenate([d[0] for d in data])
    means = np.concatenate([d[1] for d in data])
    return remove_outliers_iqr(peaks), remove_outliers_iqr(means)


def log_stats(logger, name, arr):
    logger.info(
        f"{name}: n={len(arr)}, min={np.min(arr):.4f}, max={np.max(arr):.4f}, "
        f"mean={np.mean(arr):.4f}, std={np.std(arr):.4f}"
    )


def plot_metric(ax, df, metric_name, p_value, category_order, palette):
    # Boxplot
    sns.boxplot(
        data=df, x='condition', y='value', order=category_order,
        palette=dict(zip(category_order, palette)),
        width=0.5, showfliers=False,
        boxprops={'alpha': 0.9, 'linewidth': 2, 'edgecolor': '#444'},
        medianprops={'linewidth': 2.5, 'color': 'white'},
        whiskerprops={'linewidth': 1.5},
        capprops={'linewidth': 1.5},
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
    ax.scatter(
        range(len(category_order)), means.values,
        marker='D', s=100, facecolor='white', edgecolor='black', zorder=6
    )

    # Curved significance line
    y_max, y_min = df['value'].max(), df['value'].min()
    y_range = y_max - y_min
    bar_y = y_max + y_range * 0.12
    curve_height = y_range * 0.03
    ax.plot([0, 0, 1, 1], [bar_y, bar_y+curve_height, bar_y+curve_height, bar_y],
            lw=1.8, c='black')

    # Significance stars
    ax.text(0.5, bar_y + curve_height + y_range*0.02, get_significance_stars(p_value),
            ha='center', fontsize=20, fontweight='bold')

    # Styling
    ax.set_title(f"P100 {metric_name.capitalize()} Amplitude")
    ax.set_ylabel(f"Amplitude (µV)")
    ax.set_xlabel("")
    ax.set_xticklabels(category_order)
    sns.despine(ax=ax)


def plot_peak_mean_audio_no_audio(logger):
    logger.info('Plotting Peak and Mean amplitudes for Audio and No Audio conditions')
    output_dir = Path('results/images/')
    output_dir.mkdir(parents=True, exist_ok=True)

    category_order = ['Auditory', 'Non Auditory']
    palette = ["#1f77b4", "#ff7f0e"]

    audio_files = sorted(glob('results/N100/*Auditory.npy'))
    no_audio_files = sorted(glob('results/N100/*Non Auditory.npy'))

    audio_peaks, audio_means = load_and_concatenate(audio_files)
    no_audio_peaks, no_audio_means = load_and_concatenate(no_audio_files)

    df_peak = pd.concat([
        make_long_df(audio_peaks, 'Auditory'),
        make_long_df(no_audio_peaks, 'Non Auditory')
    ], ignore_index=True)

    df_mean = pd.concat([
        make_long_df(audio_means, 'Auditory'),
        make_long_df(no_audio_means, 'Non Auditory')
    ], ignore_index=True)

    # T-tests
    t_peak, p_peak = ttest_ind(audio_peaks, no_audio_peaks, equal_var=False)
    t_mean, p_mean = ttest_ind(audio_means, no_audio_means, equal_var=False)
    logger.info(f"T-test Peak: t={t_peak:.3f}, p={p_peak:.3e}")
    logger.info(f"T-test Mean: t={t_mean:.3f}, p={p_mean:.3e}")

    # Log stats
    log_stats(logger, "Auditory Peaks", audio_peaks)
    log_stats(logger, "Auditory Means", audio_means)
    log_stats(logger, "Non-Auditory Peaks", no_audio_peaks)
    log_stats(logger, "Non-Auditory Means", no_audio_means)

    # Plotting
    sns.set(style="whitegrid", font_scale=1.3)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    plot_metric(axes[0], df_peak, 'peak', p_peak, category_order, palette)
    plot_metric(axes[1], df_mean, 'mean', p_mean, category_order, palette)

    plt.tight_layout()
    filepath = output_dir / "peakMeanAmplitudeAudioNoAudio.pdf"
    fig.savefig(filepath, format='pdf', dpi=800, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Plot saved to {filepath}")
