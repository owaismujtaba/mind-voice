import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_rel
from glob import glob


def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        q1 = cleaned_df[col].quantile(0.25)
        q3 = cleaned_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower) & (cleaned_df[col] <= upper)]
    return cleaned_df


def plot_metric(ax, y_col, title, p_value, df_avg):
    category_order = ['Visual', 'No Visual Change']
    xtick_labels = ['Visual', 'No Visual Change']

    sns.boxplot(
        data=df_avg,
        x='condition', y=y_col,
        order=category_order,
        hue='condition',
        palette='Set3',
        width=0.5,
        legend=False,
        boxprops={'alpha': 0.8, 'linewidth': 1.5},
        medianprops={'linewidth': 2, 'color': 'darkblue'},
        ax=ax
    )

    sns.stripplot(
        data=df_avg,
        x='condition', y=y_col,
        order=category_order,
        color='black',
        size=5, jitter=0.15, alpha=0.6,
        ax=ax
    )

    # Means
    means = df_avg.groupby('condition', sort=False)[y_col].mean().reindex(category_order)
    x_pos = range(len(category_order))
    ax.scatter(
        x_pos, means.values,
        marker='s', s=100, color='red', zorder=6, label='Mean'
    )

    ax.set_title(f"{title}\np = {p_value:.3f}", fontsize=20, fontweight='bold')
    ax.set_ylabel(f"{y_col.capitalize()} Amplitude (µV)", fontsize=16, fontweight='bold')
    ax.set_xlabel("")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xtick_labels, fontsize=16, fontweight='bold')

    ax.tick_params(axis='y', labelsize=14)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    sns.despine(ax=ax)


def plot_peak_mean_visual_novisual(logger):
    logger.info('Plotting Peak and Mean amplitudes for visual and no visual conditions')
    output_dir = 'results/images/'
    os.makedirs(output_dir, exist_ok=True)

    category_order = ['Visual', 'No Visual Change']

    csv_files = glob('results/P100/*.csv')
    if len(csv_files) < 1:
        logger.info('No files found to plot')
        return

    all_dfs = [pd.read_csv(f) for f in csv_files]
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.groupby(by=['subject_id', 'condition']).mean().reset_index()

    # Clean outliers
    combined_df = remove_outliers_iqr(combined_df, columns=['peak', 'mean'])

    # Extract for Visual and No Visual Change
    visual = combined_df.loc[combined_df['condition'] == category_order[0], ['mean', 'peak']].reset_index(drop=True)
    novisual = combined_df.loc[combined_df['condition'] == category_order[1], ['mean', 'peak']].reset_index(drop=True)

    #visual = remove_outliers_iqr(visual, columns=['peak', 'mean'])
    #novisual = remove_outliers_iqr(novisual, columns=['peak', 'mean'])
    
    # Match subjects
    n = min(len(visual), len(novisual))
    subject_ids = range(n)

    # Build final dataframe
    df_avg = pd.DataFrame({
        'subject_id': list(subject_ids) * 2,
        'condition': ['Visual'] * n + ['No Visual Change'] * n,
        'mean': pd.concat([visual['mean'][:n], novisual['mean'][:n]]).values,
        'peak': pd.concat([visual['peak'][:n], novisual['peak'][:n]]).values
    })

    # Pivot for paired t-tests
    peak_pivot = df_avg.pivot(index='subject_id', columns='condition', values='peak')
    mean_pivot = df_avg.pivot(index='subject_id', columns='condition', values='mean')

    # Paired t-tests (correct labels)
    t_peak, p_peak = ttest_rel(
        peak_pivot['No Visual Change'],
        peak_pivot['Visual'],
        nan_policy='omit'
    )
    t_mean, p_mean = ttest_rel(
        mean_pivot['No Visual Change'],
        mean_pivot['Visual'],
        nan_policy='omit'
    )

    # Visualization
    sns.set(style="whitegrid", font_scale=1.4)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))

    plot_metric(axes[0], 'peak', 'P100 Peak Amplitude', p_peak, df_avg)
    plot_metric(axes[1], 'mean', 'P100 Mean Amplitude', p_mean, df_avg)

    handles, labels = axes[1].get_legend_handles_labels()
    if 'Mean' in labels:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=14)

    plt.tight_layout()

    name = 'peakMeanAmplitudeVisualRest.pdf'
    filepath = Path(output_dir, name)
    plt.savefig(filepath, format='pdf', dpi=800)
    plt.show()
    plt.close(fig)

    logger.info(f"T-test (Peak): t={t_peak:.3f}, p={p_peak:.3e}")
    logger.info(f"T-test (Mean): t={t_mean:.3f}, p={p_mean:.3e}")
    logger.info('Visual')
    logger.info(visual.mean())
    logger.info('No visual Change')
    logger.info(novisual.mean())
    logger.info(f'Plot saved to {filepath}')
