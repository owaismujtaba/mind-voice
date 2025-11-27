import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import glob

from src.utils.graphics import log_print

def plot_snr_visual(logger, config):
    log_print(text=' Plotting SNR Visual ', logger=logger)
    
    directory = config['analysis']['results_dir']
    directory = Path(directory, 'SNR')
    
    files = glob.glob('results/SNR/*csv')
    visual = [f for f in files if 'visual' in f]
    
    visual_data = []

    for index in range(len(visual)): 
        filepath = visual[index]
        logger.info(f'Loading {filepath}')
        subject = filepath.split('/')[-1].split('_')[0]
        session = filepath.split('/')[-1].split('_')[1].split('_')[0]
        df = pd.read_csv(filepath)
        df['subject'] = subject
        df['session'] = session
        visual_data.append(df)
        
    visual_data = pd.concat(visual_data)
        
    sns.set(style="whitegrid", context="talk")

    channels = visual_data['Channel'].unique()
    n_channels = len(channels)

    # Map channel names to colors
    palette = dict(zip(channels, sns.color_palette("Set2", n_channels)))

    fig, axes = plt.subplots(
        1, n_channels,
        figsize=(3.2 * n_channels, 6),
        sharey=True
    )

    # Ensure axes is iterable even if n_channels = 1
    axes = np.atleast_1d(axes)

    for i, ch in enumerate(channels):
        ax = axes[i]

        # Prepare data
        visual_ch = (
            visual_data[visual_data['Channel'] == ch]
            .groupby(['subject', 'Channel'], as_index=False)
            .mean(numeric_only=True)
        )
        
        # Draw boxplot
        sns.boxplot(
            data=visual_ch,
            x='Channel',
            y='SNR_dB',
            ax=ax,
            width=0.5,
            fliersize=0,
            color=palette[ch]
        )
        
        # Overlay strip plot
        sns.stripplot(
            data=visual_ch,
            x='Channel',
            y='SNR_dB',
            ax=ax,
            color='black',
            size=6,
            alpha=0.6,
            jitter=0.15
        )
        
        # Plot channel mean (optional, can remove if you want)
        mean_val = visual_ch['SNR_dB'].mean()
        ax.scatter(
            0, mean_val,
            color='black',
            marker='s',
            s=80,
            zorder=10
        )

        # Remove titles and x-axis labels
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_xticklabels([''])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if i > 0:
            ax.spines['left'].set_visible(False)

    axes[0].set_ylabel('SNR (dB)', fontsize=18, fontweight='bold')

    for ax in axes:
        ax.tick_params(axis='both', labelsize=14, width=1.5)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight('bold')
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.5)

    # Create legend using channel names and colors
    handles = [plt.Line2D([0], [0], color=palette[ch], lw=6) for ch in channels]
    plt.legend(handles, channels, fontsize=14, loc='upper right')

    plt.tight_layout()
    plt.savefig('results/images/snr_visual.pdf', format='pdf', dpi=1200)
    logger.info('Plot Saved')
