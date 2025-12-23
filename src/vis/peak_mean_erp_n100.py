import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import mne

import pdb
import pandas as pd
from src.utils import load_all_filepaths_from_directory, log_info
from src.analysis.p100 import run_best_condition_analysis, get_best_condition
import re

from scipy.stats import ttest_ind

def set_plot_style():
    """Sets a consistent style for all plots."""
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['axes.titlecolor'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'black' # Ensure box edges are black
    plt.rcParams['grid.alpha'] = 0.3 # Keep grid subtle

def log_stats(logger, name, arr):
    """Logs basic statistics for a given numpy array."""
    arr = arr*1e6
    logger.info(
        f"{name}: n={len(arr)}, min={np.min(arr):.4f}, max={np.max(arr):.4f}, "
        f"mean={np.mean(arr):.4f}, std={np.std(arr):.4f}"
    )
    

def make_long_df(data_array, condition_name):
    """
    Helper to create a long-format DataFrame from an array for seaborn.
    (Simplified for direct DataFrame use)
    """
    return pd.DataFrame({'Amplitude': data_array, 'Condition': condition_name})

def get_window_peak_mean(evoked,  tmin, tmax, picks=None):
    """
    Compute the peak and mean amplitude within a specified time window.
    
    Args:
        data (np.ndarray): EEG data array of shape (n_channels, n_times).
        times (np.ndarray): Time vector corresponding to the data.
        tmin (float): Start time of the window (in seconds).
        tmax (float): End time of the window (in seconds).
    """
    
    if picks is not None:
        data = evoked.copy().pick(picks)
    else:
        data = evoked.copy()
    times = evoked.times
    time_mask = (times >= tmin) & (times <= tmax)
    data = data.average().get_data()
   
   
    window_data = data[:, time_mask]    
    peak_amplitudes = np.max(window_data, axis=1)
    mean_amplitudes = np.mean(window_data, axis=1)
    
    
    return peak_amplitudes.mean(), mean_amplitudes.mean()


def plot_n100_mean_peak(config, logger):
    set_plot_style()
    log_info(logger, "Plotting Mean and Peak P100 across subjects")
    results_dir = Path(config['analysis']['results_dir'], 'n100', 'data')
    
    audio_files = load_all_filepaths_from_directory(results_dir, extensions=['.fif'], startswith='audio')
    no_audio_files = load_all_filepaths_from_directory(results_dir, extensions=['.fif'], startswith='no-audio')

    audio_files = sorted(audio_files)
    no_audio_files = sorted(no_audio_files)
    
    region, tmins, tmaxs, audio_mean, audio_peak, no_vs_mean, no_vs_peak = [], [], [], [], [], [], []
    subjects, sessions = [], []
    
    for index in range(len(audio_files)):
        
        match = re.search(r'(sub-\d+)-(ses-\d+)', audio_files[index])
    
        if match:
            sub_id = match.group(1) # e.g., 'sub-01'
            ses_id = match.group(2) # e.g., 'ses-01'
        
        if sub_id == '11':
            continue
            
        subjects.append(sub_id)
        sessions.append(ses_id)
        logger.info(audio_files[index])
        logger.info(no_audio_files[index])
        
        audio_data = mne.read_epochs(audio_files[index])
        no_audio_data = mne.read_epochs(no_audio_files[index])
        
        
        
        tmin = config['analysis']['n100']['window'][0]
        tmax = config['analysis']['n100']['window'][1]
        roi = config['analysis']['n100']['selectd_channels']
        picks = config['analysis']['n100']['rois'][roi]
        tmins.append(tmin)
        tmaxs.append(tmax)
        region.append(roi)

        
        
        peak, mean = get_window_peak_mean(
            audio_data,
            tmin=tmin, 
            tmax=tmax,
            picks=picks
        )
        audio_mean.append(mean)
        audio_peak.append(peak)
        logger.info(f'{roi}, tmin: {tmin}, tmax: {tmax}')
        logger.info(f'Audio Subject {index+1}: Peak Amplitudes: {peak*1e6}, Mean Amplitudes: {mean*1e6}')
        

        peak, mean = get_window_peak_mean(
            no_audio_data,
            tmin=tmin, 
            tmax=tmax, 
        )
        
        no_vs_mean.append(mean)
        no_vs_peak.append(peak)
        logger.info(f'No audio Subject {index+1}: Peak Amplitudes: {peak*1e6}, Mean Amplitudes: {mean*1e6}')
        
    data = {
        'Subject_ID': subjects,  
        'Session_ID': sessions,
        'Region': region,
        'Tmin (s)': tmins,
        'Tmax (s)': tmaxs,
        'Audio Mean (µV)': audio_mean,
        'Audio Peak (µV)': audio_peak,
        'No audio Mean (µV)': no_vs_mean,
        'No audio Peak (µV)': no_vs_peak
    }

    # Create the DataFrame
    df_results = pd.DataFrame(data)

    
    plot_peak_mean_audio_noaudio_from_df(df_results, logger)
    

    
def plot_metric(ax, df_metric, metric_type, p_value):
    """
    Plots a boxplot for a given metric, overlays mean values,
    and adds p-value significance bars.
    """
    sns.despine(ax=ax)

    sns.boxplot(
        x='Condition',
        y='Amplitude',
        data=df_metric,
        ax=ax,
        palette={'Audio': 'skyblue', 'No audio Change': 'lightcoral'},
        width=0.6,
        fliersize=3,
        boxprops=dict(edgecolor='black', linewidth=2.5),
        whiskerprops=dict(color='black', linewidth=2.5),
        capprops=dict(color='black', linewidth=2.5),
        medianprops=dict(color='black', linewidth=2.5),
        flierprops=dict(markeredgecolor='black', linewidth=1.5)
    )


    sns.stripplot(
        x='Condition',
        y='Amplitude',
        data=df_metric,
        ax=ax,
        color='black',
        alpha=0.6,
        jitter=True,
        s=6
    )

    # ---- Mean overlay ----
    
    
    mean_df = df_metric.groupby('Condition', as_index=False)['Amplitude'].mean()
    ax.scatter(
        x=mean_df['Condition'],
        y=mean_df['Amplitude'],
        color='red',
        s=80,
        marker='s',
        zorder=10,
        label='Mean'
    )

    ax.set_title(f'{metric_type.capitalize()} Amplitude', fontsize=24, fontweight='bold')
    ax.set_ylabel('Amplitude (µV)', fontsize=22)
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    

    # ---- p-value significance bar ----
    y_max = df_metric['Amplitude'].max()
    y_min = df_metric['Amplitude'].min()
    y_range = y_max - y_min
    bar_y = y_max + y_range * 0.05
    bar_height = y_range * 0.02

    # Draw line connecting groups
    ax.plot([0, 0, 1, 1], [bar_y, bar_y + bar_height, bar_y + bar_height, bar_y],
            lw=1.5, c='k')

    # Annotate p-value above the line
        
    if p_value < 0.001:
        p_text = "***"
    elif p_value < 0.01:
        p_text = "**"
    elif p_value < 0.05:
        p_text = "*"
    else:
        p_text = "ns"

    ax.text(0.5, bar_y + bar_height + y_range * 0.01,
            p_text, ha='center', va='bottom', fontsize=14)

    ax.legend(loc='upper right', frameon=True, bbox_to_anchor=(1, 0.90), fontsize=22)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(2.5)
        
    for label in ax.get_xticklabels():
        label.set_fontsize(16)
        label.set_fontweight('bold')


def plot_peak_mean_audio_noaudio_from_df(df_input, logger):
    """
    Plots P100 peak and mean amplitudes from the provided DataFrame,
    performs t-tests, logs statistics, and saves the plot.

    Parameters:
    df_input (pd.DataFrame): The input DataFrame containing 'audio Mean (V)',
                             'audio Peak (V)', 'No audio Mean (V)',
                             and 'No audio Peak (V)' columns.
    """
    logger.info('Plotting P100 peak and mean amplitudes')

    output_dir = Path('results/images/')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data directly from the input DataFrame
    audio_peaks = df_input['Audio Peak (µV)'].to_numpy()
    audio_means = df_input['Audio Mean (µV)'].to_numpy()
    no_audio_peaks = df_input['No audio Peak (µV)'].to_numpy()
    no_audio_means = df_input['No audio Mean (µV)'].to_numpy()

    # Create long-form DataFrames for plotting
    df_peak = pd.concat([
        make_long_df(audio_peaks*1e6, 'Audio'),
        make_long_df(no_audio_peaks*1e6, 'No audio Change')
    ], ignore_index=True)

    df_mean = pd.concat([
        make_long_df(audio_means*1e6, 'Audio'),
        make_long_df(no_audio_means*1e6, 'No audio Change')
    ], ignore_index=True)

    # T-tests
    logger.info('='*50)
    t_peak, p_peak = ttest_ind(audio_peaks, no_audio_peaks, equal_var=False)
    t_mean, p_mean = ttest_ind(audio_means, no_audio_means, equal_var=False)
    logger.info(f"T-test Peak: t={t_peak:.3f}, p={p_peak:.3e}")
    logger.info(f"T-test Mean: t={t_mean:.3f}, p={p_mean:.3e}")
    logger.info('='*50)

    # Log stats
    
    logger.info("Summary statistics for audio condition")
    log_stats(logger, "audio Peaks", audio_peaks)
    log_stats(logger, "audio Means", audio_means)
    logger.info('='*50)
    logger.info("Summary statistics for No audio Change condition")
    log_stats(logger, "No-audio Peaks", no_audio_peaks)
    log_stats(logger, "No-audio Means", no_audio_means)
    logger.info('='*50)

    # Plotting
    sns.set(style="whitegrid", font_scale=1.3)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=False) # Changed sharey to False for independent scales

    plot_metric(axes[0], df_peak, 'peak', p_peak)
    plot_metric(axes[1], df_mean, 'mean', p_mean)

    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=22)  # change 14 to desired size
        ax.tick_params(axis='both', which='minor', labelsize=22) 

    plt.tight_layout()
    filepath = output_dir / "peakMeanAmplitudeaudioRest.pdf"
    fig.savefig(filepath, format='pdf', dpi=800, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Plot saved to {filepath}")
    
    # Display the plot
    plt.show() # Add this to display the plot immediately
