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




def plot_p100_grand(config, logger):
    set_plot_style()
    log_info(logger, "Plotting Mean and Peak P100 across subjects")
    results_dir = Path(config['analysis']['results_dir'], 'p100', 'data')
    
    visual_files = load_all_filepaths_from_directory(results_dir, extensions=['.fif'], startswith='visual')
    no_visual_files = load_all_filepaths_from_directory(results_dir, extensions=['.fif'], startswith='no-visual')

    visual_files = sorted(visual_files)
    no_visual_files = sorted(no_visual_files)
    
    region, tmins, tmaxs, = [], [], []
    subjects, sessions = [], []
    full_visual_data = []
    full_no_visual_data = []
    
    for index in range(len(visual_files)):
        
        match = re.search(r'(sub-\d+)-(ses-\d+)', visual_files[index])
    
        if match:
            sub_id = match.group(1) # e.g., 'sub-01'
            ses_id = match.group(2) # e.g., 'ses-01'
        
        if sub_id == '11':
            continue
            
        subjects.append(sub_id)
        sessions.append(ses_id)
        logger.info(visual_files[index])
        logger.info(no_visual_files[index])
        
        visual_data = mne.read_epochs(visual_files[index])
        no_visual_data = mne.read_epochs(no_visual_files[index])
        
        
        
        tmin = config['analysis']['p100']['window'][0]
        tmax = config['analysis']['p100']['window'][1]
        roi = config['analysis']['p100']['selectd_channels']
        picks = config['analysis']['p100']['rois'][roi]
        tmins.append(tmin)
        tmaxs.append(tmax)
        region.append(roi)
        
        full_visual_data.append(visual_data.average().pick(picks).data.mean(axis=0))
        full_no_visual_data.append(no_visual_data.average().pick(picks).data.mean(axis=0))
        
           
    plot_grand_full(np.array(full_visual_data), np.array(full_no_visual_data), times=visual_data.times, logger=logger)


def plot_grand_full(visual, no_visual, times, logger):
    import os
    set_plot_style()
    output_dir = 'results/images/'
    os.makedirs(output_dir, exist_ok=True)
    time_window = (0.08, 0.12)
    
    mean_visual_uv = visual.mean(axis=0)
    mean_rest_uv = no_visual.mean(axis=0)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(times, mean_visual_uv*1e6, label='Visual Change', color='#0072B2', linewidth=2.5)
    plt.plot(times, mean_rest_uv*1e6, label='No Visual Change', color="#D55E00", linewidth=2.5, linestyle='--')

    # Vertical line at 0
    plt.axvline(0, color='k', linestyle='--', linewidth=2.0, label='Onset (0 s)')

    # Highlight 80-120 ms
    plt.axvspan(0.08, 0.12, color="#A9C7B1", alpha=0.2, label='Window (80-120 ms)')

    # Labels and title
    plt.xlabel('Time (s)', fontsize=16, fontweight='bold')
    plt.ylabel('Amplitude (µV)', fontsize=16, fontweight='bold')

    # Grid
    plt.grid(alpha=0.3, linestyle='--')

    # Customize ticks
    plt.xticks(np.arange(-0.1, 0.5, 0.1), fontsize=16, fontweight='bold', color='black')
    plt.yticks(fontsize=14, fontweight='bold', color='black')

    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    plt.legend(frameon=False, fontsize=16)

    plt.tight_layout()
    
    name = os.path.join(output_dir, 'grand_erp_rest_visual.pdf')
    plt.savefig(name, format='pdf', dpi=800)
    logger.info('Completed Grand ERP Rest Visualization')
    logger.info(f'Grand ERP plot saved at: {name}')
    
