from logging import log
import os
from pathlib import Path
from typing import Text
from src.utils import log_info, load_all_filepaths_from_directory

import pdb
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def plot_snr(config, logger):
    
    log_info(logger=logger, text='Running SNR Plotting')
    
    
    filepaths = load_all_filepaths_from_directory(
        directory='results/snr',
        extensions='json',
    )
    
    
    all_data = []

    for filepath in filepaths:
        # Extract subject and session from filename
        parts = filepath.split("_")
        subject = parts[0].split("-")[1]
        
        session = parts[1].split("-")[1]

        with open(filepath, 'r') as f:
            data = json.load(f)

        df = pd.DataFrame({
            "subject": subject,
            "session": session,
            "channel": data["channel_names"],
            "snr_db": data["snr_chan_db"],
            "baseline_var": data["baseline_var"],
            "signal_var": data["signal_var"],
            "baseline_start": data["baseline_window"][0],
            "baseline_end": data["baseline_window"][1],
            "signal_start": data["signal_window"][0],
            "signal_end": data["signal_window"][1]
        })
        all_data.append(df)

    # Concatenate all subjects/sessions
    full_df = pd.concat(all_data, ignore_index=True)
    full_df.to_csv('snr.csv')        
    

    # Set the figure size
    fig, ax = plt.subplots(figsize=(15, 8))

    # 1. Create the boxplot
    # We use showfliers=False to avoid plotting outliers twice
    # color="white" or a light gray makes the individual points stand out better
    sns.boxplot(
        data=full_df, 
        x="channel", 
        y="snr_db", 
        color="#D3D0D0",  # Light gray for boxes
        showfliers=False,
        width=0.6,        # Slightly narrower boxes
        linewidth=1.5,    # Thicker box lines
        boxprops=dict(edgecolor='black'), # Black edges for boxes
        medianprops=dict(color='green', linewidth=3) ,
        showmeans=True,        # Display the mean marker
        meanprops={
            "marker":"s", 
            "markerfacecolor":"red", 
            "markeredgecolor":"black", 
            "markersize":12
        }
    )

    # 2. Add the individual data points (Stripplot)
    sns.stripplot(
        data=full_df, 
        x="channel", 
        y="snr_db", 
        hue="session",      
        palette="deep",   # A different, often pleasing palette
        alpha=0.8, 
        jitter=0.2,       # Control jitter spread
        dodge=True,         
        size=7,           # Increase point size
        linewidth=1,      # Add a thin line around points
        edgecolor='gray'  # Edge color for points
    )
    
    
    # --- Formatting for readability ---
    plt.xticks(rotation=0, ha='right', fontsize=18, fontweight='bold', color='black') # Increased font size
    plt.yticks(fontsize=18,fontweight='bold', color='black') # Increased font size

    # Remove title
    # plt.title("Channel SNR (dB): Boxplot with Individual Data Points") 

    plt.xlabel("Channel Name", fontsize=22, fontweight='bold', color='black') # Increased font size
    plt.ylabel("SNR (dB)", fontsize=22, fontweight='bold', color='black')   # Increased font size

    plt.ylim([0, full_df['snr_db'].max()+3])
    # Remove legend
    ax.legend_.remove()

    # Remove top and right axis spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make bottom and left spines thicker and black
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(2.5)
    
    ax.tick_params(axis='x', width=2.5, length=7, color='black', labelsize=22) # Thicker and longer x-axis ticks
    ax.tick_params(axis='y', width=2.5, length=7, color='black', labelsize=22) # Thicker and longer y-axis ticks


    plt.tight_layout()
    plt.legend([plt.Line2D([0], [0], marker='s', color='w', label='Mean',
                       markerfacecolor='red', markeredgecolor='black', markersize=8)],
           ['Mean'], loc='upper right')
    
    # Save the figure
    legend = plt.legend(
        [plt.Line2D([0], [0], marker='s', color='w',
                    markerfacecolor='red', markeredgecolor='black')],
        ['Mean'],
        loc='center right',
        fontsize=22,        # text size
        markerscale=1.8,    # marker size in legend
        handlelength=1.5,   # length of legend handle
        frameon=True
    )

    legend.get_texts()[0].set_fontweight('bold')
    
    plt.tight_layout()
    
    plt.savefig("results/images/channel_snr_distribution.pdf", format='pdf', dpi=800)
    
    
