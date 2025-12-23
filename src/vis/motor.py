import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import read_tfrs, AverageTFR
from mne import grand_average, create_info
from src.utils import load_all_filepaths_from_directory 
import pdb
import mne
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator

import glob



from pathlib import Path

def plot_motor_box(logger):
    directory = 'results/motor/numpy'
    logger.info("Plotting Box od motor")
    overt = np.load(Path(directory, 'overt.npy'))
    covert = np.load(Path(directory, 'covert.npy'))
    rest = np.load(Path(directory, 'rest.npy'))
    
    conditions = {
    'Overt Speech': overt,
    'Covert Speech': covert,
    'Rest': rest
    }
        

    # -----------------------------
    # Prepare dataframe
    # -----------------------------
    data_list = []
    for cond_name, values in conditions.items():
        for val in values:
            data_list.append({'Condition': cond_name, 'Value': val})

    df_melt = pd.DataFrame(data_list)
    df_melt['Value'] = df_melt['Value'] * 1e9  # scale if needed

    # -----------------------------
    # Remove outliers per condition (IQR)
    # -----------------------------
    def remove_outliers(group):
        Q1 = group['Value'].quantile(0.25)
        Q3 = group['Value'].quantile(0.75)
        IQR = Q3 - Q1
        return group[
            (group['Value'] >= Q1 - 1.5 * IQR) &
            (group['Value'] <= Q3 + 1.5 * IQR)
        ]

    df_clean = (
        df_melt
        .groupby('Condition', group_keys=False)
        .apply(remove_outliers)
    )

    # -----------------------------
    # Plot styling
    # -----------------------------
    sns.set_style("whitegrid")
    sns.set_context("talk")

    palette = ["#4C72B0", "#55A868", "#C44E52"]
    conditionsOrder = ["Rest", "Covert Speech", "Overt Speech"]

    plt.figure(figsize=(18, 10))
    ax = sns.boxplot(
        x='Condition',
        y='Value',
        data=df_clean,
        order=conditionsOrder,  # set desired order
        palette=palette,
        width=0.6,
        fliersize=4,
        showmeans=True,
        meanprops={
            "marker": "s",
            "markerfacecolor": "red",
            "markeredgecolor": "black",
            "markersize": 12
        },
        medianprops={
            "color": "black",
            "linewidth": 3
        }
    )

    # -----------------------------
    # Axis labels and ticks
    # -----------------------------
    ax.set_xlabel('')
    ax.set_ylabel(
        r'Mean HF Power ($\mathbf{20-300}$ Hz) × $\mathbf{10^{-3}}$',
        fontsize=24,
        fontweight='bold',
        labelpad=10
    )

    ax.tick_params(axis='both', labelsize=20)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(30)

    sns.despine(trim=True)

    # -----------------------------
    # Statistical annotations
    # -----------------------------
    pairs = [
        ("Overt Speech", "Covert Speech"),
        ("Overt Speech", "Rest"),
        ("Covert Speech", "Rest")
    ]

    annotator = Annotator(
        ax,
        pairs,
        data=df_clean,
        x='Condition',
        y='Value',
        order=conditionsOrder  # ensure correct order for stats
    )
    annotator.configure(
        test='t-test_ind',
        text_format='star',
        loc='inside',
        fontsize=26,
        color='black',
        verbose=0
    )
    annotator.apply_and_annotate()

    # -----------------------------
    # Compute means and add legend
    # -----------------------------
    meanVals = df_clean.groupby('Condition')['Value'].mean()

    legendLabels = [
        f"{cond}: {meanVals[cond]:.2f}"
        for cond in conditionsOrder
    ]

    handles = [
        plt.Line2D([0], [0], color=palette[i], lw=6)
        for i in range(len(conditionsOrder))
    ]

    legend = ax.legend(
        handles,
        legendLabels,
        title="Mean",
        title_fontsize=22,
        fontsize=20,
        frameon=True,
        loc="center",
        bbox_to_anchor=(1.02, 0.85)
    )

    for l in legend.get_texts():
        l.set_fontweight('bold')

    # -----------------------------
    # Save figure
    # -----------------------------
    plt.tight_layout()
    plt.savefig("results/images/motor_box.pdf", format="pdf", dpi=800)
    plt.show()


    
    
    
def plot_motor_average(logger, picks=None):
    folder = 'results/motor/tfr_batches'
    
    
    tfr_by_cond = {
        'Rest':[],
        'Covert': [],
        'Overt': []
    }
    
    for fname in glob.glob("results/motor/tfr_batches/*.h5"):
        
        power = mne.time_frequency.read_tfrs(fname)       
        cond_name = fname.split('/')[-1].split('_')[0]
        tfr_by_cond[cond_name].append(power.average())
        
        print(f"Loaded: {fname}")
        
    
    tfr_by_cond = {
        'Rest':mne.grand_average(
                tfr_by_cond['Rest'],
                interpolate_bads=False
        ),
        'Covert': mne.grand_average(
                tfr_by_cond['Covert'],
                interpolate_bads=False
        ),
        'Overt': mne.grand_average(
                tfr_by_cond['Overt'],
                interpolate_bads=False
        ),
    }
    
    
    channels_of_interest = ['F7','FT7', 'F8', 'FT8', 'T8','T7', 'TP7', 'TP8', 'P7', 'P8', 'PO7', 'PO8', 'Fp1', 'Fp2']
    n_cond = len(tfr_by_cond)
    
    

    # Create figure with one row and multiple columns
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_cond,
        figsize=(8 * n_cond, 10),
        constrained_layout=True
    )

    # Ensure axes is iterable
    if n_cond == 1:
        axes = [axes]
    titles = ['Rest', 'Covert Speech', 'Overt Speech']
    for i, (ax, (cond_name, power)) in enumerate(zip(axes, tfr_by_cond.items())):
        print(f"--- Visualizing Condition: {cond_name} ---")

        # Plot mean TFR into the given axis
        power.plot(
            picks=channels_of_interest,
            baseline=(-0.2, 0),
            mode="logratio",
            combine="mean",
            vlim=(-0.6, 0.6),
            axes=ax,
            colorbar=(i == n_cond - 1),  # colorbar only on last subplot
            show=False
        )


        # Y-ticks only on first subplot
        if i != 0:
            ax.set_yticks([])
            ax.set_ylabel("")
        if i == 0:
            ax.set_ylabel('Frequency (Hz)', fontsize=26, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=26, fontweight='bold')
        ax.set_title(titles[i],fontsize=30, fontweight='bold')
    # Remove top and right spines from all subplots
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for label in ax.get_yticklabels() + ax.get_xticklabels():
                    label.set_fontsize(20)
                    label.set_fontweight('bold')
                    

    cax = fig.axes[-1] 

    cax.tick_params(labelsize=24) 

    # 3. Make those numbers bold
    for label in cax.get_yticklabels():
        label.set_fontweight('bold')

    # 4. Increase the size of the side label ("logratio")
    # MNE usually puts this on the y-axis of the colorbar
    cax.set_ylabel('Amplitue (μV)', fontsize=24, fontweight='bold')

    plt.savefig('results/images/muscular.pdf', format='pdf', dpi=800)


        
        
            