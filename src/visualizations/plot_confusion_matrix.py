import seaborn as sns
import os
import glob
import re
import pandas as pd
import numpy as np  
from matplotlib import pyplot as plt


# Global plot settings
plt.rcParams.update({
    "font.size": 14,         # increase general font size
    "axes.labelsize": 20,    # axis label font size
    "axes.titlesize": 18,    # title font size
    "xtick.labelsize": 18,   # x tick labels
    "ytick.labelsize": 18,   # y tick labels
    "figure.dpi": 600,       # high-resolution figure
})


def load_decoding_cm():
    folder_path = "results/DecodingResults"
    pattern = os.path.join(folder_path, "sub-*_ses-*_confusion_matrix.csv")

    # Find all matching files
    csv_files = glob.glob(pattern)

    # Load and aggregate confusion matrices
    confusion_matrices = []
    for file in csv_files:
        df = pd.read_csv(file)
        confusion_matrices.append(df.values)

    # Convert to numpy array and sum
    aggregated_confusion = np.sum(confusion_matrices, axis=0)

    labels = ["Overt", "Covert", "Rest"]
    aggregated_confusion = pd.DataFrame(aggregated_confusion, columns=labels)
    aggregated_confusion.index = labels
    
    return aggregated_confusion


def plot_cmfusion_matrix_heatmap(cm):
    # Example heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,              
        fmt=".0f",               
        annot_kws={"size": 18, "weight": "bold"},  
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=0.5
    )
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')

    plt.xlabel("Predicted Label", fontweight='bold')
    plt.ylabel("True Label", fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/images/cm.pdf', format='pdf', dpi=800)

def plot_confusion_matrix(logger):
    logger.info('Plotting confusion matrix')
    cm = load_decoding_cm()
    plot_cmfusion_matrix_heatmap(cm)
    logger.info('Confusion matrix plot saved to results/images/cm.png')
    logger.info('Confusion matrix data:\n%s', cm)
    
    
    




















