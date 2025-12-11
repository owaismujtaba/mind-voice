import seaborn as sns
import os
import glob
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pdb

# Global plot settings
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 20,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.dpi": 600,
})


def load_decoding_cm():
    folder_path = "results/DecodingResults"
    pattern = os.path.join(folder_path, "sub-*_ses-*_confusion_matrix.csv")

    csv_files = glob.glob(pattern)

    confusion_matrices = []
    for file in csv_files:
        df = pd.read_csv(file)
        confusion_matrices.append(df.values)

    aggregated_confusion = np.sum(confusion_matrices, axis=0)

    labels = ["Overt", "Covert", "Rest"]
    aggregated_confusion = pd.DataFrame(aggregated_confusion, columns=labels)
    aggregated_confusion.index = labels

    return aggregated_confusion


def plot_cmfusion_matrix_heatmap(cm):
    plt.figure(figsize=(8, 6))

    cm_percent = cm.div(cm.sum(axis=1), axis=0) * 100
    print(cm_percent)
    annot_values = cm_percent.map(lambda x: f"{x:.1f}%")
    print(annot_values)
    sns.heatmap(
        cm_percent,
        annot=annot_values,
        fmt='',
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
    logger.info('Confusion matrix plot saved to results/images/cm.pdf')
    logger.info('Confusion matrix data:\n%s', cm)


# Example call (remove if using within a pipeline)
# plot_confusion_matrix(logger)
