import seaborn as sns
import os
import glob
import re
import pandas as pd
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



def plot_accuracy_per_subject(df):
    """
    Plots the mean validation accuracy per subject with enhanced aesthetics.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'subject_id' and 'accuracy'.
    """
    # Compute mean accuracy per subject and sort
    subject_acc = df.groupby('subject_id')['accuracy'].mean().reset_index()
    subject_acc = subject_acc.sort_values('accuracy')

    plt.figure(figsize=(16, 8))
    palette = sns.color_palette("viridis", len(subject_acc))

    # Create barplot
    ax = sns.barplot(
        x='subject_id',
        y='accuracy',
        data=subject_acc,
        palette=palette,
        edgecolor='black',
        linewidth=1.5
    )

    # Set font sizes, labels, and make them bold
    ax.set_xlabel('Subject ID', fontsize=20, fontweight='bold', labelpad=15)
    ax.set_ylabel('Validation Accuracy', fontsize=20, fontweight='bold', labelpad=15)

    # Set tick sizes and make ticks bold
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add horizontal gridlines
    ax.yaxis.grid(True, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)

    # Annotate bars with accuracy values
    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2., 
            height + 0.005,  # small offset above bar
            f'{height:.3f}', 
            ha='center', 
            va='bottom',
            fontsize=14,
            fontweight='bold'
        )

    # Tight layout, save, and show
    plt.tight_layout()
    plt.savefig('results/images/overt_covert_rest_accuracy.png', dpi=800)
    plt.show()

def load_decoding_accuracies(results_dir):
    files = glob.glob(results_dir)
    records = []
    print(files)
    # Regex to pull out sub-XX and ses-XX
    filename_re = re.compile(r"sub-([0-9a-zA-Z]+)_ses-([0-9a-zA-Z]+)_.*_accuracy\.csv")

    for filepath in files:
        print(filepath)
        fname = os.path.basename(filepath)
        m = filename_re.match(fname)
        if not m:
            # skip files that don't match the naming convention
            continue

        subject_id, session_id = m.groups()

        # Read the CSV
        df = pd.read_csv(filepath)

        # Try to grab an 'accuracy' column, otherwise fall back to first cell
        if "accuracy" in df.columns:
            accuracy = df["accuracy"].iloc[0]
        else:
            accuracy = df.iloc[0, 0]

        records.append({
            "subject_id": subject_id,
            "session_id": session_id,
            "accuracy": float(accuracy)
        })

    return records


def plot_accuracy(logger):
    logger.info('Plotting accuracy plot')
    accuracy = load_decoding_accuracies("results/DecodingResults/*_accuracy.csv")
    accuracy = pd.DataFrame(results)
    plot_accuracy_per_subject(accuracy)
    subj_acc = accuracy.groupby('subject_id')['accuracy'].mean().reset_index()
    print(subj_acc.describe())
    




















