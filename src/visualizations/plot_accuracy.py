import seaborn as sns
import os
import glob
import re
import pandas as pd
from matplotlib import pyplot as plt
import pdb


# Global plot settings for Publication Quality
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.labelsize": 20,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.dpi": 600,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
})

def plot_accuracy_per_subject(df):
    subject_acc = df.groupby('subject_id')['accuracy'].mean().reset_index()
    subject_acc = subject_acc.sort_values('accuracy')

    plt.figure(figsize=(16, 8))
    bar_color = "#4c72b0"

    ax = sns.barplot(
        x='subject_id',
        y='accuracy',
        data=subject_acc,
        color=bar_color,
        edgecolor='black',
        linewidth=1.5
    )

    ax.set_xlabel('Subject ID', fontsize=20, fontweight='bold', labelpad=15)
    ax.set_ylabel('Validation Accuracy', fontsize=20, fontweight='bold', labelpad=15)

    ax.tick_params(axis='x', labelsize=18, length=6)
    ax.tick_params(axis='y', labelsize=18, length=6)

    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, color='grey', alpha=0.7)
    ax.set_axisbelow(True)

    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2.,
            height + 0.005,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=16,
            fontweight='bold',
            color='black'
        )

    mean_acc = subject_acc['accuracy'].mean()
    ax.axhline(
        y=mean_acc,
        linestyle='--',
        linewidth=4,
        color='green',
        label=f'Mean Accuracy ({mean_acc:.2f})'
    )

    ax.axhline(
        y=0.33,
        linestyle='--',
        linewidth=4,
        color='red',
        label='Chance Level (0.33)'
    )

    ax.legend(frameon=False, fontsize=16, loc='upper left')

    plt.tight_layout()
    plt.ylim([0, 1])

    output_dir = 'results/images'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'overt_covert_rest_accuracy.pdf'),
                format='pdf', bbox_inches='tight')

    plt.show()

def load_decoding_accuracies(results_dir):
    files = glob.glob(results_dir)
    records = []
    filename_re = re.compile(r"sub-([0-9a-zA-Z]+)_ses-([0-9a-zA-Z]+)_.*_accuracy\.csv")

    for filepath in files:
        fname = os.path.basename(filepath)
        m = filename_re.match(fname)
        if not m:
            continue

        subject_id, session_id = m.groups()

        try:
            df = pd.read_csv(filepath)
            if "accuracy" in df.columns:
                accuracy = df["accuracy"].iloc[0]
            else:
                accuracy = df.iloc[0, 0]

            records.append({
                "subject_id": subject_id,
                "session_id": session_id,
                "accuracy": float(accuracy)
            })
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    return records

def plot_accuracy(logger):
    logger.info('Plotting accuracy plot')

    accuracy = load_decoding_accuracies("results/DecodingResults/*_accuracy.csv")

    if not accuracy:
        logger.warning("No accuracy files found.")
        return

    accuracy_df = pd.DataFrame(accuracy)

    plot_accuracy_per_subject(accuracy_df)

    subj_acc = accuracy_df.groupby('subject_id')['accuracy'].mean().reset_index()
    logger.info(subj_acc)
    logger.info(subj_acc.describe())
