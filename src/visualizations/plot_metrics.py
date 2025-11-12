import seaborn as sns
import os
import glob
import numpy as np  
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



def load_decoding_metrics(results_dir):
    """
    Scan `results_dir` for files like:
      sub-13_ses-01_classification_report.csv
    Extract subject/session IDs and load precision, recall, f1 from each file.

    Parameters
    ----------
    results_dir : str
        Path to the folder containing your *_classification_report.csv files.

    Returns
    -------
    List[dict]
        A list of records, each with keys:
        - 'subject_id' : str
        - 'session_id' : str
        - 'precision'  : float
        - 'recall'     : float
        - 'f1'         : float
    """
    pattern = os.path.join(results_dir, "*_classification_report.csv")
    files = glob.glob(pattern)
    records = []

    # Regex to extract subject and session IDs
    # Example filename: sub-13_ses-01_classification_report.csv
    filename_re = re.compile(r"sub-(\d+)_ses-(\d+)_classification_report\.csv")

    for filepath in files:
        fname = os.path.basename(filepath)
        m = filename_re.match(fname)
        if not m:
            # Skip files that don't match the naming convention
            continue

        subject_id, session_id = m.groups()

        # Read the CSV, assuming the file is a classification report
        df = pd.read_csv(filepath)

        # You need to define this function to extract precision, recall, f1
        # or you can compute average metrics from the report, e.g., weighted average
        p, r, f = get_precision_recall_f1(df)

        records.append({
            "subject_id": subject_id,
            "session_id": session_id,
            "precision": p,
            "recall": r,
            "f1": f
        })

    return records



def plot_metrics_per_subject(records):
    """
    Plot average precision, recall, and F1 score per subject with enhanced aesthetics.

    Parameters
    ----------
    records : List[dict]
        Each dict must have keys:
        - 'subject_id' : str
        - 'precision'  : float
        - 'recall'     : float
        - 'f1'         : float
    """

    from collections import defaultdict

    # Aggregate metrics by subject
    metrics_by_subject = defaultdict(lambda: {'precision': [], 'recall': [], 'f1': []})
    for rec in records:
        sid = rec['subject_id']
        metrics_by_subject[sid]['precision'].append(rec['precision'])
        metrics_by_subject[sid]['recall'].append(rec['recall'])
        metrics_by_subject[sid]['f1'].append(rec['f1'])

    # Compute average metrics per subject
    subjects = sorted(metrics_by_subject.keys(), key=lambda x: int(x))
    precision_avg = [np.mean(metrics_by_subject[s]['precision']) for s in subjects]
    recall_avg = [np.mean(metrics_by_subject[s]['recall']) for s in subjects]
    f1_avg = [np.mean(metrics_by_subject[s]['f1']) for s in subjects]

    # Plot settings
    x = np.arange(len(subjects))
    width = 0.25
    fig, ax = plt.subplots(figsize=(16, 8))

    # Use visually distinct colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    bars1 = ax.bar(x - width, precision_avg, width, label='Precision', color=colors[0], edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x, recall_avg, width, label='Recall', color=colors[1], edgecolor='black', linewidth=1.2)
    bars3 = ax.bar(x + width, f1_avg, width, label='F1 Score', color=colors[2], edgecolor='black', linewidth=1.2)

    # Labels, bold fonts, and limits
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, fontsize=16, fontweight='bold')
    ax.set_xlabel('Subject ID', fontsize=20, fontweight='bold', labelpad=12)
    ax.set_ylabel('Score', fontsize=20, fontweight='bold', labelpad=12)
    ax.set_ylim(0, 1)

    # Annotate bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.01,
                f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize=14,
                fontweight='bold',
                rotation=90
            )

    # Legend
    ax.legend(fontsize=16)

    # Clean spines and grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_yticks(np.round(ax.get_yticks(), 2))
    ax.set_yticklabels([f"{tick:.2f}" for tick in ax.get_yticks()], fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/images/overt_covert_rest_metrics.png', dpi=800)
    plt.show()




def get_precision_recall_f1(df):
    """
    Assuming df is a classification report with rows for each class,
    and columns like 'precision', 'recall', 'f1-score', and a row 'weighted avg'.

    This function extracts the weighted average precision, recall, f1.

    Modify this depending on your CSV structure.
    """
    try:
        weighted = df[df['class'] == 'weighted avg'].iloc[0]
        p = float(weighted['precision'])
        r = float(weighted['recall'])
        f = float(weighted['f1-score'])
    except Exception:
        # fallback to mean of all classes except totals
        p = df['precision'].mean()
        r = df['recall'].mean()
        f = df['f1-score'].mean()
    return p, r, f


def plot_metrics(logger):
    logger.info('Plotting metrics')
    metrics = load_decoding_metrics("results/DecodingResults")
    metrics_df = pd.DataFrame(metrics)
    
    plot_metrics_per_subject(metrics)
    logger.info('Metrics plot saved to results/images/overt_covert_rest_metrics.png')
    logger.info('Metrics data:\n%s', metrics_df)

    metrics_summary = metrics_df.groupby('subject_id')[['precision', 'recall', 'f1']].mean().reset_index()
    logger.info('Metrics summary per subject:\n%s', metrics_summary)
















