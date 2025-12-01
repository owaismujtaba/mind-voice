import seaborn as sns
import os
import glob
import numpy as np  
import re
import pandas as pd
from matplotlib import pyplot as plt

# Global plot settings for Publication Quality
plt.rcParams.update({
    "font.family": "sans-serif", # 'serif' is also common for latex papers
    "font.size": 14,
    "axes.labelsize": 20,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.dpi": 600,
    "axes.linewidth": 1.5,      # Thicker axis lines
    "xtick.major.width": 1.5,   
    "ytick.major.width": 1.5,
})

def load_decoding_metrics(results_dir):
    """
    Scan `results_dir` for classification reports and extract metrics.
    """
    pattern = os.path.join(results_dir, "*_classification_report.csv")
    files = glob.glob(pattern)
    records = []
    
    # Regex to extract subject and session IDs
    filename_re = re.compile(r"sub-([0-9a-zA-Z]+)_ses-([0-9a-zA-Z]+)_classification_report\.csv")

    for filepath in files:
        fname = os.path.basename(filepath)
        m = filename_re.match(fname)
        if not m:
            continue

        subject_id, session_id = m.groups()

        try:
            df = pd.read_csv(filepath)
            p, r, f = get_precision_recall_f1(df)

            records.append({
                "subject_id": subject_id,
                "session_id": session_id,
                "precision": p,
                "recall": r,
                "f1": f
            })
        except Exception as e:
            print(f"Skipping {fname}: {e}")

    return records

def plot_metrics_per_subject(records):
    """
    Plot average precision, recall, and F1 score per subject with publication aesthetics.
    """
    from collections import defaultdict

    if not records:
        print("No records found to plot.")
        return

    # Aggregate metrics by subject
    metrics_by_subject = defaultdict(lambda: {'precision': [], 'recall': [], 'f1': []})
    for rec in records:
        sid = rec['subject_id']
        metrics_by_subject[sid]['precision'].append(rec['precision'])
        metrics_by_subject[sid]['recall'].append(rec['recall'])
        metrics_by_subject[sid]['f1'].append(rec['f1'])

    # Sort subjects logically (numerical if possible, else alphabetical)
    try:
        subjects = sorted(metrics_by_subject.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
    except:
        subjects = sorted(metrics_by_subject.keys())

    # Compute averages
    precision_avg = [np.mean(metrics_by_subject[s]['precision']) for s in subjects]
    recall_avg = [np.mean(metrics_by_subject[s]['recall']) for s in subjects]
    f1_avg = [np.mean(metrics_by_subject[s]['f1']) for s in subjects]

    # Plot Setup
    x = np.arange(len(subjects))
    width = 0.25
    
    # --- PUBLICATION COLORING STRATEGY ---
    # Using the "Deep" palette: Muted Blue, Muted Green, Muted Red
    # This is high contrast but not jarring, and safe for colorblindness.
    colors = ["#9ec3dd",  # blue
          "#a39d98",  # orange
          "#b0dbb0"]  # green

    
    # Optional: Patterns for B&W print compatibility (Uncomment if needed)
    # hatches = ['', '///', '...'] 

    fig, ax = plt.subplots(figsize=(16, 8))

    # Create Bars
    bars1 = ax.bar(x - width, precision_avg, width, label='Precision', 
                   color=colors[0], edgecolor='black', linewidth=1.2) # hatch=hatches[0]
    
    bars2 = ax.bar(x, recall_avg, width, label='Recall', 
                   color=colors[1], edgecolor='black', linewidth=1.2) # hatch=hatches[1]
    
    bars3 = ax.bar(x + width, f1_avg, width, label='F1 Score', 
                   color=colors[2], edgecolor='black', linewidth=1.2) # hatch=hatches[2]

    # Axis Labels and Ticks
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, fontsize=16, fontweight='bold')
    
    ax.set_xlabel('Subject ID', fontsize=20, fontweight='bold', labelpad=15)
    ax.set_ylabel('Score', fontsize=20, fontweight='bold', labelpad=15)
    ax.set_ylim(0, 1.15) # Extended ylim to make room for legend/text

    # Formatting Spines (Tufte Style)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Grid (Behind bars)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, color='gray', alpha=0.5)
    ax.set_axisbelow(True)

    # Annotate bars vertically to save space
    def annotate_bars(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 0.015,
                    f"{height:.2f}",
                    ha='center',
                    va='bottom',
                    fontsize=14, # Slightly smaller for grouped bars
                    fontweight='bold',
                    rotation=90, # Vertical text prevents overlap in grouped plots
                    color="#000000"
                )

    annotate_bars(bars1)
    annotate_bars(bars2)
    annotate_bars(bars3)

    # Enhanced Legend
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.05), # Place legend outside/top
        ncol=3, 
        frameon=False, 
        fontsize=16
    )

    plt.tight_layout()
    
    # Ensure directory exists
    output_dir = 'results/images'
    os.makedirs(output_dir, exist_ok=True)
    plt.yticks(fontweight='bold')
    
    for label in ax.get_xticklabels():
        label.set_fontsize(16)
    
    for label in ax.get_yticklabels():
        label.set_fontsize(16)
    save_path = os.path.join(output_dir, 'overt_covert_rest_metrics.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    
    
    plt.show()

def get_precision_recall_f1(df):
    """
    Extracts metrics from classification report DataFrame.
    """
    try:
        # Standard Sklearn classification_report output often has 'weighted avg'
        # Adjust 'class' or index lookup based on exact CSV format
        if 'class' in df.columns:
            row = df[df['class'] == 'weighted avg'].iloc[0]
        elif 'Unnamed: 0' in df.columns: # Common pandas format
            row = df[df['Unnamed: 0'] == 'weighted avg'].iloc[0]
        else:
             # If index is the class name
            row = df.loc['weighted avg']

        p = float(row['precision'])
        r = float(row['recall'])
        f = float(row['f1-score'])
    except Exception:
        # Fallback: simple mean of numeric columns
        cols = ['precision', 'recall', 'f1-score']
        # Filter only numeric columns to avoid errors
        numeric_df = df[cols].apply(pd.to_numeric, errors='coerce')
        p = numeric_df['precision'].mean()
        r = numeric_df['recall'].mean()
        f = numeric_df['f1-score'].mean()
        
    return p, r, f

def plot_metrics(logger):
    logger.info('Plotting metrics')
    # Update path to match your actual structure
    metrics = load_decoding_metrics("results/DecodingResults")
    
    if not metrics:
        logger.warning("No metrics found to plot.")
        return

    metrics_df = pd.DataFrame(metrics)
    
    plot_metrics_per_subject(metrics)
    
    logger.info('Metrics plot saved to results/images/overt_covert_rest_metrics.pdf')
    
    # Calculate summary
    metrics_summary = metrics_df.groupby('subject_id')[['precision', 'recall', 'f1']].mean().reset_index()
    logger.info('Metrics summary per subject:\n%s', metrics_summary)