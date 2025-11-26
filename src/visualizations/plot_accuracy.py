import seaborn as sns
import os
import glob
import re
import pandas as pd
from matplotlib import pyplot as plt

# Global plot settings for Publication Quality
plt.rcParams.update({
    "font.family": "sans-serif", # or 'serif' for specific journals
    "font.size": 14,
    "axes.labelsize": 20,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.dpi": 600,
    "axes.linewidth": 1.5,       # Thicker spine
    "xtick.major.width": 1.5,    # Thicker ticks
    "ytick.major.width": 1.5,
})

def plot_accuracy_per_subject(df):
    """
    Plots the mean validation accuracy per subject with publication-ready aesthetics.
    """
    # Compute mean accuracy per subject and sort
    subject_acc = df.groupby('subject_id')['accuracy'].mean().reset_index()
    subject_acc = subject_acc.sort_values('accuracy')

    plt.figure(figsize=(16, 8))

    # --- COLORING STRATEGY ---
    # OPTION 1 (Standard Scientific): Uniform color. 
    # Distinct colors imply distinct groups. Since these are all just subjects,
    # a single professional color is preferred.
    bar_color = "#4c72b0"  # Standard "Deep" Seaborn Blue
    
    # OPTION 2 (Ranked Gradient): If you specifically want to emphasize the ranking visually.
    # palette = sns.color_palette("Blues_d", len(subject_acc)) # Sequential Blue
    # palette = sns.color_palette("Greys_r", len(subject_acc)) # Sequential Grey
    
    # Create barplot
    ax = sns.barplot(
        x='subject_id',
        y='accuracy',
        data=subject_acc,
        color=bar_color,       # Use 'color' for uniform, 'palette' for gradient
        # palette=palette,     # Uncomment if using Option 2
        edgecolor='black',     # Essential for high-contrast publication prints
        linewidth=1.5
    )

    # Set font sizes, labels, and make them bold
    ax.set_xlabel('Subject ID', fontsize=20, fontweight='bold', labelpad=15)
    ax.set_ylabel('Validation Accuracy', fontsize=20, fontweight='bold', labelpad=15)

    # Set tick sizes and make ticks bold
    ax.tick_params(axis='x', labelsize=18, length=6)
    ax.tick_params(axis='y', labelsize=18, length=6)
    
    # Rotating x-labels often helps readability if there are many subjects
    # plt.xticks(rotation=45) 

    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # Remove top and right spines (Tufte style / Standard scientific)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add horizontal gridlines (Behind the bars)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, color='grey', alpha=0.7)
    ax.set_axisbelow(True)

    # Annotate bars with accuracy values
    for p in ax.patches:
        height = p.get_height()
        # Only annotate if the bar is tall enough, or adjust y-position dynamically
        ax.text(
            p.get_x() + p.get_width() / 2., 
            height + 0.005, 
            f'{height:.2f}', # .2f is usually sufficient for publication figures
            ha='center', 
            va='bottom',
            fontsize=14,
            fontweight='bold',
            color='black'
        )

    # Optional: Add Chance Level Line (Standard in decoding papers)
    # plt.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Chance Level')
    # plt.legend(frameon=False, fontsize=16)

    plt.tight_layout()
    
    # Save as PDF (Vector graphics are required for most submissions)
    output_dir = 'results/images'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'overt_covert_rest_accuracy.pdf'), format='pdf', bbox_inches='tight')
    plt.show()

def load_decoding_accuracies(results_dir):
    files = glob.glob(results_dir)
    records = []
    # Regex to pull out sub-XX and ses-XX
    filename_re = re.compile(r"sub-([0-9a-zA-Z]+)_ses-([0-9a-zA-Z]+)_.*_accuracy\.csv")

    for filepath in files:
        fname = os.path.basename(filepath)
        m = filename_re.match(fname)
        if not m:
            continue

        subject_id, session_id = m.groups()

        # Read the CSV
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
    # Ensure the path matches your actual folder structure
    accuracy = load_decoding_accuracies("results/DecodingResults/*_accuracy.csv")
    
    if not accuracy:
        logger.warning("No accuracy files found.")
        return

    accuracy_df = pd.DataFrame(accuracy)
    
    plot_accuracy_per_subject(accuracy_df)
    
    subj_acc = accuracy_df.groupby('subject_id')['accuracy'].mean().reset_index()
    logger.info(subj_acc.describe())