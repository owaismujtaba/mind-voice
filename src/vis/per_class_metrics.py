import seaborn as sns
import os
import glob
import numpy as np  
import re
import pandas as pd
from matplotlib import pyplot as plt

import pdb



def load_decoding_accuracies_per_class(results_dir):
    """
    Scan `results_dir` for files like:
      sub-13_ses-01_confusion_matrix.csv

    Calculates precision, recall, and f1 from the confusion matrix.
    """
    # Look for confusion matrix files instead
    pattern = os.path.join(results_dir, "*_confusion_matrix.csv")
    files = glob.glob(pattern)

    filename_re = re.compile(r"sub-(\d+)_ses-(\d+)_confusion_matrix\.csv")
    records = []

    for filepath in files:
        fname = os.path.basename(filepath)
        m = filename_re.match(fname)
        if not m:
            continue

        subject_id, session_id = m.groups()

        # Load CSV: index_col=0 ensures the first column (actual labels) is the index
        cm_df = pd.read_csv(filepath)
        
        # Convert to numpy for easier indexing
        # Rows = Actual, Columns = Predicted
        cm = cm_df.values
        classes = cm_df.index.tolist()

        # Calculate metrics per class
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp

        for i, class_label in enumerate(classes):
            # Handle potential division by zero
            prec = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0
            rec = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

            records.append({
                "subject_id": subject_id,
                "session_id": session_id,
                "class_id": class_label,
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1)
            })

    return pd.DataFrame(records)

def display_classwise(logger):
    # Ensure this points to the directory where confusion matrices are stored
    df = load_decoding_accuracies_per_class("results/decoding")

    if df.empty:
        logger.warning("No data found. Check directory path and filenames.")
        return

    # Average metrics across subjects and sessions for each class
    metrics = ['precision', 'recall', 'f1']
    
    for metric in metrics:
        # Group by subject and class first, then by class
        sub_avg = df.groupby(['subject_id', 'class_id'])[metric].mean().reset_index()
        total_avg = sub_avg.groupby('class_id')[metric].mean()
        logger.info(f'Class-wise mean {metric}:\n{total_avg}')

    # Standard Deviations
    for metric in metrics:
        # We calculate std across subjects
        sub_avg = df.groupby(['subject_id', 'class_id'])[metric].mean().reset_index()
        std_val = sub_avg.groupby('class_id')[metric].std()
        logger.info(f'Class-wise {metric} standard deviation:\n{std_val}')