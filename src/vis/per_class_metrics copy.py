import seaborn as sns
import os
import glob
import numpy as np  
import re
import pandas as pd
from matplotlib import pyplot as plt

def load_decoding_accuracies_per_class(results_dir):
    """
    Scan `results_dir` for files like:
      sub-13_ses-01_classification_report.csv

    Extract precision, recall, f1 for each class from each file.

    Parameters
    ----------
    results_dir : str
        Path to the folder containing *_classification_report.csv files.

    Returns
    -------
    pd.DataFrame
        Columns: ['subject_id', 'session_id', 'class_id', 'precision', 'recall', 'f1']
    """
    pattern = os.path.join(results_dir, "*_classification_report.csv")
    files = glob.glob(pattern)

    filename_re = re.compile(r"sub-(\d+)_ses-(\d+)_classification_report\.csv")
    records = []

    for filepath in files:
        fname = os.path.basename(filepath)
        m = filename_re.match(fname)
        if not m:
            continue

        subject_id, session_id = m.groups()

        # Load CSV assuming no header in first column (which holds class label)
        df = pd.read_csv(filepath)
        
        # Rename the first column (index column) to 'class'
        df.rename(columns={df.columns[0]: 'class'}, inplace=True)

        # Keep only rows where 'class' is a digit (i.e., individual classes)
        df = df[df['class'].apply(lambda x: str(x).isdigit())].copy()

        for _, row in df.iterrows():
            class_id = int(row['class'])
            records.append({
                "subject_id": subject_id,
                "session_id": session_id,
                "class_id": class_id,
                "precision": float(row['precision']),
                "recall": float(row['recall']),
                "f1": float(row['f1-score'])
            })

    return pd.DataFrame(records)

def display_classwise(logger):
    df = load_decoding_accuracies_per_class("results/decoding")

    precision = df.groupby(by=['subject_id', 'class_id'])['precision'].mean().reset_index()
    precision = precision.groupby(by='class_id')['precision'].mean()
    logger.info('Class-wise precision:\n%s', precision)

    recall = df.groupby(by=['subject_id', 'class_id'])['recall'].mean().reset_index()
    recall = recall.groupby(by='class_id')['recall'].mean()
    logger.info('Class-wise recall:\n%s', recall)

    f1 = df.groupby(by=['subject_id', 'class_id'])['f1'].mean().reset_index()
    f1 = f1.groupby(by='class_id')['f1'].mean()
    logger.info('Class-wise F1 score:\n%s', f1)

    precision = df.groupby(by=['subject_id','class_id'])['precision'].mean().reset_index()
    precision_std = precision.groupby(by='class_id')['precision'].std()
    logger.info('Class-wise precision standard deviation:\n%s', precision_std)
    
    recall = df.groupby(by=['subject_id', 'session_id','class_id'])['recall'].mean().reset_index()
    recall_std = recall.groupby(by='class_id')['recall'].std()
    logger.info('Class-wise recall standard deviation:\n%s', recall_std)
    
    f1 = df.groupby(by=['subject_id', 'session_id','class_id'])['f1'].mean().reset_index()
    f1_std = f1.groupby(by='class_id')['f1'].std()
    logger.info('Class-wise F1 score standard deviation:\n%s', f1_std)
