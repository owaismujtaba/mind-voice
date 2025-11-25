import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

from bids import BIDSLayout
from src.pipelines.p100_pipeline import P100Pipeline
from src.utils.logger import create_logger
from src.utils.data import load_yaml



def plot_grand_erp_rest_visual(config, logger):
    """
    Plot grand average ERP for resting state data.

    Args:
        config (dict): Configuration dictionary.
        logger (Logger): Logger instance.
    """
    logger.info('Starting Grand ERP Rest Visualization')
    
    output_dir = 'results/images/'
    os.makedirs(output_dir, exist_ok=True)

    visual = {
        "label": "Visual Change",
        "trial_type": "Stimulus",
        "tmin": -0.2,
        "tmax": 0.5,
        "trial_mode": " ",
        "trial_unit": "Words",
        "experiment_mode": "Experiment",
        "trial_boundary": "Start",
        "modality": "Pictures"
    }
    

    rest = {
        "label": "No Visual Change",
        "trial_type": "Fixation",
        "tmin": -0.2,
        "tmax": 0.5,
        "trial_mode": " ",
        "trial_unit": "Words",
        "experiment_mode": "Experiment",
        "trial_boundary": "Start",
        "modality": "Pictures",
        "time_window": (0.08, 0.12)  # Optional window for P100
    }


    layout = BIDSLayout('BIDS', validate=True)
    subject_ids = layout.get_subjects()
    visual_data = []
    rest_data = []

    for sub in subject_ids:
        session_ids = layout.get_sessions(subject=sub)  
        for ses in session_ids:
            print(sub, ses)
            pipeline = P100Pipeline(
                        subject_id=sub,
                        session_id=ses,
                        cond_1=visual,
                        cond_2=rest,
                        channels = ['PO3', 'POz', 'PO4'], 
                        logger=logger,
                        config = config
            )
            key = f'sub-{sub}_ses-{ses}'
            pipeline.run()
            
            visual_epochs =  pipeline.cond_1_epochs.copy()
            visual_data.append(visual_epochs.get_data())
            rest_epochs =  pipeline.cond_2_epochs.copy()
            rest_data.append(rest_epochs.get_data())

    visual_data = np.concatenate(visual_data, axis=0)
    rest_data = np.concatenate(rest_data, axis=0)
    
    mean_visual = visual_data.mean(axis=0).mean(axis=0)
    mean_rest = rest_data.mean(axis=0).mean(axis=0)
    

    # Convert from volts to microvolts
    mean_visual_uv = mean_visual * 1e6
    mean_rest_uv = mean_rest * 1e6

    # Define time axis
    n_timepoints = len(mean_visual_uv)
    time = np.linspace(-0.2, 0.5, n_timepoints)

    plt.figure(figsize=(10, 5))

    # Plotting with stylish colors
    plt.plot(time, mean_visual_uv, label='Visual Change', color='#1f77b4', linewidth=2.5)
    plt.plot(time, mean_rest_uv, label='No Visual Change', color='#ff7f0e', linewidth=2.5, linestyle='--')

    # Vertical line at 0
    plt.axvline(0, color='k', linestyle='--', linewidth=2.0, label='Stimulus Onset (0 s)')

    # Highlight 80-120 ms
    plt.axvspan(0.08, 0.12, color='green', alpha=0.2, label='Window (80-120 ms)')

    # Labels and title
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Amplitude (µV)', fontsize=12, fontweight='bold')

    # Grid
    plt.grid(alpha=0.3, linestyle='--')

    # Customize ticks
    plt.xticks(np.arange(-0.2, 0.6, 0.1), fontsize=10, fontweight='bold', color='black')
    plt.yticks(fontsize=10, fontweight='bold', color='black')

    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    plt.legend(frameon=False, fontsize=12)

    plt.tight_layout()
    
    name = os.path.join(output_dir, 'grand_erp_rest_visual.png')
    plt.savefig(name, dpi=800)
    logger.info('Completed Grand ERP Rest Visualization')
    logger.info(f'Grand ERP plot saved at: {name}')
    
