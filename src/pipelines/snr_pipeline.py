from src.analysis.ica import ICADataLoader
from src.dataset.eeg_epoch_builder import EEGEpochBuilder
import numpy as np
import pdb
import json
import numpy as np
from pathlib import Path    
import os

def compute_channelwise_variance(data):
    """Return mean channel variance for array (epochs, channels, time)."""
    channel_variance = np.var(data, axis=(0, 2))
    return channel_variance.mean()


def compute_ratio(data_no_ica, data_with_ica):
    """Return SNR-improvement ratio."""
    var_no_ica = compute_channelwise_variance(data_no_ica)
    var_with_ica = compute_channelwise_variance(data_with_ica)
    return (var_no_ica - var_with_ica) / var_no_ica



def run_snr_per_subject_session(
    subject_id, session_id, 
    logger, config
    ):
    
    
    ica_data_loader = ICADataLoader(
        subject_id=subject_id,
        session_id=session_id,
        config=config,
        logger=logger
    )
    
    
    
    epoch_builder_overt = EEGEpochBuilder(
        eeg_data=ica_data_loader.raw,
        trial_mode='Real',
        trial_unit='Words',
        experiment_mode='Experiment',
        trial_boundary='Start',
        trial_type='Speech',
        modality='',
    )
    
    raw_overt_epochs = epoch_builder_overt.create_epochs(
        tmin=-0.2,
        tmax=1.5,
    )
    epoch_builder_overt = EEGEpochBuilder(
        eeg_data=ica_data_loader.cleaned_raw,
        trial_mode='Real',
        trial_unit='Words',
        experiment_mode='Experiment',
        trial_boundary='Start',
        trial_type='Speech',
        modality='',
    )
    cleaned_overt_epochs = epoch_builder_overt.create_epochs(
        tmin=-0.2,
        tmax=1.5,
    )
    
    
    epoch_builder_covert = EEGEpochBuilder(
        eeg_data=ica_data_loader.raw,
        trial_mode='Silent',
        trial_unit='Words',
        experiment_mode='Experiment',
        trial_boundary='Start',
        trial_type='Speech',
        modality='',
    )
    
    raw_covert_epochs = epoch_builder_covert.create_epochs(
        tmin=-0.2,
        tmax=1.5,
    )
    epoch_builder_covert = EEGEpochBuilder(
        eeg_data=ica_data_loader.cleaned_raw,
        trial_mode='Silent',
        trial_unit='Words',
        experiment_mode='Experiment',
        trial_boundary='Start', 
        trial_type='Speech',
        modality='',
    )
    cleaned_covert_epochs = epoch_builder_covert.create_epochs(
        tmin=-0.2,
        tmax=1.5,
    )
    
    
    
    ratio_covert = compute_ratio(
        raw_covert_epochs.get_data(),
        cleaned_covert_epochs.get_data()
    )
    ratio_overt = compute_ratio(
        raw_overt_epochs.get_data(),
        cleaned_overt_epochs.get_data()
    )
    
    print(f"Subject: {subject_id} | Session: {session_id} | Overt SNR Improvement Ratio: {ratio_overt:.4f} | Covert SNR Improvement Ratio: {ratio_covert:.4f}")
    
    logger.info(f"Subject: {subject_id} | Session: {session_id} | Overt SNR Improvement Ratio: {ratio_overt:.4f} | Covert SNR Improvement Ratio: {ratio_covert:.4f}")
    
    results = {
        "subject_id": subject_id,
        "session_id": session_id,
        "overt_snr_improvement_ratio": ratio_overt,
        "covert_snr_improvement_ratio": ratio_covert
    }
    
    directory = config['analysis']['results_dir']
    directory_path = Path(directory, 'SNRResults')
    os.makedirs(directory_path, exist_ok=True)
    with open(directory_path / f"Sub-{subject_id}_Ses-{session_id}.json", "w") as f:
        json.dump(results, f)
    
    
    
    