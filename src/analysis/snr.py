import os
import numpy as np
import pandas as pd
from src.data.data_reader import DataReader
from src.data.epochs import EpochsData
import mne
from pathlib import Path
from bids import BIDSLayout
from matplotlib import pyplot as plt
import pdb
import json
from src.utils import log_info



class SNR:
    def __init__(self, epochs, logger, config):
        self.epochs = epochs
        
        self.logger = logger
        self.config = config
        log_info(logger, "SNR initialized.") 
        self.baseline_window = config['analysis']['snr']['baseline_window']
        self.signal_window =  config['analysis']['snr']['signal_window']
    
    
    def compute_snr(self, picks=None):
        if picks is None:
            self.picks=picks
            evoked = self.epochs.average()
        else:
            evoked = self.epochs.copy().pick(picks).average()
            self.picks = self.epochs.ch_names
            
        times = evoked.times
        base_idx = np.where((times >= self.baseline_window[0]) & (times <= self.baseline_window[1]))[0]
        sig_idx  = np.where((times >= self.signal_window[0]) & (times <= self.signal_window[1]))[0]

        if len(base_idx) == 0 or len(sig_idx) == 0:
            raise ValueError("Windows are out of bounds for epoch times.")

        data = evoked.data.copy()
        data = data - data[:, base_idx].mean(axis=1, keepdims=True) # Baseline correct

        data = data * 1e6

        # Variances
        baseline_var = np.var(data[:, base_idx], axis=1)**2
        signal_var   = np.var(data[:, sig_idx], axis=1)**2

        # Overall (Spatial Average)
        snr_ratio = signal_var / baseline_var
        snr_db     = 10 * np.log10(snr_ratio)

        self.results = {
            'channel_names': evoked.ch_names,
            'snr_chan_db': snr_db.tolist(),
            'baseline_var': baseline_var.tolist(),
            'signal_var': signal_var.tolist(),
            'baseline_window': self.baseline_window,
            'signal_window': self.signal_window,
            
        }
        return self.results
        
def run_snr_subject_session(config, logger, subject, session):
    logger.info("Running P100 analysis for subject=%s, session=%s", subject, session)
    data_reader = DataReader(config, logger, subject, session)
    directory = 'derivatives/preprocessed_data'
    raw_eeg = data_reader.get_preprocessed_data(directory=directory)
    
    visual_epocher = EpochsData(
        raw= raw_eeg,
        logger = logger,
        config = config,
        tasks=['RealWordsExperiment', 'SilentWordsExperiment'],
        event='StartStimulus',
        modalities=['Pictures'],  # only visual trials
        tmin=config['analysis']['snr']['tmin'],
        tmax=config['analysis']['snr']['tmax'],
    )
    visual_epochs = visual_epocher.create_epochs() 
    
    snr = SNR(epochs=visual_epochs, logger=logger, config=config)
    results = snr.compute_snr(picks=config['analysis']['snr']['selectd_channels'])
    
    
    filename = f'sub-{subject}_ses-{session}_snr.json'
    output_dir = Path('results', 'snr')
    os.makedirs(output_dir, exist_ok=True)
    filepath = Path(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    
        
def run_snr(config, logger):
    logger.info("Starting SNR analysis pipeline.")
    layout = BIDSLayout(config['dataset']['BIDS_DIR'], validate=True)
    subject_ids = layout.get_subjects()
    for sub in subject_ids:
        if sub not in ['13', '14', '15']:
            continue
        
        session_ids = layout.get_sessions(subject=sub)
        for ses in session_ids:
            if sub == '13' and ses== '01':
                continue
            run_snr_subject_session(config, logger, sub, ses)
        
        
    logger.info("SNR analysis pipeline completed.")
    