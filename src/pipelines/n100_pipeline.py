import os
import numpy as np
import pandas as pd
from mne.epochs import Epochs
from pathlib import Path

from bids import BIDSLayout

from src.dataset.data_reader import BIDSDatasetReader
from src.dataset.eeg_epoch_builder import EEGEpochBuilder
from src.analysis.p_100_analyser import P100ComponentAnalyzer
from src.visualizations.p100_plotter import P100Plotter

from src.utils.graphics import log_print

import pdb



class N100Pipeline:
    def __init__(self, subject_id, session_id, config, logger, cond_1, cond_2, channels):
        self.subject_id = subject_id
        self.session_id = session_id
        self.config = config
        self.logger = logger
        self.cond_1 = cond_1
        self.cond_2 = cond_2
        self.channels = channels
        
    def _get_epochs_condition(self, eeg, condition_cfg):
        self._log(f"Creating epochs for condition: {condition_cfg['label']}")

        epocher = EEGEpochBuilder(
            eeg_data=eeg,
            trial_mode=condition_cfg["trial_mode"],
            trial_unit=condition_cfg["trial_unit"],
            experiment_mode=condition_cfg["experiment_mode"],
            trial_boundary=condition_cfg["trial_boundary"],
            trial_type=condition_cfg["trial_type"],
            modality=condition_cfg["modality"],
            channels=self.channels,
            logger=self.logger,
            baseline=condition_cfg['baseline']
        )

        epochs = epocher.create_epochs(
            tmin=condition_cfg["tmin"],
            tmax=condition_cfg["tmax"]
        )
        return epochs
    
    def load_data(self):
        bids_reader = BIDSDatasetReader(
            subject=self.subject_id,
            session=self.session_id,
            logger=self.logger,
            config=self.config
        )
        bids_reader.preprocess_eeg(bandpass=True, ica=True)
        self.eeg = bids_reader.processed_eeg
    
    def extract_n100_evoked(self, evoked, time_window):
        #ev = evoked.copy().apply_baseline(baseline)
        ev = evoked.copy()
        data = ev.get_data().mean(axis=1)   # mean across channels
        times = ev.times
        i0, i1 = np.searchsorted(times, time_window)
        window_data = data[:, i0:i1]
        
        mean_amp = window_data.mean(axis=1) * 1e6
        peak_idx = window_data.argmin(axis=1)    # most negative point
        peak_amp = window_data[np.arange(window_data.shape[0]), peak_idx] * 1e6
        
        return [peak_amp, mean_amp]
      
    def run(self, plot=False):
        self.load_data()
        cond1_epochs = self._get_epochs_condition(self.eeg, self.cond_1)
        cond2_epochs = self._get_epochs_condition(self.eeg, self.cond_2)
        
        res_1 = self.extract_n100_evoked(cond1_epochs, self.cond_1['time_window'])
        res_2 = self.extract_n100_evoked(cond2_epochs, self.cond_2['time_window'])
        
        self._save_results(res_1=res_1, res_2=res_2)
        
    def _log(self, message):
        self.logger.info(message)

    def _save_results(self, res_1, res_2):
        self.logger.info('Saving results')
        
        results_dir = self.config['analysis']['results_dir']
        output_dir = Path(os.getcwd(), results_dir, 'N100')
        os.makedirs(output_dir, exist_ok=True)

        cond_1_label = self.cond_1["label"]
        cond_2_label = self.cond_2["label"]
        self.logger.info(f'Saving to {output_dir}')     
        np.save(
            Path(output_dir, f"sub-{self.subject_id}_ses-{self.session_id}_{cond_1_label}.npy"),
            np.array(res_1)
        )
        np.save(
            Path(output_dir, f"sub-{self.subject_id}_ses-{self.session_id}_{cond_2_label}.npy"), 
            np.array(res_2)
        )


def run_n100_pipeline(config, logger):
    dataset_config = config['dataset']
    logger.info('Setting up P100 Analysis Pipeline')
    layout = BIDSLayout(dataset_config['BIDS_DIR'], validate=True)
    
    
    audio = {
        "label": "Auditory",
        "trial_type": "",
        "tmin": -0.1,
        "tmax": 0.5,
        "trial_mode": "Words",
        "trial_unit": "Speech",
        "experiment_mode": "Experiment",
        "trial_boundary": "Start",
        "modality": "Audio",
        "time_window": (0.08, 0.12),
        "baseline": {"tmin": -0.1, "tmax": 0}
    }
    

    no_audio = {
        "label": "Non Auditory",
        "trial_type": "",
        "tmin": 0.3,
        "tmax": 0.9,
        "trial_mode": "Words",
        "trial_unit": "Fixation",
        "experiment_mode": "Experiment",
        "trial_boundary": "Start",
        "modality": "Audio",
        "time_window": (0.38, 0.42),
        "baseline": {"tmin": 0.3, "tmax": 0.4}
    }
    
    
    subject_ids = layout.get_subjects()
    for sub in subject_ids:
        session_ids = layout.get_sessions(subject=sub)
        for ses in session_ids:
                pipe = N100Pipeline(
                    subject_id=sub, session_id=ses,
                    config=config, logger=logger,
                    cond_1=audio, cond_2=no_audio,
                    channels=['Fz', 'Cz']
                )
                
                pipe.run(plot=True)
                