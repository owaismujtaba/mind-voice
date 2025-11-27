import os
import pdb
import json
import numpy as np
from pathlib import Path    

from bids import BIDSLayout

from src.analysis.snr import EvokedSNR
from src.dataset.eeg_epoch_builder import EEGEpochBuilder
from src.dataset.data_reader import BIDSDatasetReader

from src.utils.graphics import log_print


def run_snr_per_subject_session(subject_id, session_id, logger, config, picks):
    logger.info(f'Running for sub-{subject_id} ses-{session_id}')
    out_dir = config['analysis']['results_dir']
    out_dir = Path(out_dir, 'SNR')
    os.makedirs(out_dir, exist_ok=True)
    filename = f'sub-{subject_id}_ses-{session_id}'
    
    data_reader = BIDSDatasetReader(
       config=config, logger=logger,
       subject=subject_id, session=session_id
    )
    data_reader._load_raw()
    raw_eeg = data_reader.raw_eeg
    
    visual = EEGEpochBuilder(
       eeg_data=raw_eeg,
       trial_mode='', trial_unit='Words',
       trial_boundary='Start', 
       experiment_mode='Experiment', 
       trial_type='Speech',
       modality='Pictures',
       logger=logger
    )
    
    
    visual_epochs = visual.create_epochs(tmin=-0.2, tmax=0.5)
    
    
    snr = EvokedSNR(logger=logger)
    visual_snr =snr.compute(visual_epochs, picks=picks)
    snr.save_csv(
        filename=Path(out_dir, f'{filename}_visual.csv')
    )
   
    
   
    return visual_snr
   
   
    
   
def run_snr_pipeline(config, logger):
    log_print(text='Running SNR Pipeline', logger=logger)
    occipital = ['PO3', 'POz', 'PO4']
    dataset_config = config['dataset']
    layout = BIDSLayout(dataset_config['BIDS_DIR'], validate=True)
    subject_ids = layout.get_subjects()

    visual, rest = [], []
    for sub in subject_ids:
       
        subject_ids = layout.get_subjects()
        session_ids = layout.get_sessions(subject=sub)  
        for ses in session_ids:
            results = run_snr_per_subject_session(
                subject_id=sub, session_id=ses,
                logger=logger, config=config, picks=occipital
            )
    
    
    
    
    
    