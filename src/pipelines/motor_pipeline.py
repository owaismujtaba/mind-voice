from bids import BIDSLayout
import numpy as np
import pdb
from src.dataset.data_reader import BIDSDatasetReader
from src.dataset.eeg_epoch_builder import EEGEpochBuilder
from src.analysis.motor import MotorAnalysis
from pathlib import Path
import os


from src.utils.graphics import log_print

def motor_analysis_subject(config, logger, subject, session):
    logger.info(f'Running for sub-{subject} ses-{session}')
    ch_names = ['T7', 'T8', 'FT7', 'FT8']
    hf_band = (60, 120)   
    baseline = (-0.1, 0.0) 
    analysis_window = (0.0, 0.6) 
    
    reader = BIDSDatasetReader(
        logger=logger, config=config, 
        subject=subject, session=session
    )
    reader._load_raw()
    eeg = reader.raw_eeg
    
    
    overt = EEGEpochBuilder(
        eeg_data=eeg,
        trial_mode='Real', trial_unit='Words',
        trial_boundary='Start', 
        experiment_mode='Experiment', 
        trial_type='Speech',
        modality='',
        channels=ch_names, logger=logger
    )

    covert = EEGEpochBuilder(
        eeg_data=eeg,
        trial_mode='Silent', trial_unit='Words',
        trial_boundary='Start', 
        experiment_mode='Experiment', 
        trial_type='Speech',
        modality='',
        channels=ch_names, logger=logger
    )

    rest = EEGEpochBuilder(
        eeg_data=eeg,
        trial_mode='', trial_unit='Words',
        trial_boundary='Start', 
        experiment_mode='Experiment', 
        trial_type='Fixation',
        modality='',
        channels=ch_names, logger=logger
    )
    
    overt_epochs = overt.create_epochs(tmin=-0.1, tmax=1.5)
    covert_epochs = covert.create_epochs(tmin=-0.1, tmax=1.5)
    rest_epochs = rest.create_epochs(tmin=-0.1, tmax=1.5)
    
    motor_analyser = MotorAnalysis(
        logger=logger, hf_band=hf_band, 
        baseline=baseline, analysis_window=analysis_window
    )
    
    
    covert_power = motor_analyser.bandpass_and_power(covert_epochs)
    overt_power = motor_analyser.bandpass_and_power(overt_epochs)
    rest_power = motor_analyser.bandpass_and_power(rest_epochs)
    
    logger.info(f'Overt power : {overt_power.mean(axis=0)}')
    logger.info(f'Covert power : {covert_power.mean(axis=0)}')
    logger.info(f'Rest power : {rest_power.mean(axis=0)}')   
    
    return covert_power, overt_power, rest_power

        
        
        
def motor_analysis_pipeline(config, logger):
    log_print(text='Setting Motor Analysis Pipeline', logger=logger)
    dataset_config = config['dataset']
    layout = BIDSLayout(dataset_config['BIDS_DIR'], validate=True)
    subject_ids = layout.get_subjects()

    overt = []
    covert = []
    rest = []

    for sub in subject_ids:
        session_ids = layout.get_sessions(subject=sub)
        for ses in session_ids:
                overt_z, covert_z, rest_z = motor_analysis_subject(
                    config=config, logger=logger,
                    subject=sub, session=ses
                )
                
                overt.append(overt_z)
                covert.append(covert_z)
                rest.append(rest_z)
                
    overt = np.concatenate(overt, axis=0)
    covert = np.concatenate(covert, axis=0)
    rest = np.concatenate(rest, axis=0)
    directory = Path('results', 'motor')
    os.makedirs(directory, exist_ok=True)
    np.save(Path(directory, 'overt.npy'), overt)
    np.save(Path(directory, 'covert.npy'), covert)
    np.save(Path(directory, 'rest.npy'), rest)