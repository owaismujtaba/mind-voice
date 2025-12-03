import pdb
import os
from bids import BIDSLayout
from pathlib import Path

from src.utils.data import load_yaml
from src.dataset.bids import create_bids_dataset

from src.pipelines.p100_pipeline import P100Pipeline
from src.pipelines.overt_covert_rest_pipeline import OvertCovertRestPipeline
from src.anonymization.voice_snonymizer import VoiceAnonymizerPipeline
from src.pipelines.snr_pipeline import run_snr_pipeline
from src.pipelines.motor_pipeline import motor_analysis_pipeline
from src.pipelines.n100_pipeline import run_n100_pipeline


from src.visualizations.peak_mean_visual_rest import plot_peak_mean_visual_novisual
from src.visualizations.erp_grand_visual_rest import plot_grand_erp_rest_visual
from src.visualizations.plot_accuracy import plot_accuracy
from src.visualizations.plot_confusion_matrix import plot_confusion_matrix
from src.visualizations.plot_metrics import plot_metrics
from src.visualizations.display_per_class_metrics import display_classwise
from src.visualizations.plot_snr import plot_snr_visual
from src.visualizations.erp_grand_audio_no_audio import plot_grand_erp_audio_no_audio
from src.visualizations.peak_mean_audio_no_audio import plot_peak_mean_audio_no_audio

from src.utils.logger import create_logger


config = 'config.yaml'
config = load_yaml(config)

dataset_config = config['dataset']
if dataset_config['create_bids']:
    logger = create_logger('bids')
    logger.info('Creating BIDS Dataset')
    file_info = dataset_config['config_path']
    dataset_info = load_yaml(file_info)
    dataset_details = dataset_info['filepaths']
    create_bids_dataset(dataset_details, logger, config)


analysis_config = config['analysis']
if analysis_config['p100']:
    logger = create_logger('p100')
    visual = {
        "label": "Visual",
        "trial_type": "Stimulus",
        "tmin": -0.1,
        "tmax": 0.5,
        "trial_mode": "",
        "trial_unit": "Words",
        "experiment_mode": "Experiment",
        "trial_boundary": "Start",
        "modality": "Pictures",
        "baseline": {"tmin": -0.1, "tmax": 0}
    }
    

    rest = {
        "label": "No Visual Change",
        "trial_type": "Fixation",
        "tmin": 0.3,
        "tmax": 0.9,
        "trial_mode": "",
        "trial_unit": "Words",
        "experiment_mode": "Experiment",
        "trial_boundary": "Start",
        "modality": "Pictures",
        "time_window": (0.08, 0.12),
        "baseline": {"tmin": 0.3, "tmax": 0.4}
    }
    logger.info('Setting up P100 Analysis Pipeline')
    layout = BIDSLayout(dataset_config['BIDS_DIR'], validate=True)
    subject_ids = layout.get_subjects()

    

    for sub in subject_ids:
        session_ids = layout.get_sessions(subject=sub)
        for ses in session_ids:
                pipe = P100Pipeline(
                    subject_id=sub, session_id=ses,
                    config=config, logger=logger,
                    cond_1=visual, cond_2=rest,
                    channels=['PO3', 'POz', 'PO4']
                )
                
                pipe.run()

if analysis_config['n100']:
    logger = create_logger('n100')
    run_n100_pipeline(config=config, logger=logger)       

if config['anonymize']['anonymize_audio']:
    anonymize_config = config['anonymize']
    logger = create_logger('anonymize')
    layout = BIDSLayout(dataset_config['BIDS_DIR'], validate=True)
    subject_ids = layout.get_subjects()

    for sub in subject_ids:
        session_ids = layout.get_sessions(subject=sub)  
        for ses in session_ids:
            directory = Path(dataset_config['BIDS_DIR'], f'sub-{sub}', f'ses-{ses}', 'audio')
            filepath = [os.path.join(directory, file) for file in os.listdir(directory) if file.lower().endswith(".wav")][0]
            logger.info(filepath)
            pipeline = VoiceAnonymizerPipeline(
                pitch_steps=anonymize_config['pitch_steps'], 
                formant_ratio=anonymize_config['formant_ratio'],
                logger=logger
            )
            anonymized_audio = pipeline.fit_transform(filepath)
            pipeline.save(anonymized_audio, pipeline.target_sr, Path(directory, "anonymized.wav"))


if analysis_config['morter']:
    logger = create_logger('motor')
    motor_analysis_pipeline(config, logger)

plot_config = config['plotting']
logger = create_logger('plotting')

if plot_config['grand_erp_visual_real']:
    plot_grand_erp_rest_visual(config, logger)

if plot_config['peak_mean_amplitude']:
    plot_peak_mean_visual_novisual(logger)


    
if plot_config['accuracy_plots']:
    plot_accuracy(logger)
  
if plot_config['confusion_matrix']:
    plot_confusion_matrix(logger)


if plot_config['metrics_plots']:
    plot_metrics(logger)

if plot_config['peak_mean_amplitude_audio']:
    plot_peak_mean_audio_no_audio(logger)

if plot_config['grand_erp_audio_no_audio']:
    plot_grand_erp_audio_no_audio(config, logger)

if plot_config['display_per_class_metrics']:
    display_classwise(logger)

if plot_config['plot_snr']:
    plot_snr_visual(config=config, logger=logger)


if analysis_config['snr']:
    logger = create_logger('snr')
    run_snr_pipeline(config=config, logger=logger)


if analysis_config['decoding']:
    logger = create_logger('classification')
    layout = BIDSLayout(dataset_config['BIDS_DIR'], validate=True)
    subject_ids = layout.get_subjects()

    for sub in subject_ids:
        session_ids = layout.get_sessions(subject=sub)  
        for ses in session_ids:
            pipeline = OvertCovertRestPipeline(
                    subject_id=sub, session_id=ses,
                    config=config, logger=logger
                )
            pipeline.run()
    


    
