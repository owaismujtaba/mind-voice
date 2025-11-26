import os
import pandas as pd
import numpy as np
from pathlib import Path
import csv
from scipy.io.wavfile import write
from mnelab.io.xdf import read_raw_xdf

from mne_bids import BIDSPath, write_raw_bids
import mne
from pyxdf import resolve_streams, match_streaminfos



from src.utils.graphics import log_print

import pdb

class XDFDataReader:
    def __init__(self, filepath,logger, sub_id='01', ses_id='01', load_eeg=True, load_audio=False):
        self.xdf_filepath = filepath
        self.sub_id = sub_id
        self.ses_id = ses_id
        self.logger = logger
        log_print(logger, ' Initializing XDFDataReader Class')
        self.logger.info("Resolving streams from XDF file...")
        self.streams = resolve_streams(self.xdf_filepath)

        self.read_xdf_file(load_eeg, load_audio)

    def _load_stream(self, stream_type, label):
        self.logger.info(f"Loading {label} Stream...")
        try:
            stream_id = match_streaminfos(self.streams, [{'type': stream_type}])[0]
            if label=='Audio':
                self.audio = read_raw_xdf(self.xdf_filepath, stream_ids=[stream_id])
            else:
                self.eeg = read_raw_xdf(self.xdf_filepath, stream_ids=[stream_id])
            #setattr(self, stream_type.lower(), read_raw_xdf(self.xdf_filepath, stream_ids=[stream_id]))
            self.logger.info(f"{label} Stream Loaded Successfully!")
        except Exception as e:
            self.logger.info(f"Error loading {label} stream: {e}")

    def read_xdf_file(self, load_eeg=True, load_audio=False):
        self.logger.info("Reading XDF File...")
        if load_eeg:
            self._load_stream("EEG", "EEG")
        if load_audio:
            self._load_stream("Audio", "Audio")


class BIDSDataset:
    def __init__(self, xdf_reader, logger, config):
        self.logger = logger
        self.config = config
        log_print(logger, ' Initializing BIDSDataset Class')     
        
        self.xdf_reader = xdf_reader
        self.sub_id = xdf_reader.sub_id
        self.ses_id = xdf_reader.ses_id
        self.eeg = xdf_reader.eeg
        
        
        self._setup_paths()
        self.preprocess_eeg()
       

    def _setup_paths(self):
        self.logger.info('Setting paths')
        self.bids_root =  self.config['dataset']['BIDS_DIR']  

        self.bidspath = BIDSPath(
            subject= self.sub_id, session=self.ses_id,
            task='VCV', run='01', datatype='eeg',
            root=self.bids_root
        )
        self.eeg_sr = self.config['dataset']['EEG_SR']
        self.audio_sr = self.config['dataset']['AUDIO_SR']
        self.filename = f'sub-{self.sub_id}_ses-{self.ses_id}_VowelStudy_run-01'
        
    def preprocess_eeg(self):
        self.logger.info(f'Preprocessing eeg, resampling {self.eeg_sr}')
        self.eeg = self.eeg.resample(self.eeg_sr)
        
        
        
    def create_bids_files(self):
        self._create_bids_file_eeg()
        #self._create_bids_file_audio()

    def _create_bids_file_eeg(self):
        self.logger.info('Creating BIDS File for EEG')
        unique_annotations = set(self.eeg.annotations.description)
        event_id = {desc: i+1 for i, desc in enumerate(unique_annotations)}

        write_raw_bids(
            self.eeg, bids_path = self.bidspath,
            allow_preload=True, format="EDF", 
            overwrite=True, event_id=event_id
        )

    def _create_bids_file_audio(self):    
        self.logger.info('Creating BIDS File for Audio')
        audio = self.xdf_reader.audio.get_data()

        output_dir = Path(self.bids_root, f'sub-{self.sub_id}',
            f'ses-{self.ses_id}' , 'audio'
        )
        os.makedirs(output_dir, exist_ok=True)

        audio = audio.flatten()
        audio = audio / np.max(np.abs(audio))
        audio = (audio * 32765).astype(np.int16)
        
        filename = f'sub-{self.sub_id}_ses-{self.ses_id}_task-VCV_run-01'

        audio_filepath = Path(output_dir, f'{filename}_audio.wav')
        
        write(str(audio_filepath), self.audio_sr,audio)

        events_fileapth = Path(output_dir, f'{filename}_events.tsv')
        annotations = self.eeg.annotations

        with open(events_fileapth, "w", newline="") as tsvFile:
            writer = csv.writer(tsvFile,  delimiter='\t')
            writer.writerow(['onset', 'duration', 'description'])
            for onset, duration, description in zip(
                    annotations.onset, annotations.duration, 
                    annotations.description
                ):
                writer.writerow([onset, duration, description])

def create_bids_dataset(dataset_details, logger, config):
    logger.info('Inside create_bids_dataset')
    for details in dataset_details:
        filepath = details['path']
        sub_id = details['subject_id']
        ses_id = details['session_id']
        logger.info(f"subjetc:{sub_id} session:{ses_id} path:{filepath}")
        xdf_reader = XDFDataReader(
            filepath=filepath,
            logger=logger,
            sub_id=sub_id,
            ses_id=ses_id
        )
        bids = BIDSDataset(
            xdf_reader=xdf_reader, 
            logger=logger, config=config
        )
        bids.create_bids_files()
