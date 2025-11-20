import mne
import os
import numpy as np 
from pathlib import Path
from pyprep import NoisyChannels
from mnelab.io.xdf import read_raw_xdf
from mne_bids import BIDSPath, read_raw_bids
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


class BIDSDatasetReader:
    def __init__(self,config, logger, subject, session):
        self.sub_id = subject
        self.ses_id = session
        self.logger = logger
        self.config = config
        log_print(logger, ' Initalizing BIDSDatasetReader')
        self._setup_paths()
        
    def _setup_paths(self):
        self.bidspath = BIDSPath(
            subject=self.sub_id, session=self.ses_id,
            task='VCV', run='01', datatype='eeg',
            root=self.config['dataset']['BIDS_DIR']
        )
        
        self.bids = self.config['dataset']['BIDS_DIR']
        self.processed_dir = Path(self.bids, 'derivaties')
        self.filename = f'sub-{self.sub_id}_ses-{self.ses_id}_processed.fif'
    
    
    def load_raw_eeg(self):
        self.logger.info("Loading Raw Data")
        self.raw_eeg = read_raw_bids(self.bidspath, verbose=False)
        self.raw_eeg.load_data()
        
    def _set_reference(self):
        ref = self.config['preprocessing']['EEG_REFERENCE']
        self.logger.info(f'Setting refrence : {ref}')
        self.raw_eeg.set_eeg_reference(ref)
        self.logger.info("Setting EEG Reference")
        
    def preprocess_eeg(self,bandpass=False, ica=False):
        self.logger.info('Precprcessing EEG')
        folder_name =''
        if bandpass:
            l_freq = self.config['preprocessing']['EEG_FILTER']['l_freq']
            h_freq = self.config['preprocessing']['EEG_FILTER']['h_freq']
            folder_name = f'{folder_name}low-{l_freq}_high-{h_freq}'
        if ica:
            folder_name = f'{folder_name}_ICA-True'

        out_dir = Path(self.processed_dir, folder_name)
        os.makedirs(out_dir, exist_ok=True)
        filepath = Path(out_dir, self.filename)
        
        if os.path.exists(filepath):
            self.logger.info('Processed file present')
            self.load_processed(filepath)
            return 
        
        self.load_raw_eeg()
        self._set_reference()
        
        
        self.logger.info('Applying notch filter')
        self.processed_eeg = self.raw_eeg.copy()
        self.processed_eeg.notch_filter(self.config['preprocessing']['NOTCH'])
        self.logger.info('Notch filter completed')
        
        if bandpass:
            self.logger.info(f'Applying bandpass low: {l_freq}, high: {h_freq}')
            self.processed_eeg.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
            self.logger.info('Bandpass completed')       

        if ica:
            ica_parms = self.config['preprocessing']['ICA_PARAMS']
            self.apply_ica(ica_parms)       
            
        self._save_processed(filepath)
        
    def load_processed(self, filepath):
        self.processed_eeg = mne.io.read_raw_fif(filepath, preload=True)
    
    def apply_ica(self, ica_parms):
        self.logger.info(f"Removing Artifacts using ICA {ica_parms}")
        ica = mne.preprocessing.ICA(**ica_parms)
        ica.fit(self.processed_eeg)   
        eog_indices, _ = ica.find_bads_eog(self.processed_eeg, ch_name=['EOG1', 'EOG2'])
        ica.exclude = eog_indices
        self.processed_eeg = ica.apply(self.processed_eeg)
        self.logger.info('ICA Completed')   
        
    def _save_processed(self, filepath):
        self.processed_eeg.save(filepath, overwrite=True)
        self.logger.info(f'processed file saved to {filepath}')
        

class BIDSDatasetReader1:
    def __init__(self, sub_id, ses_id, config, logger):
        self.config = config
        self.logger = logger
        self.logger.info("Initializing BIDSDatasetReader Class")
        self.sub_id = sub_id
        self.ses_id = ses_id
        
        self.raw = None
        
        self._setup_bidspath()
        self.processed_dir = Path(os.getcwd(),  'processed')
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.processed_file = self.processed_dir / f"sub-{sub_id}_ses-{ses_id}_processed-raw.fif"
        
        self.read_or_process_data()
    
    def read_or_process_data(self):
        if self.processed_file.exists():
            self.logger.info(f"Loading Processed EEG Data: sub-{self.sub_id} ses-{self.ses_id}")
            self.processed_file = mne.io.read_raw_fif(self.processed_file, preload=True, verbose=False)
        else:
            self.read_bids_subject_data()
            self.preprocess()
            self.save_processed_data()
    
    def preprocess(self):
        self.logger.info("Preprocessing EEG")
        self._set_channel_types_and_montage()
        self._remove_bad_channels()
        self._apply_filter()
        self._set_reference()
        self._remove_artifacts()
    
    
    
    def _remove_bad_channels(self):
        self.logger.info("Interpolating Bad Channels")
        prep = NoisyChannels(self.raw.copy())
        prep.find_bad_by_deviation()
        prep.find_bad_by_correlation()
        self.raw.info['bads'] = prep.get_bads()
        self.raw.interpolate_bads(reset_bads=True)

    def _apply_filter(self):
        self.logger.info("Applying Filter")
        #self.raw.filter(l_freq=low, h_freq=high, fir_design='firwin', verbose=False)
        self.raw.filter(**self.config['preprocessing']['EEG_FILTER'], fir_design='firwin', verbose=False)

    def _set_reference(self):
        self.raw.set_eeg_reference(self.config['preprocessing']['EEG_REFERENCE'])
        self.logger.info("Setting EEG Reference")
        
    def _remove_artifacts(self):
        self.logger.info("Removing Artifacts using ICA")
        ica = mne.preprocessing.ICA(**self.config['preprocessing']['ICA_PARAMS'])
        ica.fit(self.raw)   
        eog_indices, _ = ica.find_bads_eog(self.raw, ch_name=['EOG1', 'EOG2'])
        ica.exclude = eog_indices
        self.raw = ica.apply(self.raw)
    
    def _setup_bidspath(self):
        self.bidspath = BIDSPath(
            subject=self.sub_id, session=self.ses_id,
            task='VCV', run='01', datatype='eeg',
            root=self.config['dataset']['BIDS_DIR']
        )   
    
    def read_bids_subject_data(self):
        self.logger.info("Loading Raw Data")
        self.raw = read_raw_bids(self.bidspath, verbose=False)
        self.raw.load_data()
    
    def save_processed_data(self):
        self.logger.info("Saving Processed EEG Data")
        self.raw.save(self.processed_file, overwrite=True)
        self.processed_file=self.raw



