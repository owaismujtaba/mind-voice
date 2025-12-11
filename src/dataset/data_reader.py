import mne
import os
import numpy as np 
from pathlib import Path
import json
from pyprep import NoisyChannels
from mne_bids import BIDSPath, read_raw_bids

from src.utils.graphics import log_print
from mne.preprocessing import ICA
import pdb

mne.set_config('MNE_USE_CUDA', 'true')
os.environ['MNE_CUDA_SEGMENT_SIZE'] = '16384'




class BIDSDatasetReader:
    def __init__(self, config, logger, subject, session):
        self.sub_id = subject
        self.ses_id = session
        self.logger = logger
        self.config = config

        log_print(text='Initializing BIDSDatasetReader',logger=self.logger)
        self._setup_paths()
    
    def _setup_paths(self):
        self.bidspath = BIDSPath(
            subject=self.sub_id, session=self.ses_id,
            task='VCV', run='01', datatype='eeg',
            root=self.config['dataset']['BIDS_DIR']
        )
        self.bids_dir = Path(self.config['dataset']['BIDS_DIR'])
        self.processed_dir = self.bids_dir / 'derivatives'
        self.filename = f'sub-{self.sub_id}_ses-{self.ses_id}_processed.fif'
        
        self.output_dir = self.config['analysis']['results_dir']
        self.bad_channels_file = Path(self.output_dir, 'BadChannels', f'sub-{self.sub_id}_ses-{self.ses_id}_bads.json')

    def _load_raw(self):
        self.logger.info("Loading raw EEG data")
        self.raw_eeg = read_raw_bids(self.bidspath, verbose=False)
        self.raw_eeg.load_data()
        self.logger.info('Data read sucessfully')
    
    def _set_reference(self):
        ref = self.config['preprocessing']['EEG_REFERENCE']
        self.logger.info(f'Setting EEG reference: {ref}')
        if hasattr(self, 'raw_eeg'):
            self.raw_eeg.set_eeg_reference(ref)

    def _interpolate_bad_channels(self):
        self.logger.info("Interpolating bad channels")
        prep = NoisyChannels(self.raw_eeg.copy())
        prep.find_bad_by_deviation()
        prep.find_bad_by_correlation()
        bads = prep.get_bads()
        if 'FCz' in bads:
            bads.remove('FCz')
        self.raw_eeg.info['bads'] = bads
        self._save_bad_channels(bads=bads)
        self.logger.info(f'Bad channels: {bads}')
        self.raw_eeg.interpolate_bads(reset_bads=True)
    
    def _set_channel_types_and_montage(self):
        self.logger.info("Setting channel types and montage")
        try:
            self.raw_eeg.set_channel_types({'EOG1': 'eog', 'EOG2': 'eog'})
        except Exception:
            self.raw_eeg.rename_channels({'TP9': 'EOG1', 'TP10': 'EOG2'})
            self.raw_eeg.set_channel_types({'EOG1': 'eog', 'EOG2': 'eog'})
        
        montage = mne.channels.make_standard_montage(self.config['dataset']['MONTAGE'])
        self.raw_eeg.set_montage(montage)
        
    def _apply_notch_filter(self):
        self.logger.info('Applying notch filter')
        self.processed_eeg.notch_filter(self.config['preprocessing']['NOTCH'], picks='eeg')
        self.logger.info('Notch filter completed')
    
    def _apply_bandpass_filter(self, l_freq, h_freq):
        self.logger.info(f'Applying bandpass filter: {l_freq}-{h_freq} Hz')
        self.processed_eeg.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
        self.logger.info('Bandpass filter completed')
    
    def load_processed(self, filepath):
        self.logger.info(f'Loading processed EEG from {filepath}')
        self.processed_eeg = mne.io.read_raw_fif(filepath, preload=True)
    
    def _apply_ica(self, ica_params):
        self.logger.info(f'Applying ICA with params: {ica_params}')
        ica = ICA(**ica_params)
        ica.fit(self.processed_eeg)
        eog_indices, _ = ica.find_bads_eog(self.processed_eeg, ch_name=['EOG1', 'EOG2'])
        ica.exclude = eog_indices
        self.processed_eeg = ica.apply(self.processed_eeg)
        self.logger.info('ICA completed')
    
    def _save_bad_channels(self, bads):
        self.bad_channels_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.bad_channels_file, 'w') as f:
            json.dump(bads, f)
        self.logger.info(f'Bad channels saved to {self.bad_channels_file}')

    def preprocess_eeg(self, bandpass=False, ica=False):
        folder_name = ''
        if bandpass:
            l_freq = self.config['preprocessing']['EEG_FILTER']['l_freq']
            h_freq = self.config['preprocessing']['EEG_FILTER']['h_freq']
            folder_name = f'low-{l_freq}_high-{h_freq}'
        if ica:
            folder_name += '_ICA-True'

        out_dir = self.processed_dir / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / self.filename

        if filepath.exists():
            self.logger.info('Processed EEG file exists. Loading...')
            self.load_processed(filepath)
            return
        self.logger.info('Preprocessing EEG')
        self._load_raw()
        self._set_channel_types_and_montage()
        self._set_reference()
        self._interpolate_bad_channels()

        self.processed_eeg = self.raw_eeg.copy()
        self._apply_notch_filter()
        if ica:
            self._apply_ica(self.config['preprocessing']['ICA_PARAMS'])
        if bandpass:
            self._apply_bandpass_filter(l_freq, h_freq)
        
        
        self._save_processed(filepath)
    
    
    
    def _save_processed(self, filepath):
        self.processed_eeg.save(filepath, overwrite=True)
        self.logger.info(f'Processed EEG saved at {filepath}')


