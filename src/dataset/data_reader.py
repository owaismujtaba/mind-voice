import mne
from pathlib import Path
from pyprep import NoisyChannels
from mnelab.io.xdf import read_raw_xdf
from mne_bids import BIDSPath, read_raw_bids
from pyxdf import resolve_streams, match_streaminfos

from src.utils.graphics import styled_print

import pdb

import concurrent.futures
import numpy as np # Needed for the conversion



class XDFDataReader:
    def __init__(self, filepath, sub_id='01', ses_id='01', load_eeg=True, load_audio=True):
        styled_print("🚀", "Initializing XDFDataReader Class", "yellow", panel=True)
        styled_print("👤", f"Subject: {sub_id} |  Session: {ses_id}", "cyan")

        self.xdf_filepath = filepath
        self.sub_id = sub_id
        self.ses_id = ses_id

        styled_print("📡", "Resolving streams from XDF file...", "magenta")
        self.streams = resolve_streams(self.xdf_filepath)

        self.read_xdf_file(load_eeg, load_audio)

    def _load_stream(self, stream_type, label):
        styled_print("", f"Loading {label} Stream...", "yellow")
        try:
            stream_id = match_streaminfos(self.streams, [{'type': stream_type}])[0]
            if label=='Audio':
                self.audio = read_raw_xdf(self.xdf_filepath, stream_ids=[stream_id])
            else:
                self.eeg = read_raw_xdf(self.xdf_filepath, stream_ids=[stream_id])
            #setattr(self, stream_type.lower(), read_raw_xdf(self.xdf_filepath, stream_ids=[stream_id]))
            styled_print("✅", f"{label} Stream Loaded Successfully!", "green")
        except Exception as e:
            styled_print("⚠️", f"Error loading {label} stream: {e}", "red", panel=True)

    def read_xdf_file(self, load_eeg=False, load_audio=True):
        styled_print("📂", "Reading XDF File...", "magenta")
        if load_eeg:
            self._load_stream("EEG", "EEG")
        if load_audio:
            self._load_stream("Audio", "Audio")


class BIDSDatasetReader:
    def __init__(self, sub_id, ses_id, config, logger):
        self.config = config
        self.logger = logger
        self.logger.info("Initializing BIDSDatasetReader Class")
        styled_print("🚀", "Initializing BIDSDatasetReader Class", "yellow", panel=True)
        self.sub_id = sub_id
        self.ses_id = ses_id
        
        self.raw = None
        
        self._setup_bidspath()
        self.processed_dir = Path(config['dataset']['BIDS_DIR']) / "derivatives" / "processed_eeg"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.processed_file = self.processed_dir / f"sub-{sub_id}_ses-{ses_id}_processed-raw.fif"
        
        self.read_or_process_data()
    
    def read_or_process_data(self):
        if self.processed_file.exists():
            self.logger.info(f"Loading Processed EEG Data: sub-{self.sub_id} ses-{self.ses_id}")
            styled_print("", f"Loading Processed EEG Data: sub-{self.sub_id} ses-{self.ses_id}", color='green')
            self.processed_file = mne.io.read_raw_fif(self.processed_file, preload=True, verbose=False)
        else:
            self.read_bids_subject_data()
            self.preprocess()
            self.save_processed_data()
    
    def preprocess(self):
        self.logger.info("Preprocessing EEG")
        styled_print('', 'Preprocessing EEG', color='red')
        self._set_channel_types_and_montage()
        self._remove_bad_channels()
        self._apply_filter()
        self._set_reference()
        self._remove_artifacts()
    
    def _set_channel_types_and_montage(self):
        self.logger.info("Setting  Channels and Montage")
        styled_print('', 'Setting Channels and Montage', color='cyan')
        try:
            self.raw.set_channel_types({'EOG1': 'eog', 'EOG2': 'eog'})
        except Exception:
            self.raw.rename_channels({'TP9': 'EOG1', 'TP10': 'EOG2'})
            self.raw.set_channel_types({'EOG1': 'eog', 'EOG2': 'eog'})
        montage = mne.channels.make_standard_montage(config.EEG_MONTAGE)
        self.raw.set_montage(montage)
    
    def _remove_bad_channels(self):
        self.logger.info("Interpolating Bad Channels")
        styled_print('', 'Interpolating Bad Channels', color='cyan')
        prep = NoisyChannels(self.raw.copy())
        prep.find_bad_by_deviation()
        prep.find_bad_by_correlation()
        self.raw.info['bads'] = prep.get_bads()
        self.raw.interpolate_bads(reset_bads=True)

    def _apply_filter(self):
        self.logger.info("Applying Filter")
        low = self.config['eeg_filter']['low']
        high = self.config['eeg_filter']['high']
        self.raw.filter(l_freq=low, h_freq=high, fir_design='firwin', verbose=False)
        #self.raw.filter(**config.EEG_FILTER, fir_design='firwin', verbose=False)

    def _set_reference(self):
        self.raw.set_eeg_reference(config.EEG_REFERENCE)

    def _remove_artifacts(self):
        self.logger.info("Removing Artifacts using ICA")
        styled_print('', 'Removing Artifacts using ICA', color='cyan')
        ica = mne.preprocessing.ICA(**config.ICA_PARAMS)
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
        styled_print('', 'Loading Raw Data', color='cyan')
        self.raw = read_raw_bids(self.bidspath, verbose=False)
        self.raw.load_data()
    
    def save_processed_data(self):
        self.logger.info("Saving Processed EEG Data")
        styled_print('', 'Saving Processed EEG Data', color='green')
        self.raw.save(self.processed_file, overwrite=True)
        self.processed_file=self.raw



