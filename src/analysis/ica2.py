from src.dataset.data_reader import BIDSDatasetReader
from mne_bids import BIDSPath, read_raw_bids
from src.utils.graphics import styled_print
from mne.preprocessing import ICA
import mne
from pyprep import NoisyChannels
from pathlib import Path
import os
import json
import numpy as np

class ICADataLoader:
    def __init__(self, subject_id, session_id, config, logger):
        self.subject_id = subject_id
        self.session_id = session_id
        self.config = config
        self.logger = logger

        self.logger.info(f"Initializing ICAAnalyzer for Subject: {subject_id}, Session: {session_id}")

        self.raw = None
        self.ica = None

        self.n_components = config['preprocessing']['ICA_PARAMS']['n_components']
        self.random_state = config['preprocessing']['ICA_PARAMS']['random_state']

        self._setup_dirs()
        self._setup_bidspath()
        self.load_data()
        if not self._check_if_already_processed():
            
            self._set_channel_types_and_montage()
            self._remove_bad_channels()
            self._apply_filter()
            self.fit_ica()

    def _setup_dirs(self):
        self.logger.info("Setting up directories for ICA")

        main_dir = Path(self.config['dataset']['BIDS_DIR'], 'derivaties')
        self.ica_dir = main_dir / 'ica'
        self.ica_dir.mkdir(parents=True, exist_ok=True)

        self.ica_data_dir = self.ica_dir / "data"
        self.ica_data_dir.mkdir(parents=True, exist_ok=True)

        self.ica_data_file = self.ica_data_dir / (
            f"sub-{self.subject_id}_ses-{self.session_id}_task-VCV_run-01_desc-ica_eeg.fif"
        )

        self.components_dir = self.ica_dir / "components"
        self.components_dir.mkdir(parents=True, exist_ok=True)

        self.components_file = self.components_dir / (
            f"sub-{self.subject_id}_ses-{self.session_id}_task-VCV_run-01_desc-ica_components.fif"
        )

        self.bads_dir = self.ica_dir / "bads"
        self.bads_dir.mkdir(parents=True, exist_ok=True)
        self.bads_file = self.bads_dir / (
            f"sub-{self.subject_id}_ses-{self.session_id}_bad_channels.json"
        )

        self.logger.info(f"ICA data will be saved to {self.ica_data_file}")
        self.logger.info(f"ICA components will be saved to {self.components_file}")
        self.logger.info(f"Bad channels info will be saved to {self.bads_file}")

    
    def _setup_bidspath(self):
        self.logger.info("Setting up BIDS Path")
        self.bidspath = BIDSPath(
            subject=self.subject_id,
            session=self.session_id,
            task='VCV',
            run='01',
            datatype='eeg',
            root=self.config['dataset']['BIDS_DIR']
        )

    def _check_if_already_processed(self):
        if self.ica_data_file.exists() and self.components_file.exists():
            self.logger.info("ICA has already been processed for this subject and session. Loading existing data.")
            self.ica_data_file = mne.io.read_raw_fif(self.ica_data_file, preload=True)
            self.cleaned_raw = self.ica_data_file
            return True
        return False


    def load_data(self):
        self.logger.info("Loading Raw Data")
        self.raw = read_raw_bids(self.bidspath, verbose=True)
        self.raw.load_data()


    def fit_ica(self):
        self.logger.info("Fitting ICA")

        self.ica = ICA(
            n_components=self.n_components,
            random_state=self.random_state
        )
        self.ica.fit(self.cleaned_raw)

        self.logger.info("ICA Fitting Complete")
        self.logger.info("Applying ICA to Remove Artifacts")

        self.cleaned_raw = self.ica.apply(self.cleaned_raw.copy())

        self.cleaned_raw.save(self.ica_data_file, overwrite=True)
        self.logger.info(f"Saved cleaned data to {self.ica_data_file}")

        self.save_ica_components()


    def _apply_filter(self):
        self.logger.info("Applying Filter")

        self.cleaned_raw.notch_filter(freqs=[50, 100], verbose=False)
        self.cleaned_raw.filter(
            **self.config['preprocessing']['EEG_FILTER'],
            fir_design='firwin',
            verbose=False
        )

        self.logger.info("Filtering Complete")


    def _set_channel_types_and_montage(self):
        self.logger.info("Setting Channels and Montage")

        try:
            self.raw.set_channel_types({'EOG1': 'eog', 'EOG2': 'eog'})
        except Exception:
            self.raw.rename_channels({'TP9': 'EOG1', 'TP10': 'EOG2'})
            self.raw.set_channel_types({'EOG1': 'eog', 'EOG2': 'eog'})

        montage = mne.channels.make_standard_montage(self.config['dataset']['MONTAGE'])
        self.raw.set_montage(montage)


    def _remove_bad_channels(self):
        self.logger.info("Finding Bad Channels")

        prep = NoisyChannels(self.raw.copy())
        prep.find_bad_by_deviation()
        prep.find_bad_by_correlation()

        self.raw.info['bads'] = prep.get_bads()
        self.bad_channels = prep.get_bads()

        self.logger.info(f"Identified {len(self.bad_channels)} Bad Channels: {self.bad_channels}")
        self.logger.info("Interpolating Bad Channels")

        self.cleaned_raw = self.raw.interpolate_bads(reset_bads=True)
        self._save_bads_info()


    def _save_bads_info(self):
        self.logger.info("Saving Bad Channels Information")
        with open(self.bads_file, 'w') as f:
            json.dump(self.bad_channels, f)
        self.logger.info(f"Saved bad channels info to {self.bads_file}")

   
    def save_ica_components(self):
        self.logger.info("Saving ICA Components")
        self.ica.save(self.components_file)
        self.logger.info(f"Saved ICA components to {self.components_file}")


