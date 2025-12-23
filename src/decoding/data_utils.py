
import pdb
import mne
import numpy as np
from mne.epochs import Epochs
from src.utils import log_info
import mne
import os
import numpy as np 
from pathlib import Path
import json
from pyprep import NoisyChannels
from mne_bids import BIDSPath, read_raw_bids

from mne.preprocessing import ICA
import pdb


class BIDSDatasetReader:
    def __init__(self, config, logger, subject, session):
        self.sub_id = subject
        self.ses_id = session
        self.logger = logger
        self.config = config

        log_info(text='Initializing BIDSDatasetReader',logger=self.logger)
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





class EEGEpochBuilder:
    """
    Builds EEG epochs from pre-loaded EEG data based on filtered event annotations.
    
    Attributes:
        eeg_data (mne.io.Raw): The loaded EEG data.
        annotations (list): Annotations extracted from the EEG data.
        criteria (list): List of filtering criteria to apply to event descriptions.
    """
    def __init__(self, eeg_data, trial_mode='', trial_unit='', 
                 experiment_mode='', trial_boundary='', 
                 trial_type='', modality='', channels=None, logger=None, baseline=None):
        self.logger = logger
        log_info(text='Initializing Epoch Builder', logger=logger)
        self.baseline=baseline
        self.eeg_data = eeg_data
        self.annotations = eeg_data.annotations
        self.criteria = [
            trial_mode, trial_unit, experiment_mode,
            trial_boundary, trial_type, modality
        ]
        self.channels = channels
        if channels:
            self.eeg_data.pick(self.channels)  
    def _filter_events(self):
        """
        Filters EEG event annotations based on predefined criteria.
        
        Returns:
            list: Filtered annotations that match all criteria.
        """
        filtered_events = []
        for event in self.annotations:
            if all(criterion in event['description'] for criterion in self.criteria):
                filtered_events.append(event)
        return filtered_events

    def create_epochs(self, tmin, tmax):
        """
        Creates epochs from EEG data using filtered events.
        
        Args:
            tmin (float): Start time before event in seconds.
            tmax (float): End time of event in seconds.
        
        Returns:
            mne.Epochs: The resulting epoched EEG data.
        
        Raises:
            ValueError: If no matching events are found.
        """
        self.logger.info('Creating Epochs')
        log_info(logger=self.logger, text=self.criteria + [tmin, tmax])
        filtered_events = self._filter_events()

        if not filtered_events:
            raise ValueError("No matching events found for epoching.")

        event_list = []
        event_id_map = {} 
        event_counter = 1

        for event in filtered_events:
            onset_sample = int(event['onset'] * self.eeg_data.info['sfreq'])  
            description = event['description']

            if description not in event_id_map:
                event_id_map[description] = event_counter
                event_counter += 1

            event_list.append([onset_sample, 0, event_id_map[description]])

        events = np.array(event_list)
        
        if self.baseline:
            self.logger.info(f'Baseline {self.baseline}')
            tmin = self.baseline['tmin']
            tmax = self.baseline['tmax']
            epochs = mne.Epochs(
                self.eeg_data, events, event_id=event_id_map, 
                tmin=tmin, tmax=tmax, baseline=[tmin, tmax],
                preload=True
            )
        else:
            epochs = mne.Epochs(
                self.eeg_data, events, event_id=event_id_map, 
                tmin=tmin, tmax=tmax,
                preload=True
            )
        self.epochs = epochs
        return epochs



class SpeechEEGDatasetLoader:
    def __init__(
        self,
        subject_id: str,
        session_id: str,
        label:int,
        condition_config: dict , 
        config,
        logger
    ) -> None:
        self.config=config
        self.logger = logger
        self.logger.info('Initializing SpeechEEGDatasetLoader')
        self.subject_id = subject_id
        self.session_id = session_id
        self.label = label
        self.condition_config = condition_config

    def load_data(self):
        self.bids_reader = BIDSDatasetReader(
            subject=self.subject_id,
            session=self.session_id,
            config=self.config,
            logger=self.logger
        )
        self.bids_reader.preprocess_eeg(bandpass=True, ica=True)
        self.eeg = self.bids_reader.processed_eeg
        
        return self
    
    def _create_epochs(self) -> Epochs:
        """
        Helper to create epochs using EEGEpochBuilder.

        Args:
            config (dict): Configuration dictionary for the condition.

        Returns:
            Epochs: MNE Epochs object
        """
        self.load_data()
        config = self.condition_config
        self.logger.info(config)
        
        return   EEGEpochBuilder(
            eeg_data=self.eeg,
            trial_mode=config["trial_mode"],
            trial_unit=config["trial_unit"],
            experiment_mode=config["experiment_mode"],
            trial_boundary=config["trial_boundary"],
            trial_type=config["trial_type"],
            modality=config["modality"],
            logger=self.logger,
            baseline=config['baseline']
        ).create_epochs(
            tmin=config["tmin"],
            tmax=config["tmax"]
        )
    
    def get_data(self):
        epochs = self._create_epochs()
        data = epochs.get_data()
        labels = [self.label for i in range(data.shape[0])]
        return data, labels