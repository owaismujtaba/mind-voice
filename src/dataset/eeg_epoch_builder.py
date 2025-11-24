import mne
import numpy as np


from src.dataset.data_reader import BIDSDatasetReader
from src.utils.graphics import styled_print, print_criteria, log_print

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
                 trial_type='', modality='', channels=None, logger=None):
        styled_print('', 'Initializing EEGEpochBuilder Class', 'red', panel=True)
        self.logger = logger
        log_print(text='Initializing Epoch Builder', logger=logger)
        self.eeg_data = eeg_data
        self.annotations = eeg_data.annotations
        self.criteria = [
            trial_mode, trial_unit, experiment_mode,
            trial_boundary, trial_type, modality
        ]
        self.channels = channels
        if channels:
            self.eeg_data.pick(self.channels)  # Pick specified channels
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
        styled_print('', 'Creating EPOCHS', color='green')
        self.logger.info('Creating Epochs')
        print_criteria(self.criteria + [tmin, tmax])
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
        epochs = mne.Epochs(
            self.eeg_data, events, event_id=event_id_map, 
            tmin=tmin, tmax=tmax, baseline=(tmin, tmin+0.2), 
            preload=True
        )
        self.epochs = epochs
        return epochs
