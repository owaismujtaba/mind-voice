import numpy as np
from mne import Evoked
from mne.epochs import Epochs

from src.utils.graphics import log_print

import pdb


class P100ComponentAnalyzer:
    def __init__(self, epochs: Epochs,logger, channels):
        """
        Initialize the P100 analyzer.

        Args:
            epochs (Epochs): Preprocessed MNE Epochs object.
            time_window (tuple): Time window (in seconds) to look for P100 peak.
            channels (list or None): List of channel names to average. If None, uses default P100-relevant channels.
        """
        self.epochs = epochs
        self.logger = logger
        log_print(text='Initializing P100ComponentAnalyzer', logger=logger)

        self.channels = channels

        valid_chs = [ch for ch in self.channels if ch in self.epochs.ch_names]
        if not valid_chs:
            raise ValueError("None of the selected channels are present in the Epochs object.")

        self.channels = valid_chs
        picked_epochs = self.epochs.copy().pick(self.channels)
        self.evoked = picked_epochs

    def get_evoked(self) -> Evoked:
        """
        Returns:
            Evoked: The averaged ERP (evoked response).
        """
        return self.evoked

    def get_p100_peak(self, time_window=(0.08, 0.12)):
        """
        Compute P100 peak latency and amplitude after averaging across selected channels.

        Args:
            baseline_window (tuple or None): Optional time window (in seconds) for baseline correction.

        Returns:
            tuple: (peak_latency in seconds, peak_amplitude in µV, mean_amplitude in µV)
        """
        self.logger.info('Caculating  Peak and Mean')
        valid_chs = [ch for ch in self.channels if ch in self.evoked.ch_names]
        if not valid_chs:
            raise ValueError("None of the selected channels are present in the data.")

        
        
        ch_indices = [self.evoked.ch_names.index(ch) for ch in valid_chs]
        avg_data = np.mean(self.evoked.get_data()[:,ch_indices, :], axis=1)
        
       
        time_mask = (self.evoked.times >= time_window[0]) & (self.evoked.times <= time_window[1])
        if not np.any(time_mask):
            raise ValueError("No data points found within the specified time window.")
        

        data_window = avg_data[:, time_mask]
        times_window = self.evoked.times[time_mask]

        peak_idx = np.argmax(data_window, axis=1)

        self.peak_amplitude =data_window[np.arange(data_window.shape[0]), peak_idx] * 1e6
        self.mean_amplitude = np.mean(data_window, axis=1) * 1e6   # Convert to µV
        self.logger.info(f'Peak Shape: {self.peak_amplitude.shape}, Mean SHape: {self.mean_amplitude.shape}')
        return self.peak_amplitude, self.mean_amplitude