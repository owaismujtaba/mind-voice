import numpy as np
from mne import Evoked
from mne.epochs import Epochs

from src.utils.graphics import log_print

class P100ComponentAnalyzer:
    def __init__(self, epochs: Epochs,logger, channels, time_window=(0.08, 0.12)):
        """
        Initialize the P100 analyzer.

        Args:
            epochs (Epochs): Preprocessed MNE Epochs object.
            time_window (tuple): Time window (in seconds) to look for P100 peak.
            channels (list or None): List of channel names to average. If None, uses default P100-relevant channels.
        """
        self.epochs = epochs
        self.time_window = time_window
        self.logger = logger
        log_print(text='Initializing P100ComponentAnalyzer', logger=logger)

        self.channels = channels

        valid_chs = [ch for ch in self.channels if ch in self.epochs.ch_names]
        if not valid_chs:
            raise ValueError("None of the selected channels are present in the Epochs object.")

        self.channels = valid_chs
        picked_epochs = self.epochs.copy().pick(self.channels)
        self.evoked = picked_epochs.average()

    def get_evoked(self) -> Evoked:
        """
        Returns:
            Evoked: The averaged ERP (evoked response).
        """
        return self.evoked

    def get_p100_peak(self, baseline_window=None):
        """
        Compute P100 peak latency and amplitude after averaging across selected channels.

        Args:
            baseline_window (tuple or None): Optional time window (in seconds) for baseline correction.

        Returns:
            tuple: (peak_latency in seconds, peak_amplitude in µV, mean_amplitude in µV)
        """
        self.logger.info('Caculating Latency, peak and mean')
        valid_chs = [ch for ch in self.channels if ch in self.evoked.ch_names]
        if not valid_chs:
            raise ValueError("None of the selected channels are present in the data.")

        ch_indices = [self.evoked.ch_names.index(ch) for ch in valid_chs]

        avg_data = np.mean(self.evoked.data[ch_indices, :], axis=0)

        if baseline_window:
            baseline_mask = (self.evoked.times >= baseline_window[0]) & (self.evoked.times <= baseline_window[1])
            if not np.any(baseline_mask):
                raise ValueError("No data points found in the baseline window.")
            baseline = np.mean(avg_data[baseline_mask])
            avg_data = avg_data - baseline

        time_mask = (self.evoked.times >= self.time_window[0]) & (self.evoked.times <= self.time_window[1])
        if not np.any(time_mask):
            raise ValueError("No data points found within the specified time window.")

        data_window = avg_data[time_mask]
        times_window = self.evoked.times[time_mask]

        peak_idx = np.argmax(data_window)
        self.latency = times_window[peak_idx]
        self.peak_amplitude = data_window[peak_idx] * 1e6  # Convert to µV
        self.mean_amplitude = np.mean(data_window) * 1e6   # Convert to µV

        self.logger.info(f'Latency: {self.latency}, Peak: {self.peak_amplitude}, Mean: {self.mean_amplitude}')
        return self.latency, self.peak_amplitude, self.mean_amplitude