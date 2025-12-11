import numpy as np
import mne
from scipy.signal import hilbert
import pdb
from src.utils.graphics import log_print

class MotorAnalysis:
    def __init__(self,logger, hf_band=None, baseline=None, analysis_window=None):
        self.hf_band = hf_band
        self.baseline = baseline
        self.analysis_window = analysis_window
        self.logger = logger
        log_print(text='Initializing MotorAnalysis', logger=self.logger)
        
        
    def bandpass_and_power(self, epochs):
        epochs = epochs.copy().filter(
            self.hf_band[0], self.hf_band[1], 
            fir_design='firwin'
        )
        
        sfreq = epochs.info['sfreq']  # sampling frequency
        end_sample = int((self.baseline[1] - epochs.tmin) * sfreq)

        # Slice the epochs to remove baseline
        epochs_data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
        epochs_data = epochs_data[:, :, end_sample:]
                
        power = epochs_data ** 2
        aveged_power = power.mean(axis=(2))
        
        return aveged_power
        
        
    