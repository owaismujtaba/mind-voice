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
        sfreq = epochs.info['sfreq']
        data = epochs.get_data()
        
        n_epochs, n_ch, n_times = data.shape
        data_reshaped = data.reshape(n_epochs * n_ch, n_times)
        filt = mne.filter.filter_data(
            data_reshaped, sfreq=sfreq, l_freq=self.hf_band[0], 
            h_freq=self.hf_band[1], verbose=False, method='fir'
        )  
        filt = filt.reshape(n_epochs, n_ch, n_times)
        
        analytic = hilbert(filt, axis=-1)
        power = np.abs(analytic) ** 2
        
        return power
        
        
    def baseline_zscore(self, power, times):
        bmin, bmax = self.baseline
        bidx = np.where((times >= bmin) & (times <= bmax))[0]
        if len(bidx) == 0:
            raise ValueError("Baseline window matches no time points.")
        baseline_mean = power[:, :, bidx].mean(axis=-1, keepdims=True)  # shape (n_epochs, n_ch, 1)
        baseline_std  = power[:, :, bidx].std(axis=-1, keepdims=True)   # shape (n_epochs, n_ch, 1)
        baseline_std[baseline_std == 0] = 1e-12
        z = (power - baseline_mean) / baseline_std
        return z