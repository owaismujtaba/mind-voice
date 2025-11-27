import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

import os

from src.utils.graphics import log_print

class EvokedSNR:
    """
    A class to compute and save SNR on Evoked (ERP) data.
    """

    def __init__(self, baseline_window=(-0.2, 0.0), signal_window=(0.08, 0.50), to_uV=True, logger=None):
        self.baseline_window = baseline_window
        self.signal_window = signal_window
        self.to_uV = to_uV
        self.logger = logger
        log_print(text='Initializing EvokedSNR', logger=self.logger)

    def compute(self, epochs, picks=None, baseline_window=None, signal_window=None):
        
        self.logger.info('Computig SNR')
        t_base = baseline_window if baseline_window is not None else self.baseline_window
        t_sig = signal_window if signal_window is not None else self.signal_window
        
        if picks is None:
            self.picks=picks
            evoked = epochs.average()
        else:
            evoked = epochs.copy().pick(picks).average()
            self.picks = epochs.ch_names

        times = evoked.times
        base_idx = np.where((times >= t_base[0]) & (times <= t_base[1]))[0]
        sig_idx  = np.where((times >= t_sig[0]) & (times <= t_sig[1]))[0]

        if len(base_idx) == 0 or len(sig_idx) == 0:
            raise ValueError("Windows are out of bounds for epoch times.")

        data = evoked.data.copy()
        data = data - data[:, base_idx].mean(axis=1, keepdims=True) # Baseline correct

        if self.to_uV:
            data = data * 1e6

        # Variances
        baseline_var = np.var(data[:, base_idx], axis=1)
        signal_var   = np.var(data[:, sig_idx], axis=1)

        # RMS
        baseline_rms = np.sqrt(baseline_var)
        signal_rms   = np.sqrt(signal_var)

        # SNR (Linear & dB)
        snr_lin = signal_var / baseline_var
        with np.errstate(divide='ignore', invalid='ignore'):
            snr_db = 10 * np.log10(snr_lin)

        # Overall (Spatial Average)
        overall_lin = signal_var.mean() / baseline_var.mean()
        overall_db     = 10 * np.log10(overall_lin)

        self.results = {
            'channel_names': evoked.ch_names,
            'snr_chan_db': snr_db,
            'snr_chan_linear': snr_lin,
            'snr_overall_db': overall_db,
            'snr_overall_linear': overall_lin,
            'baseline_var': baseline_var,
            'signal_var': signal_var,
            'baseline_rms': baseline_rms,
            'signal_rms': signal_rms,
            'settings': {
                'baseline_window': t_base, 
                'signal_window': t_sig,
                'units': 'uV' if self.to_uV else 'V'
            }
        }
        return self.results

    def save_csv(self, filename):
        results = self.results
        df = pd.DataFrame({
            'Channel': results['channel_names'],
            'SNR_dB': results['snr_chan_db'],
            'SNR_Linear': results['snr_chan_linear'],
            'Signal_RMS': results['signal_rms'],
            'Baseline_RMS': results['baseline_rms']
        })
        
        df.to_csv(filename, index=False)
        self.logger.info(f'Saved to: {filename}')
        self.logger.info(df)

   



