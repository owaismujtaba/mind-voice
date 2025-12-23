import os
import numpy as np
import pandas as pd
from src.data.data_reader import DataReader
from src.data.epochs import EpochsData
import mne
from pathlib import Path
from bids import BIDSLayout
from matplotlib import pyplot as plt
import pdb

from src.utils import log_info

class N100Analyzer:
    def __init__(self, audio_epochs, no_audio_epochs, logger, config):
        self.audio_epochs = audio_epochs
        self.no_audio_epochs = no_audio_epochs
        self.logger = logger
        self.config = config
        log_info(logger, "N100Analyzer initialized.")
        
        
    def compute_n100(self):
        self.logger.info("Computing N100 component.")        
        self.audio_evoked = self.audio_epochs.average()
        self.no_audio_evoked = self.no_audio_epochs.average()
        self.logger.info("N100 component computed.")
        
     

    def plot_n100(self, subject, session):
        self.logger.info("Plotting P100 component for subject=%s, session=%s", subject, session)
        
        channels = self.config['analysis']['n100']['selectd_channels']
        tmin = self.config['analysis']['n100']['tmin']
        tmax = self.config['analysis']['n100']['tmax']

        ch_indices = [self.visual_evoked.ch_names.index(ch) for ch in channels]

        visual_avg = self.visual_evoked.data[ch_indices, :].mean(axis=0)
        no_visual_avg = self.no_visual_evoked.data[ch_indices, :].mean(axis=0)
        
        

        # Compute time vector
        n_samples = visual_avg.shape[0]
        times = np.linspace(tmin, tmax, n_samples)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, visual_avg, label='Visual', linestyle='-')
        ax.plot(times, no_visual_avg, label='No Visual', linestyle='--')

        # Vertical line at stimulus onset (x=0)
        ax.axvline(x=0, color='black', linewidth=1, linestyle='--', label='Stimulus Onset')
        ax.axhline(y=0, color='black', linewidth=1, linestyle='-',)

        ax.axvspan(0.08, 0.12, color='yellow', alpha=0.3, label='P100 window')

        # Labels
        ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Amplitude (µV)', fontsize=14, fontweight='bold')

        # Tick styling
        ax.tick_params(axis='x', labelsize=12, width=1, length=6)
        ax.tick_params(axis='y', labelsize=12, width=1, length=6)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
            
        ax.text(0.05, ax.get_ylim()[1]*0.9, channels, color='black', fontsize=12, fontweight='bold')
        xticks = np.arange(np.ceil(tmin*10)/10, np.floor(tmax*10)/10 + 0.01, 0.1)
        ax.set_xticks(xticks)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(fontsize=12)
        
        results_dir = Path(self.config['analysis']['results_dir'], 'p100', 'plots')
        os.makedirs(results_dir, exist_ok=True)
        plot_path = Path(results_dir, f'sub-{subject}_ses-{session}_p100.png')
        fig.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        self.logger.info("P100 plot saved to %s", plot_path)


    def save(self, subject=None, session=None):
        self.logger.info("Saving N100 results. sub-%s ses-%s", subject, session)
        

        results_dir = Path(self.config['analysis']['results_dir'], 'n100', 'data')
        results_dir.mkdir(parents=True, exist_ok=True)

        audio_path = results_dir / f'audio-evoked-sub-{subject}-ses-{session}.fif'
        no_audio_path = results_dir / f'no-audio-evoked-sub-{subject}-ses-{session}.fif'

        self.audio_epochs.save(audio_path, overwrite=True)
        self.no_audio_epochs.save(no_audio_path, overwrite=True)
        
        self.logger.info("N100 results saved to %s.", results_dir)

    


def compute_erp_condition_diffs(evokeds, rois, windows,  mode='pos'):
    # Ensure we have exactly two conditions
    cond_names = list(evokeds.keys())
    if len(cond_names) != 2:
        raise ValueError("`evokeds` must contain exactly two conditions.")
    cond1, cond2 = cond_names

    # Normalize windows to list
    if isinstance(windows, tuple):
        windows = [windows]

    rows = []
    
    for roi_name, roi_chs in rois.items():
        for (tmin, tmax) in windows:
            # For each condition, compute peak and mean
            stats = {}
            for cond in (cond1, cond2):
                evoked = evokeds[cond]

                # Pick ROI channels
                picks = mne.pick_channels(evoked.ch_names, roi_chs)
                if len(picks) == 0:
                    raise ValueError(f"No ROI channels found in evoked for ROI '{roi_name}'.")

                times = evoked.times
                win_mask = (times >= tmin) & (times <= tmax)
                if not np.any(win_mask):
                    raise ValueError(f"No time points in window [{tmin}, {tmax}] s")

                roi_data = evoked.get_data()[picks][:, win_mask].mean(axis=0)  # average across channels
                roi_times = times[win_mask]

                # Peak
                if mode == 'pos' or mode == 'positive':
                    idx_peak = np.argmax(roi_data)
                elif mode == 'neg' or mode == 'negative':
                    idx_peak = np.argmin(roi_data)
                else:
                    raise ValueError("mode must be 'pos' or 'neg'")

                peak_amp = roi_data[idx_peak]       # Volts
                peak_lat = roi_times[idx_peak]      # seconds
                mean_amp = roi_data.mean()          # Volts

                stats[cond] = dict(
                    peak_amp_uv=peak_amp * 1e6,
                    peak_lat_ms=peak_lat * 1e3,
                    mean_amp_uv=mean_amp * 1e6,
                )

            # Compute differences: cond1 - cond2
            row = dict(
                roi=roi_name,
                tmin=tmin,
                tmax=tmax,
                # cond1=cond1,
                # cond2=cond2,
                # peak_amp_uv_cond1=stats[cond1]['peak_amp_uv'],
                # peak_amp_uv_cond2=stats[cond2]['peak_amp_uv'],
                peak_diff_uv=stats[cond1]['peak_amp_uv'] - stats[cond2]['peak_amp_uv'],
                # mean_amp_uv_cond1=stats[cond1]['mean_amp_uv'],
                # mean_amp_uv_cond2=stats[cond2]['mean_amp_uv'],
                mean_diff_uv=stats[cond1]['mean_amp_uv'] - stats[cond2]['mean_amp_uv'],
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


        
def run_best_condition_analysis(config, logger, visual_epochs, no_visual_epochs, subject, session):
    logger.info("Running Best Condition analysis for subject=%s, session=%s", subject, session)
     
    evokeds = {
        'Visual': visual_epochs.average(),
        'No Visual': no_visual_epochs.average(),
    }
    
    
    rois = config['analysis']['p100'].get('rois', {
            'Posterior': ['PO3', 'POz', 'PO4'],
    })
    
    windows_p100 = [
        # --- Main / literature-based ---
        (0.08, 0.14),
        (0.09, 0.13),
        (0.095, 0.12),

        # --- Robustness / sensitivity analysis ---
        (0.08, 0.12),
        (0.09, 0.13),
        (0.10, 0.14),
    ]
    
    
    df_p100 = compute_erp_condition_diffs(
        evokeds=evokeds,
        rois=rois,
        windows=windows_p100,
        mode='pos'
    )
    df_p100 = get_best_condition(df=df_p100, n_top=df_p100.shape[0])
    results_dir = Path(config['analysis']['results_dir'], 'p100')
    outpath = Path(results_dir, 'best_conf')
    os.makedirs(outpath, exist_ok=True)
    csv_path = Path(outpath, f'sub-{subject}_ses-{session}_p100_best_condition_analysis.csv')
    df_p100.to_csv(csv_path, index=False)
    logger.info("Best Condition analysis results saved to %s", csv_path)
    
    return df_p100
    
def get_best_condition(
        df, n_top=3, combine='both',
        effect_direction='positive',
        peak_col='peak_diff_uv',
        mean_col='mean_diff_uv'
    ):
    df = df.copy()

    df['peak_score'] = df[peak_col]
    df['mean_score'] = df[mean_col]

    # Build a single ranking score
    if combine == 'peak':
        df['score'] = df['peak_score']
    elif combine == 'mean':
        df['score'] = df['mean_score']
    elif combine == 'both':
        # Z-score each metric so they contribute comparably
        for col in ['peak_score', 'mean_score']:
            vals = df[col].values
            if np.allclose(vals.std(), 0):
                df[col + '_z'] = 0.0
            else:
                df[col + '_z'] = (vals - vals.mean()) / vals.std()
        df['score'] = df['peak_score_z'] + df['mean_score_z']
    else:
        raise ValueError("combine must be one of {'both', 'peak', 'mean'}")

    if effect_direction == 'pos' or effect_direction == 'positive':
        df_sorted = df.sort_values('score', ascending=False).head(n_top)
    else:
        df_sorted = df.sort_values('score', ascending=True).head(n_top) 
        
    return df_sorted
 
    
def run_n100_per_subject_session(config, logger, subject, session):
    logger.info("Running P100 analysis for subject=%s, session=%s", subject, session)
    
    # Initialize DataReader
    data_reader = DataReader(config, logger, subject, session)
    raw_eeg = data_reader.get_preprocessed_data()
    
    # Create EpochsData instance
    audio_epocher = EpochsData(
        raw= raw_eeg,
        logger = logger,
        config = config,
        tasks=['RealWordsExperiment', 'SilentWordsExperiment', 'RealSyllablesExperiment', 'SilentSyllablesExperiment'],
        event='StartStimulus',
        modalities=['Audio'],  # only visual trials
        tmin=config['analysis']['n100']['tmin'],
        tmax=config['analysis']['n100']['tmax'],
        baseline=(config['analysis']['n100']['tmin'], 0)
    )
    audio_epochs = audio_epocher.create_epochs()  
    
    no_audio_epocher = EpochsData(
        raw = raw_eeg,
        logger = logger,
        config = config,
        event_offset=config['analysis']['n100']['no_audio_offset'],
        tasks=['RealWordsExperiment', 'SilentWordsExperiment', 'RealSyllablesExperiment', 'SilentSyllablesExperiment'],
        event='StartFixation',
        modalities=['Audio'], 
        tmin=config['analysis']['n100']['tmin'],
        tmax=config['analysis']['n100']['tmax'],
        baseline=(config['analysis']['n100']['tmin'], 0)
    )
    no_audio_epochs = no_audio_epocher.create_epochs()  
    p100_analyzer = N100Analyzer(
        audio_epochs=audio_epochs,
        no_audio_epochs=no_audio_epochs,
        logger=logger,
        config=config
    )
    p100_analyzer.compute_n100()  
    #p100_analyzer.plot_p100(subject, session)
    p100_analyzer.save(subject=subject, session=session)
    
    
    '''
    run_best_condition_analysis(
        config, logger, 
        audio_epochs, no_audio_epochs, 
        subject, session
    )
    '''
    


    logger.info("Best Condition for subject=%s, session=%s:", subject, session)

def run_n100_pipeline(config, logger):
    logger.info("Starting N100 analysis pipeline.")
    layout = BIDSLayout(config['dataset']['BIDS_DIR'], validate=True)
    subject_ids = layout.get_subjects()
    for sub in subject_ids:
        if sub == '13':
            continue
        session_ids = layout.get_sessions(subject=sub)
        for ses in session_ids:
            run_n100_per_subject_session(config, logger, sub, ses)
        
        
    logger.info("N100 analysis pipeline completed.")