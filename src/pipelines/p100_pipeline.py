import os
import pandas as pd
from mne.epochs import Epochs
from pathlib import Path

from src.dataset.data_reader import BIDSDatasetReader
from src.dataset.eeg_epoch_builder import EEGEpochBuilder
from src.analysis.p_100_analyser import P100ComponentAnalyzer
from src.visualizations.p100_plotter import P100Plotter

from src.utils.graphics import log_print

import pdb




class P100Pipeline:
    def __init__(self, subject_id, session_id, config, logger, cond_1, cond_2, channels):
        self.subject_id = subject_id
        self.session_id = session_id
        self.config = config
        self.logger = logger
        self.cond_1 = cond_1
        self.cond_2 = cond_2
        self.channels = channels

    def run(self, plot=True):
        log_print(text=f'P100 Pipeline sub-{self.subject_id}, ses-{self.session_id}', logger=self.logger)

        bids_reader = BIDSDatasetReader(
            subject=self.subject_id,
            session=self.session_id,
            logger=self.logger,
            config=self.config
        )
        bids_reader.preprocess_eeg(bandpass=True, ica=True)
        eeg = bids_reader.processed_eeg

        self.cond_1_epochs = self._get_epochs_condition(eeg, self.cond_1)
        self.cond_2_epochs = self._get_epochs_condition(eeg, self.cond_2)

        cond_1_res, self.analyzer_1 = self._analyze_p100(self.cond_1_epochs, time_window=(0.08, 0.12))
        cond_2_res, self. analyzer_2 = self._analyze_p100(self.cond_2_epochs, time_window=(0.38, 0.42))
        
        if plot:
            self._save_results(cond_1_res, cond_2_res)
            self._plot()

    def _analyze_p100(self, epochs, time_window=(0.08, 0.12)):
        analyzer = P100ComponentAnalyzer(
            epochs=epochs,
            logger=self.logger,
            channels=self.channels, 
        )
        lat, peak, mean = analyzer.get_p100_peak(time_window=time_window)
        return [lat, peak, mean], analyzer

    def _get_epochs_condition(self, eeg, condition_cfg):
        self._log(f"Creating epochs for condition: {condition_cfg['label']}")
        
        epocher = EEGEpochBuilder(
            eeg_data=eeg,
            trial_mode=condition_cfg["trial_mode"],
            trial_unit=condition_cfg["trial_unit"],
            experiment_mode=condition_cfg["experiment_mode"],
            trial_boundary=condition_cfg["trial_boundary"],
            trial_type=condition_cfg["trial_type"],
            modality=condition_cfg["modality"],
            channels=self.channels,
            logger=self.logger,
            baseline=condition_cfg['baseline']
        )

        epochs = epocher.create_epochs(
            tmin=condition_cfg["tmin"],
            tmax=condition_cfg["tmax"]
        )
        return epochs

    def _save_results(self, res_1, res_2):
        self._log('Saving results to CSV')

        results_dir = self.config['analysis']['results_dir']
        output_dir = Path(os.getcwd(), results_dir, 'P100')
        os.makedirs(output_dir, exist_ok=True)

        cond_1_label = self.cond_1["label"]
        cond_2_label = self.cond_2["label"]

        results = {
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "condition": [cond_1_label, cond_2_label],
            "latency": [res_1[0], res_2[0]],
            "peak": [res_1[1], res_2[1]],
            "mean": [res_1[2], res_2[2]]
        }

        df = pd.DataFrame(results)
        csv_path = os.path.join(
            output_dir,
            f"sub-{self.subject_id}_ses-{self.session_id}_p100_{cond_1_label}_{cond_2_label}.csv"
        )

        df.to_csv(csv_path, index=False)
        self._log(f"Results saved to {csv_path}")
        
        
    def _plot(self):
        self.logger.info('Plotting P100')
        plotter = P100Plotter(
            condition1=self.analyzer_1,
            condition2=self.analyzer_2,
            name1=self.cond_1['label'],
            name2=self.cond_2['label'],
            sub_id=self.subject_id,
            ses_id=self.session_id,
            config=self.config,
            logger=self.logger
        )
        plotter.plot_evokeds()
        

    def _log(self, message):
        self.logger.info(message)




