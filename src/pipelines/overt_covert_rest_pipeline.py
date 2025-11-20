import os
import numpy as np
import pandas as pd
from pathlib import Path
from imblearn.over_sampling import RandomOverSampler  

from src.decoding.overt_covert_rest import SpeechEEGDatasetLoader
from src.decoding.overt_covert_rest_model import OvertCoverRestClassifier
import pdb

class OvertCovertRestPipeline:
    def __init__(self, subject_id='01', session_id='01', config=None, logger=None):
        self.subject_id = subject_id
        self.session_id = session_id
        
        self.model = None
        self.history = None
        self.config = config
        self.logger = logger
        self.logger.info('Initializing OvertCovertRestPipeline')
        cur_dir = os.getcwd()
        self.output_dir = Path(
            cur_dir, config['analysis']['results_dir'],
            'DecodingResults'
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_condition_config(self, trial_mode, trial_type):
        return {
            "trial_mode": trial_mode,
            "trial_unit": 'Words',
            "experiment_mode": 'Experiment',
            "trial_boundary": 'Start',
            "trial_type": trial_type,
            "modality": '',
            "tmin": -0.2,
            "tmax": 1.5
        }

    def _load_condition_data(self, label, config):
        loader = SpeechEEGDatasetLoader(
            subject_id=self.subject_id,
            session_id=self.session_id,
            label=label,
            condition_config=config,
            config = self.config,
            logger = self.logger
        )
        return loader.get_data()

    def load_data(self):
        
        overt_cfg = self._get_condition_config('Real', 'Speech')
        covert_cfg = self._get_condition_config('Silent', 'Speech')
        rest_cfg = self._get_condition_config('', 'Fixation')

        overt, overt_labels = self._load_condition_data(0, overt_cfg)
        covert, covert_labels = self._load_condition_data(1, covert_cfg)
        rest, rest_labels = self._load_condition_data(2, rest_cfg)

        X = np.concatenate([overt, covert, rest], axis=0)
        y = np.concatenate([overt_labels, covert_labels, rest_labels], axis=0)

        # Reshape for oversampling: (samples, features)
        n_samples, n_channels, n_timepoints = X.shape
        X_reshaped = X.reshape(n_samples, -1)

        # Apply oversampling
        self.logger.info('Removing data imbalance using RandomOverSampler')
        ros = RandomOverSampler(random_state=42)
        X_balanced, y_balanced = ros.fit_resample(X_reshaped, y)

        # Reshape back to original shape
        X_balanced = X_balanced.reshape(-1, n_channels, n_timepoints)

        self.X = X_balanced[:,:200:]
        self.X = self.normalizePerSamplePerChannel(self.X)
        self.y = y_balanced

        self.logger.info(f"Data loaded and oversampled: {self.X.shape[0]} samples, "
              f"{self.X.shape[1]} channels, {self.X.shape[2]} timepoints")
        
    def normalizePerSamplePerChannel(self, X):
        """
        Normalize each (sample, channel) pair independently over timepoints.
        """
        self.logger.info('Normalizing X')
        mean = X.mean(axis=2, keepdims=True)  # shape: (N, channels, 1)
        std = X.std(axis=2, keepdims=True) + 1e-8
        return (X - mean) / std

    def train(self, test_split=0.2):
        input_shape = (self.X.shape[1], self.X.shape[2])
        self.model = OvertCoverRestClassifier(inputShape=input_shape)
        self.model.compileModel()
        self.model.summary()
        self.accuracy, self.report, self.confusion_matrix = self.model.trainWithSplit(self.X, self.y, validationSplit=test_split, logger=self.logger)
        print("Training completed.")


    def save_results(self):
        # Save training history
        acc_df = pd.DataFrame({'accuracy':[self.accuracy]})
        filename = f"sub-{self.subject_id}_ses-{self.session_id}_overt_covert_rest_accuracy.csv"
        filepath = Path(self.output_dir, filename)
        acc_df.to_csv(filepath, index=False)
        self.logger.info(f"Training history saved to {filepath}")
        
        report_df = pd.DataFrame(self.report).transpose()
        report_filename = f"sub-{self.subject_id}_ses-{self.session_id}_classification_report.csv"
        report_path = Path(self.output_dir, report_filename)
        report_df.to_csv(report_path)
        self.logger.info(f"Classification report saved to {report_path}")

        cm_df = pd.DataFrame(self.confusion_matrix)
        cm_filename = f"sub-{self.subject_id}_ses-{self.session_id}_confusion_matrix.csv"
        cm_path = Path(self.output_dir, cm_filename)
        cm_df.to_csv(cm_path, index=False)
        self.logger.info(f"Confusion matrix saved to {cm_path}")

    def run(self):
        self.load_data()
        self.train()
        self.save_results()
