from logging import log
import mne
import os
import pandas as pd
from pathlib import Path
import numpy as np
from src.data.data_reader import DataReader
from src.data.epochs import EpochsData

from bids import BIDSLayout
from pyprep.prep_pipeline import PrepPipeline
from mne.preprocessing import ICA

from src.utils import log_info
from src.data.epochs import EpochsData

from imblearn.over_sampling import RandomOverSampler  
from src.decoding.model import OvertCoverRestClassifier

import pdb
from src.utils import set_all_seeds

set_all_seeds()

class OvertCovertRestDataset:
    def __init__(self, eeg, logger, config):
        self.logger = logger
        self.config = config
        log_info(logger=logger, text='Initializing OvertCovertRestDataset')
        self.eeg = eeg
        self.tmin = config['decoding']['tmin']
        self.tmax = config['decoding']['tmax']
        

    def _get_overt_condition(self): 
        tasks=['RealWordsExperiment']
        event='StartStimulus'
        modalities=['Audio', 'Text', 'Pictures']
        
        return tasks, event, modalities
        
    def _get_covert_condition(self): 
        tasks=['SilentWordsExperiment']
        event='StartStimulus'
        modalities=['Audio', 'Text', 'Pictures']
        return tasks, event, modalities
        
    def _get_rest_condition(self): 
        tasks=['SilentWordsExperiment', 'RealWordsExperiment']
        event='StartFixation'
        modalities=['Audio', 'Text', 'Pictures']
        return tasks, event, modalities
    
    
    def get_epochs(self):
        tasks, event, modalities = self._get_covert_condition()
        epocher = EpochsData(
            raw= self.eeg, logger = self.logger, config = self.config,
            tasks=tasks,  event=event, modalities=modalities,
            tmin= self.tmin,   tmax=self.tmax,
            baseline=(self.config['decoding']['tmin'], 0)
        )
        covert_epochs = epocher.create_epochs()
        
        tasks, event, modalities = self._get_overt_condition()
        epocher = EpochsData(
            raw= self.eeg, logger = self.logger, config = self.config,
            tasks=tasks,  event=event, modalities=modalities,
            tmin= self.tmin,   tmax=self.tmax,
            baseline=(self.config['decoding']['tmin'], 0)
        )
        overt_epochs = epocher.create_epochs()
        
        tasks, event, modalities = self._get_rest_condition()
        epocher = EpochsData(
            raw= self.eeg, logger = self.logger, config = self.config,
            tasks=tasks,  event=event, modalities=modalities,
            tmin= self.tmin,   tmax=self.tmax,
            baseline=(self.config['decoding']['tmin'], 0)
        )
        rest_epochs = epocher.create_epochs()
        
        return overt_epochs, covert_epochs, rest_epochs

class DecodingPipeline:
    def __init__(self, config, logger, overt, covert, rest, subject, session):
        self.config = config
        self.logger = logger
        self.subject = subject
        self.session = session
        log_info(logger=logger, text='Decoder')
        self.logger.info('Subject: %s Session: %s', subject, session)
        self.overt = overt
        self.covert = covert
        self.rest = rest
        self.output_dir = 'results/decoding'
        
    def _get_data(self, epochs, label):
        data = epochs.get_data()
        labels = [label for i in range(data.shape[0])]
        return data, labels
    
    def load_data(self):
        overt, overt_labels = self._get_data(epochs=self.overt, label=0)
        covert, covert_labels = self._get_data(epochs=self.covert, label=1)
        rest, rest_labels = self._get_data(epochs=self.rest, label=2)

        X = np.concatenate([overt, covert, rest], axis=0)
        y = np.concatenate([overt_labels, covert_labels, rest_labels], axis=0)
        
        
        n_samples, n_channels, n_timepoints = X.shape
        X_reshaped = X.reshape(n_samples, -1)

        # Apply oversampling
        self.logger.info('Removing data imbalance using RandomOverSampler')
        ros = RandomOverSampler(random_state=42)
        X_balanced, y_balanced = ros.fit_resample(X_reshaped, y)

        # Reshape back to original shape
        X_balanced = X_balanced.reshape(-1, n_channels, n_timepoints)

        X = X_balanced[:,:200:]
        
        X = self.normalize(X)
        y = y_balanced

        self.logger.info(f"Data loaded and oversampled: {X.shape[0]} samples, "
              f"{X.shape[1]} channels, {X.shape[2]} timepoints")
        
        
        return X, y
    
    
    def train(self):
        X, y = self.load_data()
        
        input_shape = (X.shape[1], X.shape[2])
        self.model = OvertCoverRestClassifier(input_shape=input_shape)
        self.model.compile_model()
        self.model.summary()
        self.accuracy, self.report, self.confusion_matrix = self.model.train_with_split(
            x=X, y=y, logger=self.logger,
            epochs = self.config['decoding']['epochs'],
            batch_size=self.config['decoding']['batch_size'],
        )
        print("Training completed.")
        self.save_results()
        
    
    def save_results(self):
        # Save training history
        os.makedirs(self.output_dir, exist_ok=True)
        acc_df = pd.DataFrame({'accuracy':[self.accuracy]})
        filename = f"sub-{self.subject}_ses-{self.session}_overt_covert_rest_accuracy.csv"
        filepath = Path(self.output_dir, filename)
        acc_df.to_csv(filepath, index=False)
        self.logger.info(f"Training history saved to {filepath}")
        
        report_df = pd.DataFrame(self.report).transpose()
        report_filename = f"sub-{self.subject}_ses-{self.session}_classification_report.csv"
        report_path = Path(self.output_dir, report_filename)
        report_df.to_csv(report_path)
        self.logger.info(f"Classification report saved to {report_path}")

        cm_df = pd.DataFrame(self.confusion_matrix)
        cm_filename = f"sub-{self.subject}_ses-{self.session}_confusion_matrix.csv"
        cm_path = Path(self.output_dir, cm_filename)
        cm_df.to_csv(cm_path, index=False)
        self.logger.info(f"Confusion matrix saved to {cm_path}")
        
    def normalize(self, X):
        self.logger.info('Normalizing X')
        mean = X.mean(axis=2, keepdims=True)  # shape: (N, channels, 1)
        std = X.std(axis=2, keepdims=True) + 1e-8
        return (X - mean) / std

class EEGPreprocessor:
    """EEG preprocessing pipeline using PREP and ICA."""

    def __init__(self, subject, session, config, logger):
        self.subject = subject
        self.session = session
        self.config = config
        self.logger = logger
        log_info(logger=logger, text='Initializing EEGProcessor')
        self.logger.info('sub-%s, ses-%s', subject, session)

        self.derivatives_dir = "derivatives/decoding"
        os.makedirs(self.derivatives_dir, exist_ok=True)

        subject_id = f"sub-{subject}"
        session_id = f"_ses-{session}" if session else ""
        self.filename = (
            f"{subject_id}{session_id}"
            "_task-VCV_run-01_desc-preproc_eeg.fif"
        )
        self.filepath = os.path.join(
            self.derivatives_dir, self.filename
        )

        self.raw_eeg = None
        self.preprocessed_data = None

    def load_if_exists(self):
        """Load preprocessed data if it already exists."""
        if os.path.exists(self.filepath):
            self.logger.info("Preprocessed file found. Loading: %s",self.filepath)
            self.preprocessed_data = mne.io.read_raw_fif(
                self.filepath, preload=True
            )
            return True
        return False

    def load_raw_data(self):
        """Load raw EEG data using DataReader."""
        reader = DataReader(
            config=self.config,
            logger=self.logger,
            subject=self.subject,
            session=self.session,
        )
        reader.load_raw_data()
        self.raw_eeg = reader.raw_eeg
        self.raw_eeg.set_eeg_reference([])

    def setup_channel_types(self):
        """Set EOG channel types if available."""
        try:
            self.raw_eeg.set_channel_types(
                {"EOG1": "eog", "EOG2": "eog"}
            )
        except (KeyError, ValueError):
            self.logger.warning(
                "EOG channels not found; skipping channel setup"
            )

    def run_prep_pipeline(self):
        """Run the PREP pipeline."""
        montage_name = self.config["dataset"].get("MONTAGE", "standard_1020")
        random_state = self.config["preprocessing"]["RANDOM_STATE"]
        notch_freqs = self.config["analysis"]["motor"].get("linenoise")

        montage = mne.channels.make_standard_montage(montage_name)
        self.raw_eeg.set_montage(montage)

        prep_params = {
            "ref_chs": "eeg",
            "reref_chs": "eeg",
            "line_freqs": notch_freqs,
        }

        self.logger.info("Running PREP pipeline")
        prep = PrepPipeline(
            raw=self.raw_eeg.copy(),
            prep_params=prep_params,
            montage=montage,
            channel_wise=True,
            random_state=random_state,
        )
        prep.fit()

        return prep.raw

    def run_ica(self, raw_prep):
        """Run ICA for EOG artifact removal."""
        raw_ica = raw_prep.copy().filter(
            l_freq=1.0,
            h_freq=None,
            fir_design="firwin",
        )

        picks_eeg = mne.pick_types(
            raw_ica.info, eeg=True, eog=False
        )

        ica_params = self.config["preprocessing"][
            "ICA_PARAMS"
        ]
        ica = ICA(
            n_components=ica_params["n_components"],
            random_state=ica_params["random_state"],
            method=ica_params.get(
                "ica_method", "fastica"
            ),
        )

        self.logger.info("Fitting ICA")
        ica.fit(raw_ica, picks=picks_eeg)

        eog_picks = mne.pick_types(
            raw_prep.info, eog=True
        )
        eog_ch_names = [
            raw_prep.ch_names[i] for i in eog_picks
        ]

        if eog_ch_names:
            eog_inds, scores = ica.find_bads_eog(
                raw_prep, ch_name=eog_ch_names
            )
        else:
            eog_inds, scores = ica.find_bads_eog(
                raw_prep
            )

        self.logger.info(
            "EOG ICA scores: %s", scores
        )

        ica.exclude = eog_inds
        raw_clean = raw_prep.copy()
        ica.apply(raw_clean)

        self.logger.info("ICA applied; excluded components: %s", ica.exclude)

        return raw_clean

    def apply_final_filter(self, raw_clean):
        """Apply final EEG band-pass filter."""
        self.logger.info('Applying bandpass')
        bandpass = self.config["decoding"]["bandpass"]
        raw_clean.filter(
            l_freq=bandpass[0],
            h_freq=bandpass[1],
            fir_design="firwin",
        )
        return raw_clean

    def save(self):
        """Save preprocessed EEG data to disk."""
        self.preprocessed_data.save(self.filepath, overwrite=True)
        self.logger.info("Preprocessed data saved to %s", self.filepath)

    def run(self):
        """Execute the full preprocessing pipeline."""
        if self.load_if_exists():
            return self.preprocessed_data

        self.load_raw_data()
        self.setup_channel_types()
        raw_prep = self.run_prep_pipeline()
        raw_clean = self.run_ica(raw_prep)
        self.preprocessed_data = self.apply_final_filter(
            raw_clean
        )

        self.logger.info("Preprocessing completed")
        self.save()

        return self.preprocessed_data


def run_decoding_per_subject_session(config, logger, sub, ses):
    preprocessor = EEGPreprocessor(
    subject=sub, session=ses,
            config=config, logger=logger
    )
    processed_data = preprocessor.run()
    
    
    datatset = OvertCovertRestDataset(
        eeg=processed_data, logger=logger, config=config
    )
    overt, covert, rest = datatset.get_epochs()
    
    decoder = DecodingPipeline(
        config=config, logger=logger, 
        subject=sub, session=ses,
        overt=overt, covert=covert, rest=rest
    )
    
    decoder.train()
    
    
    
    
def run_decoding_pipeline(config, logger):
    output_dir = 'derivatives/decoding'
    os.makedirs(output_dir, exist_ok= True)
    
    log_info(logger=logger, text="Starting Decoding pipeline.")
    layout = BIDSLayout(config['dataset']['BIDS_DIR'], validate=True)
    subject_ids = layout.get_subjects()
    for sub in subject_ids:
        session_ids = layout.get_sessions(subject=sub)
        for ses in session_ids:
            if sub == '13' and ses == '01':
                continue       
            run_decoding_per_subject_session(config, logger, sub, ses)
            
        
    