from src.dataset.data_reader import BIDSDatasetReader
from src.dataset.eeg_epoch_builder import EEGEpochBuilder
import pdb
from mne.epochs import Epochs

class SpeechEEGDatasetLoader:
    def __init__(
        self,
        subject_id: str,
        session_id: str,
        label:int,
        condition_config: dict , 
        config,
        logger
    ) -> None:
        self.config=config
        self.logger = logger
        self.logger.info('Initializing SpeechEEGDatasetLoader')
        self.subject_id = subject_id
        self.session_id = session_id
        self.label = label
        self.condition_config = condition_config

    def load_data(self):
        self.bids_reader = BIDSDatasetReader(
            subject=self.subject_id,
            session=self.session_id,
            config=self.config,
            logger=self.logger
        )
        self.bids_reader.preprocess_eeg(bandpass=True, ica=True)
        self.eeg = self.bids_reader.processed_eeg
        
        return self
    
    def _create_epochs(self) -> Epochs:
        """
        Helper to create epochs using EEGEpochBuilder.

        Args:
            config (dict): Configuration dictionary for the condition.

        Returns:
            Epochs: MNE Epochs object
        """
        self.load_data()
        config = self.condition_config
        self.logger.info(config)
        return   EEGEpochBuilder(
            eeg_data=self.eeg,
            trial_mode=config["trial_mode"],
            trial_unit=config["trial_unit"],
            experiment_mode=config["experiment_mode"],
            trial_boundary=config["trial_boundary"],
            trial_type=config["trial_type"],
            modality=config["modality"],
            logger=self.logger,
            baseline=config['baseline']
        ).create_epochs(
            tmin=config["tmin"],
            tmax=config["tmax"]
        )
    
    def get_data(self):
        epochs = self._create_epochs()
        data = epochs.get_data()
        labels = [self.label for i in range(data.shape[0])]
        return data, labels