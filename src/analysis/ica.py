from src.dataset.data_reader import BIDSDatasetReader
from mne_bids import BIDSPath, read_raw_bids
from src.utils.graphics import styled_print
from mne.preprocessing import ICA
import mne
from pyprep import NoisyChannels
from pathlib import Path
import os
import json
import numpy as np

import pdb


class ICADataLoader:
    def __init__(self, subject_id, session_id, config, logger):
        self.subject_id = subject_id
        self.session_id = session_id
        self.config = config
        self.logger = logger

        self.logger.info(f"Initializing ICADataloader for Subject: {subject_id}, Session: {session_id}")

        data_reader = BIDSDatasetReader(
            config=self.config,
            logger=logger,
            subject=self.subject_id,
            session=self.session_id
        )
                
        data_reader.preprocess_eeg(bandpass=True, ica=True)
        
        self.raw = data_reader.raw_eeg
        self.cleaned_raw = data_reader.processed_eeg