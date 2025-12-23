import os

import mne
from mne.preprocessing import ICA
from mne_bids import BIDSPath, read_raw_bids
from pyprep.prep_pipeline import PrepPipeline


class DataReader:
    def __init__(self, config, logger, subject, session=None):
        self.config = config
        self.logger = logger
        self.subject = subject
        self.session = session

        self.logger.info(
            "Initializing DataReader | subject=%s, session=%s",
            subject,
            session,
        )

        self._setup_paths()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def _setup_paths(self):
        bids_root = self.config["dataset"]["BIDS_DIR"]

        self.bids_path = BIDSPath(
            subject=self.subject,
            session=self.session,
            datatype="eeg",
            task="VCV",
            run="01",
            root=bids_root,
        )

        self.logger.info("BIDS path set to %s", self.bids_path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_raw_data(self):
        self.logger.info("Loading raw EEG data from BIDS")
        self.raw_eeg = read_raw_bids(
            self.bids_path,
            verbose=False,
        )
    
        self.logger.info("Raw EEG data loaded")
        self.raw_eeg.load_data()
        self.logger.info("Raw EEG data preloaded into memory")

    def _set_channel_types(self):
        try:
            self.raw_eeg.set_channel_types(
                {"EOG1": "eog", "EOG2": "eog"}
            )
        except Exception:
            self.logger.warning(
                "EOG channels not found; skipping channel type setup"
            )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def preprocess_data(self):
        self.load_raw_data()
        self.raw_eeg.set_eeg_reference([])
        self._set_channel_types()

        montage_name = self.config["dataset"].get(
            "MONTAGE", "standard_1020"
        )
        random_state = self.config["preprocessing"]["RANDOM_STATE"]
        notch_freqs = self.config["preprocessing"].get("NOTCH")

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

        raw_prep = prep.raw

        # --------------------------------------------------------------
        # ICA (high-pass only for ICA training)
        # --------------------------------------------------------------
        raw_ica = raw_prep.copy().filter(
            l_freq=1.0,
            h_freq=None,
            fir_design="firwin",
        )

        picks_eeg = mne.pick_types(
            raw_ica.info,
            eeg=True,
            eog=False,
        )

        ica_params = self.config["preprocessing"]["ICA_PARAMS"]
        ica = ICA(
            n_components=ica_params["n_components"],
            random_state=ica_params["random_state"],
            method=ica_params.get("ica_method", "fastica"),
        )

        self.logger.info("Fitting ICA")
        ica.fit(raw_ica, picks=picks_eeg)

        eog_picks = mne.pick_types(
            raw_prep.info,
            eog=True,
        )
        eog_ch_names = [
            raw_prep.ch_names[i] for i in eog_picks
        ]

        if eog_ch_names:
            eog_inds, scores = ica.find_bads_eog(
                raw_prep,
                ch_name=eog_ch_names,
            )
        else:
            eog_inds, scores = ica.find_bads_eog(raw_prep)

        self.logger.info("EOG ICA scores: %s", scores)

        ica.exclude = eog_inds
        raw_clean = raw_prep.copy()
        ica.apply(raw_clean)

        self.logger.info(
            "ICA applied; excluded components: %s",
            ica.exclude,
        )

        # --------------------------------------------------------------
        # Final EEG filtering
        # --------------------------------------------------------------
        eeg_filter = self.config["preprocessing"]["EEG_FILTER"]
        raw_clean.filter(
            l_freq=eeg_filter["l_freq"],
            h_freq=eeg_filter["h_freq"],
            fir_design="firwin",
        )

        self.preprocessed_data = raw_clean
        self.logger.info("Preprocessing completed")

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------
    def get_preprocessed_data(self, directory=None):
        if directory:
            save_path = self._get_save_path(directory=directory)
        else:
            save_path = self._get_save_path()

        if os.path.exists(save_path):
            self.logger.info(
                "Loading preprocessed data from %s",
                save_path,
            )
            self.preprocessed_data = mne.io.read_raw_fif(
                save_path,
                preload=True,
            )
        else:
            self.logger.info(
                "Preprocessed file not found; running preprocessing"
            )
            self.preprocess_data()
            self._save_preprocessed_data()

        return self.preprocessed_data

    def _get_save_path(self, directory=None):
        if directory:
            derivatives_dir=directory
        else:
            derivatives_dir = "derivatives/preprocessed_data"
        os.makedirs(derivatives_dir, exist_ok=True)

        subject = f"sub-{self.subject}"
        session = (
            f"_ses-{self.session}" if self.session else ""
        )

        filename = (
            f"{subject}{session}_task-VCV_run-01_"
            "desc-preproc_eeg.fif"
        )

        return os.path.join(derivatives_dir, filename)

    def _save_preprocessed_data(self):
        save_path = self._get_save_path()
        self.preprocessed_data.save(
            save_path,
            overwrite=True,
        )

        self.logger.info(
            "Preprocessed data saved to %s",
            save_path,
        )
