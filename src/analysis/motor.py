# ============================================================
# Full Class-Based Motor SNR Pipeline 
# ============================================================

import os
import gc
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import mne
import pdb


# ============================================================
# Epoch Builder
# ============================================================

class EpochBuilder:
    def __init__(
        self,
        tasks,
        event,
        modalities=None,
        tmin=-0.2,
        tmax=0.7,
        baseline=None,
        event_offset=0.0,
        picks=None,
    ):
        self.tasks = tasks
        self.event = event
        self.modalities = modalities
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.event_offset = event_offset
        self.picks = picks

    def build(self, raw):
        samples = []
        metadata = []

        for ann in raw.annotations:
            desc = ann["description"]
            onset = ann["onset"]

            matched_task = next(
                (t for t in self.tasks if f"{t}{self.event}:" in desc),
                None,
            )
            if matched_task is None:
                continue

            rhs = desc.split(":", 1)[1]
            modality, item = rhs.split("_", 1) if "_" in rhs else (rhs, "")

            if self.modalities and modality not in self.modalities:
                continue

            sample = raw.time_as_index(onset + self.event_offset)[0]
            samples.append(sample)

            metadata.append(
                dict(
                    task=matched_task,
                    modality=modality,
                    item=item,
                    description=desc,
                )
            )

        if not samples:
            raise RuntimeError("No matching annotations found.")

        events = np.c_[samples, np.zeros(len(samples), int), np.ones(len(samples), int)]

        return mne.Epochs(
            raw,
            events,
            event_id={"trial": 1},
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=self.baseline,
            picks=self.picks,
            preload=True,
            metadata=pd.DataFrame(metadata),
        )


# ============================================================
# Epoch Manager
# ============================================================

class EpochManager:
    def __init__(self, out_dir="results/motor/epochs"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def robust_concatenate(epochs_list):
        all_bads = set().union(*(ep.info["bads"] for ep in epochs_list))
        for ep in epochs_list:
            ep.info["bads"] = list(all_bads)

        mne.channels.equalize_channels(epochs_list, copy=False)

        sfreqs = {ep.info["sfreq"] for ep in epochs_list}
        if len(sfreqs) > 1:
            target = epochs_list[0].info["sfreq"]
            for ep in epochs_list:
                if ep.info["sfreq"] != target:
                    ep.resample(target)

        return mne.concatenate_epochs(epochs_list)

    def load_or_create(self, name, builder, raw_files, logger):
        out_file = self.out_dir / f"{name.lower()}-epo.fif"

        if out_file.exists():
            logger.info(f"Loading {name} epochs")
            return mne.read_epochs(out_file, preload=True)

        logger.info(f"Creating {name} epochs")
        epochs = []

        for f in raw_files:
            raw = mne.io.read_raw_fif(f, preload=False)
            epochs.append(builder.build(raw))

        epochs = self.robust_concatenate(epochs)
        epochs.save(out_file, overwrite=True)
        return epochs


# ============================================================
# TFR Computer
# ============================================================

class TFRComputer:
    def __init__(self, fmin, fmax, n_freqs, method="multitaper"):
        self.fmin = fmin
        self.fmax = fmax
        self.n_freqs = n_freqs
        self.method = method

    def compute(self, epochs, picks="eeg", n_jobs=-1):
        sfreq = float(epochs.info["sfreq"])
        nyq = sfreq / 2.0
        fmax_eff = min(self.fmax, nyq - 1e-6)

        if fmax_eff <= self.fmin:
            raise ValueError("Invalid frequency range")

        freqs = np.logspace(
            np.log10(self.fmin),
            np.log10(fmax_eff),
            self.n_freqs,
        )
        n_cycles = freqs / 2.0

        return epochs.compute_tfr(
            method=self.method,
            freqs=freqs,
            n_cycles=n_cycles,
            average=False,
            picks=picks,
            n_jobs=n_jobs,
            verbose=False,
        )


# ============================================================
# Batched TFR Processor
# ============================================================

class BatchTFRProcessor:
    def __init__(self, tfr_computer, batch_size=1000, out_dir="results/motor/tfr_batches"):
        self.tfr_computer = tfr_computer
        self.batch_size = batch_size
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(self, condition_name, epochs):
        n_epochs = len(epochs)

        for start in range(0, n_epochs, self.batch_size):
            end = min(start + self.batch_size, n_epochs)
            out_file = self.out_dir / f"{condition_name}_trials_{start}-{end-1}-tfr.h5"
            
            if out_file.exists():
                continue

            batch = epochs[start:end].copy()
            tfr = self.tfr_computer.compute(batch)
            tfr.save(out_file, overwrite=True)

            del batch, tfr
            gc.collect()


# ============================================================
# ROI Power Extractor
# ============================================================

class ROIPowerExtractor:
    def __init__(
        self,
        channels,
        tmin=0.0,
        tmax=0.6,
        fmin=20,
        fmax=300,
    ):
        self.channels = channels
        self.tmin = tmin
        self.tmax = tmax
        self.fmin = fmin
        self.fmax = fmax

    def extract(self, file):
        tfr = mne.time_frequency.read_tfrs(file)[0]
        tfr.pick(self.channels)
        tfr.crop(self.tmin, self.tmax, self.fmin, self.fmax)
        power = tfr.data.mean(axis=(1, 2, 3))
        del tfr
        gc.collect()
        return file, power


# ============================================================
# Main Pipeline
# ============================================================

class MotorPipeline:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        band = config["analysis"]["motor"]["bandpass"]

        self.tfr_computer = TFRComputer(
            fmin=band[0],
            fmax=band[1],
            n_freqs=config["analysis"]["motor"]["n_freqs"],
        )

        self.batch_processor = BatchTFRProcessor(self.tfr_computer)

        self.roi_extractor = ROIPowerExtractor(
            channels=[
                "F7", "T7", "T8", "FT7", "FT8",
                "TP7", "P7", "PO7", "TP8", "P8", "PO8",
            ]
        )

        self.epoch_manager = EpochManager()

    def run(self):
        raw_files = glob.glob("derivatives/motor/*.fif")[:24]
        
        overt_builder = EpochBuilder(
            tasks=["RealWordsExperiment", "RealSyllablesExperiment"],
            event="StartSpeech",
            modalities=["Pictures", "Text", "Audio"],
        )

        covert_builder = EpochBuilder(
            tasks=["SilentWordsExperiment", "SilentSyllablesExperiment"],
            event="StartSpeech",
            modalities=["Pictures", "Text", "Audio"],
        )

        rest_builder = EpochBuilder(
            tasks=["SilentWordsExperiment", "SilentSyllablesExperiment"],
            event="StartFixation",
            modalities=["Pictures", "Text", "Audio"],
            event_offset=0.3,
        )

        overt = self.epoch_manager.load_or_create("Overt", overt_builder, raw_files, self.logger)
        covert = self.epoch_manager.load_or_create("Covert", covert_builder, raw_files, self.logger)
        rest = self.epoch_manager.load_or_create("Rest", rest_builder, raw_files, self.logger)

        for name, epochs in {"Overt": overt, "Covert": covert, "Rest": rest}.items():
            self.batch_processor.run(name, epochs)
            
        
        self.aggregate_power()

    def aggregate_power(self):
        files = glob.glob("results/motor/tfr_batches/*.h5")
        results = {"Overt": [], "Covert": [], "Rest": []}
        
        

        with ProcessPoolExecutor(max_workers=3) as executor:
            for file, power in executor.map(self.roi_extractor.extract, files):
                for key in results:
                    if key in file:
                        results[key].append(power)

        out_dir = Path("results/motor/numpy")
        out_dir.mkdir(exist_ok=True)

        for key, values in results.items():
            np.save(out_dir / f"{key.lower()}.npy", np.concatenate(values))
