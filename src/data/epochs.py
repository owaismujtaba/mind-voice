import numpy as np
import pandas as pd
import mne
import pdb

from src.utils import log_info

class EpochsData:
    def __init__(
        self, config, logger, raw, tasks,
        event, modalities=None,tmin=-0.2,
        tmax=0.5, baseline=None,
        event_offset=0.0, picks=None,
    ):
        self.config = config
        self.logger = logger
        log_info(logger=logger, text="Initializing EpochsData")
        self.raw = raw
        self.tasks = tuple(tasks)
        self.event = event
        self.modalities = set(modalities) if modalities else None
        self.event_offset = event_offset
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.picks = picks
        

        self._task_patterns = {
            task: f"{task}{self.event}:"
            for task in self.tasks
        }
        

    # ------------------------------------------------------------------
    # Epoch creation
    # ------------------------------------------------------------------
    def create_epochs(self):
        
        self.logger.info(
            "Creating epochs for tasks: %s, event: %s, modalities: %s",
            self.tasks,
            self.event,
            self.modalities,
        )
        self.logger.info(f'Time window: {self.tmin} to {self.tmax} sec')
        self.logger.info(f'Baseline: {self.baseline}')
        annotations = self.raw.annotations
        sfreq = self.raw.info["sfreq"]

        samples = []
        tasks = []
        modalities = []
        items = []
        descriptions = []

        for onset, duration, description in zip(
            annotations.onset,
            annotations.duration,
            annotations.description,
        ):
            matched_task = self._match_task(description)
            if matched_task is None:
                continue
            modality, item = self._parse_description(description)

            if self.modalities and modality not in self.modalities:
                continue

            onset_shifted = onset + self.event_offset
            sample = self.raw.time_as_index(onset_shifted)[0]

            samples.append(sample)
            tasks.append(matched_task)
            modalities.append(modality)
            items.append(item)
            descriptions.append(description)

        if not samples:
            raise RuntimeError(
                "No matching annotations found for the specified "
                "tasks, event, and modality filters."
            )

        
        
        events = self._build_events(samples)
        event_id = {"trial": 1}
        
        metadata = self._build_metadata(
            tasks,
            modalities,
            items,
            descriptions,
        )

        self.logger.info(
            "Creating %d epochs (tmin=%.3f, tmax=%.3f)",
            len(samples),
            self.tmin,
            self.tmax,
        )

        return mne.Epochs(
            self.raw,
            events=events,
            event_id=event_id,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=self.baseline,
            picks=self.picks,
            preload=True,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _match_task(self, description):
        for task, pattern in self._task_patterns.items():
            if pattern in description:
                return task
        return None

    @staticmethod
    def _parse_description(description):
        if ":" not in description:
            return "", ""

        rhs = description.split(":", 1)[1]
        if "_" in rhs:
            return rhs.split("_", 1)

        return rhs, ""

    @staticmethod
    def _build_events(samples):
        samples = np.array(samples, dtype=int)
        codes = np.ones_like(samples)
        events = np.c_[samples, np.zeros_like(samples), codes]
        return events

    @staticmethod
    def _build_metadata(tasks, modalities, items, descriptions):
        return pd.DataFrame(
            {
                "task": tasks,
                "modality": modalities,
                "item": items,
                "description": descriptions,
            }
        )
