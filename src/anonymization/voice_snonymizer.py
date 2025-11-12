import os
import librosa
import soundfile as sf
import parselmouth
import numpy as np


class VoiceAnonymizerPipeline:
    """
    A pipeline-style class for voice anonymization using pitch and formant shifting.
    """

    def __init__(self, pitch_steps=4, formant_ratio=1.2, target_sr=16000, logger=None):
        """
        Initializes the VoiceAnonymizerPipeline.

        Args:
            pitch_steps (int): Number of semitones for pitch shift.
            formant_ratio (float): Scaling factor for formants.
            target_sr (int): Target sample rate for processing.
        """
        self.pitch_steps = pitch_steps
        self.formant_ratio = formant_ratio
        self.target_sr = target_sr
        self.logger = logger
        self.logger.info('Initializing VoicAnonymizer Pipeline;')
    def load(self, file_path):
        """
        Loads audio and resamples to target sample rate.

        Args:
            file_path (str): Path to input audio file.

        Returns:
            tuple: (audio, sample_rate)
        """
        self.logger.info('Loading audio file')
        try:
            audio, sr = librosa.load(file_path, sr=self.target_sr)
            return audio, sr
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {e}")

    def pitch_shift(self, audio, sr):
        """
        Applies pitch shifting.

        Args:
            audio (np.ndarray): Input audio.
            sr (int): Sample rate.

        Returns:
            np.ndarray: Pitch-shifted audio.
        """
        self.logger.info('Applying pitch shift')
        try:
            return librosa.effects.pitch_shift(audio, sr=sr, n_steps=self.pitch_steps)
        except Exception as e:
            raise RuntimeError(f"Failed during pitch shift: {e}")

    def formant_shift(self, audio, sr):
        """
        Applies formant shifting using Parselmouth's 'Change gender' method.

        Args:
            audio (np.ndarray): Input audio.
            sr (int): Sample rate.

        Returns:
            np.ndarray: Formant-shifted audio.
        """
        self.logger.info('Applying formant shift')
        try:
            snd = parselmouth.Sound(audio, sampling_frequency=sr)
            # Change gender parameters: pitch floor, pitch ceiling, formant ratio, pitch range, duration
            manipulated = parselmouth.praat.call(
                snd,
                "Change gender",
                75,                    # pitch floor in Hz
                600,                   # pitch ceiling in Hz
                self.formant_ratio,    # formant shift ratio
                1.0,                   # pitch range factor
                1.0,                   # duration factor
                1.0
            )
            return manipulated.values.T.flatten()
        except Exception as e:
            raise RuntimeError(f"Failed during formant shift: {e}")

    def transform(self, audio, sr, pitch=True, formant=True):
        """
        Applies pitch and formant shifting.

        Args:
            audio (np.ndarray): Input audio signal.
            sr (int): Sample rate.
            pitch (bool): Whether to apply pitch shifting.
            formant (bool): Whether to apply formant shifting.

        Returns:
            np.ndarray: Transformed anonymized audio.
        """
        self.logger.info('Anonymizing the audio')
        if pitch:
            audio = self.pitch_shift(audio, sr)
        if formant:
            audio = self.formant_shift(audio, sr)
        return audio

    def fit(self, X=None, y=None):
        """
        Placeholder for compatibility with scikit-learn pipelines.
        """
        return self

    def fit_transform(self, file_path, save_path=None):
        """
        Loads and transforms audio from a file path.

        Args:
            file_path (str): Path to the input audio file.
            save_path (str, optional): If provided, save anonymized output to this path.

        Returns:
            np.ndarray: Anonymized audio signal.
        """
        audio, sr = self.load(file_path)
        anonymized_audio = self.transform(audio, sr)
        if save_path:
            self.save(anonymized_audio, sr, save_path)
        return anonymized_audio

    def save(self, audio, sr, output_path):
        """
        Saves audio to a file.

        Args:
            audio (np.ndarray): Audio to save.
            sr (int): Sample rate.
            output_path (str): Destination file path.
        """
        self.logger.info('Saving audio')
        try:
            sf.write(output_path, audio, sr)
            print(f"Anonymized audio saved to '{output_path}'")
        except Exception as e:
            raise RuntimeError(f"Failed to save audio: {e}")
