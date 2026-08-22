import numpy as np
import yaml
import scipy
import time
import random
from pathlib import Path
from utils.config_finder import find_config_path
from typing import Optional

## REMARK: THIS FILE IS USED FOR TRAINING ONLY. HENCE, WE CAN USE TENSORFLOW.IO FOR AUDIO PROCESSING IF NEEDED

class AudioNorm:
    def __init__(self, window_size: float, sr: int, hop_size: float):
        self.window_size = window_size
        self.sampling_rate = sr
        self.hop_size = hop_size

    @classmethod
    def from_config(cls, config_path: Optional[Path] = None) -> "AudioNorm":
        """
        Use settings in config.yaml to normalize audio. 
        
        Uses the default hardcoded path but can be changed by the optional parameter config_path

        Params:
            config_path (str): The file path for the config.yaml file (OPTIONAL)
        """

        if config_path is None:
            config_path = find_config_path(__file__)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        try:
            shared_cfg = config["shared"]
            audio_cfg = config["audio_config"]
            merged_cfg = {
                "window_size": shared_cfg["window_size"],
                "sr": shared_cfg["sr"],
                "hop_size": shared_cfg["hop_size"]
            }
        except KeyError as e:
            raise ValueError(
                f"config.yaml is missing expected key {e}. Check that 'shared.sr', 'shared.window_size', and 'audio_config' are present."
            ) from e

        return cls(**merged_cfg)

    def normalize_sampling_rate(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Used to convert a sound sample to a target sampling rate

        Args:
            audio (np.ndarray): The audio for which the sampling rate must be changed
            orig_sr (int): The original sampling rate of the audio
            target_sr (int): The target sampling rate. It uses the default value from config.yaml if no argument is passed.
            
        Returns:
            The sound sample at the target sampling rate
        """

        if orig_sr is None:
            raise ValueError("Original sampling rate cannot be None.")
        elif orig_sr <= 0:
            raise ValueError("Original sampling rate cannot be zero or negative.")
        
        if target_sr == None:
            target_sr = self.sampling_rate
        elif target_sr <= 0:
            raise ValueError("Target sampling rate cannot be zero or negative.")

        if orig_sr == target_sr:
            return audio
        
        # Find the greatest common divisor to get the lowest possible integer ratios
        gcd = np.gcd(orig_sr, target_sr)
        
        # Up-sampling factor
        up = target_sr // gcd
        # Down-sampling factor
        down = orig_sr // gcd
        
        # Apply polyphase resampling
        y_resampled = scipy.signal.resample_poly(audio, up, down)
        
        return y_resampled

    def silence_padding(self, audio_samples: np.ndarray) -> np.ndarray:
        """
        Pads a 1-D audio np.ndarray up to the target window size, by adding samples at the end. 
        If the audio length >= window size ignore.

        ASSUMPTION: The audio input contains 1 full bird call. This is important as the padding
        can be done at the start or the end of the call.

        Args:
            audio_samples (np.ndarray): Audio samples data. Accepted shape is (Number of samples, )

        Returns:
            padded audio samples of shape (Number of samples,)

        """

        # Calculate the length of the audio
        number_of_samples = audio_samples.shape[0]
        target_sample_count = self.sampling_rate * self.window_size

        # If audio length >= window size, ignore it
        if number_of_samples >= target_sample_count:
            return audio_samples
        
        pad_amount = int(target_sample_count - number_of_samples)

        padded = np.pad(audio_samples, (0, pad_amount), mode="constant", constant_values=0.0)

        return padded.astype(np.float32)

    def noise_padding(self):
        """
        This will be done in the data augmentation part as Background Noise Injection (SNR between-class mixing)
        """
        raise NotImplementedError
    
    def segmentation(self, audio: np.ndarray) -> list[np.ndarray]:
        """
        Splits audio into overlapping fixed-length segments using a sliding window.
        Incomplete final windows are padded with silence at the end.

        Args:
            audio (np.ndarray): 1-D array of audio samples, shape (N,)

        Returns:
            List of np.ndarray segments, each of shape (window_samples,)
        """
        # Convert time durations to integer sample counts
        window_samples = int(self.window_size * self.sampling_rate)   # e.g. 3.0 * 44100 = 132300
        hop_samples    = int(self.hop_size   * self.sampling_rate)    # e.g. 1.0 * 44100 = 44100

        segments = []
        start = 0

        while start < len(audio):
            segment = audio[start : start + window_samples]  # slice the window, check for 

            # If this slice is shorter than the window, pad it to the full window size using silence at the end
            if len(segment) < window_samples:
                segment = self.silence_padding(segment)

            segments.append(segment)
            start += hop_samples   # increment by hop

        return segments

if __name__ == "__main__":
    audionorm = AudioNorm.from_config()
    print("Ran comfortably")