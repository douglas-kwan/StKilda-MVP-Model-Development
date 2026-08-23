import numpy as np
import yaml
import scipy
import random
from pathlib import Path
from utils.config_finder import find_config_path
from typing import Optional, Tuple

## REMARK: THIS FILE IS USED FOR TRAINING ONLY. HENCE, WE CAN USE TENSORFLOW.IO FOR AUDIO PROCESSING IF NEEDED

class AudioNorm:
    def __init__(self, window_size: float, sr: int, hop_size: float, clip_pad_max_ratio: float):
        """
        Constructor for the AudioNorm class. Initializes the window size, sampling rate, and hop size for audio normalization.

        Args:
            window_size (float): The duration of each audio segment in seconds.
            sr (int): The target sampling rate in Hz.
            hop_size (float): The hop size for overlapping segments in seconds.
        """
        self._window_size = window_size
        self._sampling_rate = sr
        self._hop_size = hop_size
        self._clip_pad_max_ratio = clip_pad_max_ratio

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
                "hop_size": audio_cfg["hop_size"],
                "clip_pad_max_ratio": audio_cfg["clip_pad_max_ratio"]
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
            target_sr = self._sampling_rate
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

    def random_clipping(self, audio_samples: np.ndarray, orig_sr: int) -> Tuple[np.ndarray, int]:
        """
        This function handles audio samples of all durations.

        It firsts resamples the input audio to the sampling rate set in config.yaml file. It pads the audio with
        silence at both the start and the end based on the clip_pad_max_ratio in config.yaml. Finally, it clips a 
        random window of duration set in config.yaml from the padded audio.

        @Params:
            audio_samples (np.ndarray): The samples of the input audio
            orig_sr (int): The original sampling rate of the audio

        @Return:
            clipped audio (np.ndarray): The clipped audio sample
            sample rate (int): The sample rate as set in config.yaml
        """

        # Check if audio has same sampling rate as in config.yaml file
        if orig_sr != self._sampling_rate:
            audio_samples = self.normalize_sampling_rate(audio_samples, orig_sr, self._sampling_rate)

        # Calculate the number of samples in a window
        window_sample_amount = int(self._window_size * self._sampling_rate)

        # Calculate the amount of silence padding we can apply at the start and end
        max_pad_amount = int(self._clip_pad_max_ratio * window_sample_amount)

        # Obtain the padded audio
        padded_audio = np.pad(audio_samples, (max_pad_amount, max_pad_amount), mode="constant", constant_values=0.0)

        # Calculate the max_clip_start_index
        max_clip_start_index = len(padded_audio) - window_sample_amount

        # Get the clip start index
        clip_start_index = random.randint(0, max_clip_start_index)

        # Cut the clip from the audio
        clipped_audio = padded_audio[clip_start_index: clip_start_index + window_sample_amount]

        return clipped_audio, self._sampling_rate




if __name__ == "__main__":
    audionorm = AudioNorm.from_config()
    print("Ran comfortably")