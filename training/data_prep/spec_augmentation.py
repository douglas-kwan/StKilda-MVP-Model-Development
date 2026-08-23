import yaml
from pathlib import Path
from typing import Optional, Tuple
import tensorflow as tf
from utils.config_finder import find_config_path


class SpecAugment:
    def __init__(
        self, 
        n_mels: int,
        freq_mask_param: int = 12, 
        time_mask_param: int = 20, 
        n_freq_masks: int = 1, 
        n_time_masks: int = 1
    ):
        """
        Initializes the SpecAugment configuration.
        """
        self.n_mels = n_mels
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    @classmethod
    def from_config(cls, config_path: Optional[Path] = None) -> "SpecAugment":
        """
        Use settings in config.yaml to configure SpecAugment. 
        
        Uses the default hardcoded path but can be changed by the optional parameter config_path

        Params:
            config_path (str): The file path for the config.yaml file (OPTIONAL)
        """
        if config_path is None:
            config_path = find_config_path(__file__)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        try:
            spec_cfg = config["spec_config"]
            spec_aug_cfg = config["spec_aug_config"]
            
            merged_cfg = {
                "n_mels": spec_cfg["n_mels"],
                "freq_mask_param": spec_aug_cfg["freq_mask_param"],  
                "time_mask_param": spec_aug_cfg["time_mask_param"],
                "n_freq_masks": spec_aug_cfg["n_freq_masks"],
                "n_time_masks": spec_aug_cfg["n_time_masks"]
            }
        except KeyError as e:
            raise ValueError(
                f"config.yaml is missing expected key {e}. Check that 'spec_config' is present."
            ) from e

        return cls(**merged_cfg)

    def augment(self, spectrogram: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        The main mapping function. Takes in a spectrogram and label tensor, 
        applies the masking, and returns them for the dataset pipeline.
        """
        aug_spec = spectrogram
        
        for _ in range(self.n_freq_masks):
            aug_spec = self._frequency_masking(aug_spec)
            
        for _ in range(self.n_time_masks):
            aug_spec = self._time_masking(aug_spec)
            
        return aug_spec, label
    
    def _frequency_masking(self, spectrogram: tf.Tensor) -> tf.Tensor:
        """Applies frequency masking by identifying the mel-frequency axis."""
        shape = tf.shape(spectrogram)
        # Dynamically infer the frequency axis based on the n_mels parameter from config
        freq_axis = tf.cond(tf.equal(shape[0], self.n_mels), lambda: 0, lambda: 1)
        
        return self._mask_axis(spectrogram, self.freq_mask_param, freq_axis)

    def _time_masking(self, spectrogram: tf.Tensor) -> tf.Tensor:
        """Applies time masking by targeting the non-frequency axis."""
        shape = tf.shape(spectrogram)
        # The time axis is the one that is NOT the frequency axis
        time_axis = tf.cond(tf.equal(shape[0], self.n_mels), lambda: 1, lambda: 0)
        
        return self._mask_axis(spectrogram, self.time_mask_param, time_axis)
  
    def _mask_axis(self, spec: tf.Tensor, max_mask_size: int, axis: int) -> tf.Tensor:
        shape = tf.shape(spec)
        axis_len = shape[axis]
        
        max_size = tf.minimum(max_mask_size, axis_len)
        
        mask_size = tf.random.uniform(shape=[], minval=0, maxval=max_size + 1, dtype=tf.int32)
        mask_start = tf.random.uniform(shape=[], minval=0, maxval=axis_len - mask_size + 1, dtype=tf.int32)
        mask_end = mask_start + mask_size
        
        # True for unmasked areas, False for masked areas
        indices = tf.range(axis_len, dtype=tf.int32)
        unmasked_region = tf.logical_or(indices < mask_start, indices >= mask_end)
        
        # Reshape boolean mask for broadcasting
        rank = tf.rank(spec)
        mask_shape = tf.where(tf.equal(tf.range(rank), axis), axis_len, 1)
        unmasked_region = tf.reshape(unmasked_region, mask_shape)
        
        # Replace masked areas with the minimum dB value of the spectrogram (true silence)
        min_db_value = tf.reduce_min(spec)
        return tf.where(unmasked_region, spec, min_db_value)

if __name__ == "__main__":
    spec_aug = SpecAugment.from_config()