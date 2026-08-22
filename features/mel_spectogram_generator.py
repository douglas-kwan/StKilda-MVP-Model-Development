import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt
import yaml
from utils.config_finder import find_config_path
from typing import Optional
from pathlib import Path

class MelSpectrogramGen:

    def __init__(self, 
                 sr: int = 44100,
                 low_hz: float = 500.0,
                 high_hz: float = 6000.0,
                 filter_order: int = 4,
                 n_fft: int = 1024,
                 hop_length: int = 512,
                 n_mels: int = 64):
        """
        Constructor for MelSpectrogramGen class

        Args:
            sr (int): The sampling rate of input audio in Hertz
            low_hz (float): The lowest frequency allowed by the bandpass filter
            high_hz (float): The highest frequency allowed by the bandpass filter
            filter_order (int): Controls how sharp the bandpass filter's cutoff is. Higher implies sharper transition but more computation
            n_fft (int): Size (in audio samples) of each little chunk of audio used to measure frequency content. Affects resolution of spectogram. Large size increases total sound sample in chunk (Time) but decreases frequency resolution
            hop_length (int): How far (in samples) we slide forward between each chunk. Smaller = more overlap = smoother spectrogram = more compute
            n_mels (int): Number of frequency bands in the final spectrogram (its height). This becomes part of the model input shape
        """
        self.sr = sr
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Bandpass filter mathematical coefficients
        self._sos = self._design_bandpass(self.low_hz, self.high_hz, filter_order)

        # mel filterbank matrix 
        self._mel_fb = self._build_mel_filterbank(self.low_hz, self.high_hz)

        # A "window" function (a smooth taper shape) applied to each little audio chunk before measuring its frequency content. 
        # This reduces artifacts at the edges of each chunk. Hanning is a standard, safe choice.
        self._window = np.hanning(n_fft).astype(np.float32)


    @classmethod
    def from_config(cls, config_path: Optional[Path] = None) -> "MelSpectrogramGen":
        """
        Use the settings found in config.yaml file to create spectrogram.

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
            spec_cfg = config["spec_config"]
            merged_cfg = {"sr": shared_cfg["sr"], **spec_cfg}
        except KeyError as e:
            raise KeyError(
                f"config.yaml is missing expected key {e}. Check that 'shared.sr' and 'spec_config' are present."
            ) from e

        return cls(**merged_cfg)

    ###########################################################################
    #                           PUBLIC METHODS                                #
    ###########################################################################
    def bandpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Applied the bandpass filter

        Args:
            audio (np.ndarray): A 1-D array of audio samples, dtype float32, with values roughly between -1.0 and 1.0.  (e.g. one fixed-length block recorded from the microphone, or a clip loaded from a file.)

        Returns:
            The filter audio with same shape and dtype
        """

        # Check that input is valid before applying bandpass maths operations
        self._validate_audio(audio)

        # Apply filter to audio
        # return sosfilt(self._sos, audio).astype(np.float32)
        return sosfiltfilt(self._sos, audio).astype(np.float32)
    
    def generate_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Generate the mel spectrogram.
        Note: It does not apply the bandpass filter

        Args:
            audio (np.ndarray): A 1-D array of audio samples, dtype float32, with values roughly between -1.0 and 1.0.  (e.g. one fixed-length block recorded from the microphone, or a clip loaded from a file.)
        
        Returns
        -------
        np.ndarray
            A 2-D array of shape (n_mels, n_frames) — this is the "image"
            the model actually sees. n_mels is the number of frequency
            bands (set in config), n_frames depends on the audio length,
            n_fft, and hop_length.
        """
        self._validate_audio(audio)

        power_spec = self._stft_power(audio)

        mel_spec = self._mel_fb @ power_spec

        log_mel = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
 
        return log_mel.astype(np.float32)
    
    ###########################################################################
    #                           PRIVATE METHODS                               #
    ###########################################################################

    def _validate_audio(self, audio: np.ndarray) -> None:
        """
        Sanity-checks the input audio BEFORE we try to process it, and
        raises a clear, human-readable error if something looks wrong.
        This exists because bugs here (wrong dtype, wrong shape) would
        otherwise cause confusing errors deep inside numpy, or worse,
        silently produce wrong (but not crashing) results.
        """
        # dtype check: we require float32 specifically, because that's
        # what the rest of the pipeline (and the training data) assumes.
        if audio.dtype != np.float32:
            raise TypeError(
                f"Expected np.float32 audio, got {audio.dtype}. "
                f"Cast with audio.astype(np.float32); if source is int16, "
                f"normalize first with audio.astype(np.float32) / 32768.0."
            )
 
        # shape check: we require a flat, 1-D array (just a list of
        # numbers). Audio from sounddevice often comes as 2-D
        # (n_samples, n_channels) even for mono audio, so this catches
        # that common mistake.
        if audio.ndim != 1:
            raise ValueError(
                f"Expected 1-D mono audio, got shape {audio.shape}. "
                f"If this came from sounddevice, squeeze the channel dim: "
                f"audio[:, 0]."
            )
 
        # length check: we need at least n_fft samples to compute even
        # one FFT window.
        if audio.shape[0] < self.n_fft:
            raise ValueError(
                f"Audio block has {audio.shape[0]} samples, shorter than "
                f"n_fft={self.n_fft}. Pad or accumulate a longer block first."
            )
 
        # memory-layout check: some numpy operations we use later
        # (sliding_window_view) require the array's data to be laid out
        # "contiguously" in memory. Audio sliced out of a larger buffer
        # (common with streaming audio) can sometimes fail this check.
        if not audio.flags["C_CONTIGUOUS"]:
            raise ValueError(
                "Audio array is not C-contiguous (common with sounddevice "
                "buffer slices). Fix with np.ascontiguousarray(audio)."
            )

    def _design_bandpass(self, low_hz: float, high_hz: float, order: int):
        """
        Builds the Mathematical coefficients for a Butterworth bandpass

        Args:
            low_hz (float): The lowest frequency allowed by the bandpass filter
            high_hz (float): The highest frequency allowed by the bandpass filter
            order (int): Controls how sharp the bandpass filter's cutoff is. Higher implies sharper transition but more computation
        """

        # Calculate Nyquist frequency. That is half the sample rate
        nyq = self.sr / 2.0

        return butter(
            order, 
            [low_hz / nyq, high_hz / nyq],
            btype = "band", # Type of filter (bandpass, highpass, lowpass, etc...)
            output = "sos" 
        )
    
    def _build_mel_filterbank(self, fmin: float, fmax: float) -> np.ndarray:
        """
        Builds the "mel filterbank" — a fixed matrix that we'll multiply
        against the raw frequency spectrum later. Each row of this matrix
        is a triangular "weighting curve" that picks out and blends
        together a range of nearby frequencies into a single mel band.
 
        WHY MEL BANDS? Human (and likely animal) hearing doesn't perceive
        pitch linearly — the difference between 100Hz and 200Hz sounds
        much bigger than the difference between 5000Hz and 5100Hz, even
        though both are "100Hz apart". The mel scale reshapes frequency
        to better match that perceptual spacing, which tends to help
        models learn more efficiently from audio.

        Args:
            fmin (float): Minimum allowed frequency used by bandpass filter
            fmax (float): Maximum allowed frequency used by bandpass filter
        """

        # How many raw FFT frequency bins we'll have per chunk.
        n_freq_bins = self.n_fft // 2 + 1
 
        # The actual Hz value represented by each of those raw FFT bins.
        fft_freqs = np.linspace(0, self.sr / 2.0, n_freq_bins)
 
        # Convert our Hz range into "mel" units, then lay out n_mels + 2
        # evenly-spaced points along that mel scale. The "+2" gives us
        # the extra left/right edge points needed to build triangles.
        mel_min, mel_max = self._hz_to_mel(fmin), self._hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
 
        # Convert those evenly-spaced mel points back into Hz — because
        # they're evenly spaced in MEL space, they end up unevenly
        # spaced in Hz space (closer together at low frequencies, wider
        # apart at high frequencies), which is the whole point.
        hz_points = self._mel_to_hz(mel_points)
 
        # Build one triangular filter per mel band. `fb` will end up
        # shaped (n_mels, n_freq_bins) — one row per mel band, one
        # column per raw FFT frequency bin.
        fb = np.zeros((self.n_mels, n_freq_bins), dtype=np.float32)
        for m in range(1, self.n_mels + 1):
            # Each triangle has a left edge, a peak (center), and a
            # right edge, taken from consecutive hz_points.
            f_left, f_center, f_right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
 
            # These two lines compute the "rising" and "falling" slopes
            # of the triangle across all FFT frequency bins at once.
            left_slope = (fft_freqs - f_left) / (f_center - f_left + 1e-10)
            right_slope = (f_right - fft_freqs) / (f_right - f_center + 1e-10)
 
            # The triangle's value at each frequency bin is whichever
            # slope is smaller (this creates the rise-then-fall shape),
            # clipped at 0 so it never goes negative outside the triangle.
            fb[m - 1] = np.maximum(0, np.minimum(left_slope, right_slope))
 
        return fb

    @staticmethod
    def _hz_to_mel(f):
        """Converts a frequency in Hz to the 'mel' perceptual scale."""
        # HTK mel formula
        return 2595.0 * np.log10(1.0 + f / 700.0)
 
    @staticmethod
    def _mel_to_hz(m):
        """The reverse of _hz_to_mel: converts mel units back to Hz."""
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
 
    def _stft_power(self, audio: np.ndarray) -> np.ndarray:
        """
        Computes the "Short-Time Fourier Transform" power spectrum.
 
        In plain terms: slide a small window along the audio, and at
        each position, measure how much of each frequency is present in
        that little chunk. Doing this repeatedly as we slide along in
        time gives us a 2-D grid: frequency vs. time — the raw material
        for a spectrogram, before we apply the mel filterbank.
        """
        # How many overlapping chunks ("frames") fit in this audio,
        # given the chunk size (n_fft) and how far we slide each time
        # (hop_length).
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
 
        # `sliding_window_view` is a numpy trick that creates a "view"
        # of many overlapping windows into the audio WITHOUT actually
        # copying the data for each one — this keeps it fast and
        # memory-efficient.
        frames = np.lib.stride_tricks.sliding_window_view(audio, self.n_fft)
 
        # We only want every `hop_length`-th window (not literally every
        # single one), and only up to n_frames of them.
        frames = frames[:: self.hop_length][:n_frames]
 
        # Apply the smooth "window" taper (built in __init__) to each
        # chunk, to reduce edge artifacts before the FFT.
        frames = frames * self._window
 
        # `np.fft.rfft` computes the frequency content of each chunk at
        # once (it's applied along axis=1, i.e. across each row/frame).
        # "r" in rfft means "real-input FFT" — a faster version for
        # real-valued audio (as opposed to complex numbers).
        spec = np.fft.rfft(frames, axis=1)
 
        # We care about the ENERGY (power) at each frequency, not the
        # raw complex FFT output, so we take the magnitude and square it.
        # `.T` transposes the result so frequency is the first axis and
        # time (frames) is the second — matching (n_mels, n_frames) later.
        power = (np.abs(spec) ** 2).T
 
        return power.astype(np.float32)
    


if __name__ == "__main__":
    gen = MelSpectrogramGen.from_config()
    print("Ran comfortably")