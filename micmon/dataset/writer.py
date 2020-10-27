import os
import pathlib
import numpy as np

from micmon.audio import AudioSegment


class DatasetWriter:
    def __init__(self, path: str,
                 low_freq: int = AudioSegment.default_low_freq,
                 high_freq: int = AudioSegment.default_high_freq,
                 bins: int = AudioSegment.default_bins):
        self.path = os.path.abspath(os.path.expanduser(path))
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.bins = bins
        self.samples = []
        self.classes = []

    def __add__(self, sample: AudioSegment):
        self.samples.append(sample.spectrum(low_freq=self.low_freq, high_freq=self.high_freq, bins=self.bins))
        self.classes.append(sample.label)
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pathlib.Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path,
                            samples=np.array(self.samples),
                            classes=np.array(self.classes),
                            cutoff_frequencies=np.array([self.low_freq, self.high_freq]))

        self.samples = []
