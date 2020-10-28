from typing import Optional

import numpy as np


class AudioSegment:
    default_low_freq = 20
    default_high_freq = 20000
    default_bins = 100

    def __init__(self, data: bytes, sample_rate: int = 44100, channels: int = 1, label: Optional[int] = None):
        self.data = data
        self.audio = np.frombuffer(data, dtype=np.int16)
        self.sample_rate = sample_rate
        self.channels = channels
        self.duration = len(self.audio) / (sample_rate * channels)
        self.label = label

    def fft(self, low_freq: int = default_low_freq, high_freq: int = default_high_freq) -> np.ndarray:
        return np.absolute(np.fft.rfft(self.audio))[low_freq:high_freq]

    def spectrum(self, low_freq: int = default_low_freq, high_freq: int = default_high_freq,
                 bins: int = default_bins) -> np.ndarray:
        fft = self.fft(low_freq=low_freq, high_freq=high_freq)
        bin_size = int(len(fft) / bins)
        return np.array([
            np.average(fft[i * bin_size: i * bin_size + bin_size]) / (self.duration * ((1 << 16) - 1))
            for i in range(bins)
        ])

    def plot_audio(self):
        import matplotlib.pyplot as plt
        plt.plot(self.audio)
        plt.show()

    def plot_spectrum(self, low_freq: int = default_low_freq, high_freq: int = default_high_freq,
                      bins: int = default_bins):
        import matplotlib.pyplot as plt
        spectrum = self.spectrum(low_freq=low_freq, high_freq=high_freq, bins=bins)
        plt.ylim(0, 1)
        plt.bar(range(len(spectrum)), spectrum)
        plt.show()
