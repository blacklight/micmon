import os
import numpy as np

from .writer import DatasetWriter
from ..audio import AudioSegment


class Dataset:
    def __init__(self, samples: np.ndarray, classes: np.ndarray, validation_split: float = 0.,
                 low_freq: float = AudioSegment.default_low_freq, high_freq: float = AudioSegment.default_high_freq):
        self.samples = samples
        self.classes = classes
        self.labels = np.sort(np.unique(classes))
        self.validation_split = validation_split
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.train_samples, self.train_classes, self.validation_samples, self.validation_classes = [np.array([])] * 4
        self.shuffle()

    @classmethod
    def load(cls, npz_path: str, validation_split: float = 0.):
        dataset = np.load(os.path.abspath(os.path.expanduser(npz_path)))
        return cls(samples=dataset['samples'],
                   classes=dataset['classes'],
                   validation_split=validation_split,
                   low_freq=dataset['cutoff_frequencies'][0],
                   high_freq=dataset['cutoff_frequencies'][1])

    @classmethod
    def scan(cls, datasets_path, validation_split: float = 0.):
        datasets_path = os.path.abspath(os.path.expanduser(datasets_path))
        return [
            cls.load(os.path.join(datasets_path, file), validation_split=validation_split)
            for file in os.listdir(datasets_path)
            if os.path.isfile(os.path.join(datasets_path, file))
            and file.endswith('.npz')
        ]

    def shuffle(self):
        data = np.array([
            (self.samples[i], self.classes[i])
            for i in range(len(self.samples))
        ], dtype=object)

        np.random.shuffle(data)
        self.samples = np.array([p[0] for p in data])
        self.classes = np.array([p[1] for p in data])

        pivot = int(len(data) - (self.validation_split * len(data)))
        self.train_samples = np.array([p[0] for p in data[:pivot]])
        self.train_classes = np.array([p[1] for p in data[:pivot]])
        self.validation_samples = np.array([p[0] for p in data[pivot:]])
        self.validation_classes = np.array([p[1] for p in data[pivot:]])
