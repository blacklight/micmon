import json
import os
import pathlib
import numpy as np

from typing import List, Optional
from keras import Sequential
from keras.layers import Layer
from keras.models import load_model, Model as _Model

from micmon.audio import AudioSegment
from micmon.dataset import Dataset


class Model:
    labels_file_name = 'labels.json'
    freq_file_name = 'freq.json'

    # noinspection PyShadowingNames
    def __init__(self, layers: Optional[List[Layer]] = None, labels: Optional[List[str]] = None,
                 model: Optional[_Model] = None,
                 optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=('accuracy',),
                 low_freq: int = AudioSegment.default_low_freq,
                 high_freq: int = AudioSegment.default_high_freq):
        assert layers or model
        self.label_names = labels
        self.cutoff_frequencies = (int(low_freq), int(high_freq))

        if layers:
            self._model = Sequential(layers)
            self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        else:
            self._model = model

    def fit(self, dataset: Dataset, *args, **kwargs):
        return self._model.fit(dataset.train_samples, dataset.train_classes, *args, **kwargs)

    def evaluate(self, dataset: Dataset, *args, **kwargs):
        return self._model.evaluate(dataset.validation_samples, dataset.validation_classes, *args, **kwargs)

    def predict(self, audio: AudioSegment):
        spectrum = audio.spectrum(low_freq=self.cutoff_frequencies[0], high_freq=self.cutoff_frequencies[1])
        output = self._model.predict(np.array([spectrum]))
        prediction = int(np.argmax(output))
        return self.label_names[prediction] if self.label_names else prediction

    def save(self, path: str, *args, **kwargs):
        path = os.path.abspath(os.path.expanduser(path))
        is_file = path.endswith('.h5') or path.endswith('.pb')
        if is_file:
            model_dir = str(pathlib.Path(path).parent)
        else:
            model_dir = path

        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
        self._model.save(path, *args, **kwargs)
        if self.label_names:
            labels_file = os.path.join(model_dir, self.labels_file_name)
            with open(labels_file, 'w') as f:
                json.dump(self.label_names, f)

        if self.cutoff_frequencies:
            freq_file = os.path.join(model_dir, self.freq_file_name)
            with open(freq_file, 'w') as f:
                json.dump(self.cutoff_frequencies, f)

    @classmethod
    def load(cls, path: str, *args, **kwargs):
        path = os.path.abspath(os.path.expanduser(path))
        is_file = path.endswith('.h5') or path.endswith('.pb')
        if is_file:
            model_dir = str(pathlib.Path(path).parent)
        else:
            model_dir = path

        model = load_model(path, *args, **kwargs)
        labels_file = os.path.join(model_dir, cls.labels_file_name)
        freq_file = os.path.join(model_dir, cls.freq_file_name)
        label_names = []
        frequencies = []

        if os.path.isfile(labels_file):
            with open(labels_file, 'r') as f:
                label_names = json.load(f)

        if os.path.isfile(freq_file):
            with open(freq_file, 'r') as f:
                frequencies = json.load(f)

        return cls(model=model, labels=label_names, low_freq=frequencies[0], high_freq=frequencies[1])
