import json
import os
import numpy as np

from typing import List, Optional, Union, Tuple
from keras import Sequential, losses, optimizers, metrics
from keras.layers import Layer
from keras.models import load_model, Model as _Model

from micmon.audio import AudioSegment
from micmon.dataset import Dataset


class Model:
    labels_file_name = 'labels.json'
    freq_file_name = 'freq.json'

    # noinspection PyShadowingNames
    def __init__(self, layers: Optional[List[Layer]] = None, labels: Optional[List[str]] = None,
                 model: Optional[_Model] = None, optimizer: Union[str, optimizers.Optimizer] = 'adam',
                 loss: Union[str, losses.Loss] = losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics: List[Union[str, metrics.Metric]] = ('accuracy',),
                 cutoff_frequencies: Tuple[int, int] = (AudioSegment.default_low_freq, AudioSegment.default_high_freq)):
        assert layers or model
        self.label_names = labels
        self.cutoff_frequencies = list(map(int, cutoff_frequencies))

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

    def save(self, model_dir: str, *args, **kwargs):
        model_dir = os.path.abspath(os.path.expanduser(model_dir))
        self._model.save(model_dir, *args, **kwargs)

        if self.label_names:
            labels_file = os.path.join(model_dir, self.labels_file_name)
            with open(labels_file, 'w') as f:
                json.dump(self.label_names, f)

        if self.cutoff_frequencies:
            freq_file = os.path.join(model_dir, self.freq_file_name)
            with open(freq_file, 'w') as f:
                json.dump(self.cutoff_frequencies, f)

    @classmethod
    def load(cls, model_dir: str, *args, **kwargs):
        model_dir = os.path.abspath(os.path.expanduser(model_dir))
        model = load_model(model_dir, *args, **kwargs)
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

        return cls(model=model, labels=label_names, cutoff_frequencies=frequencies)
