# This script shows how to train a neural network to detect sounds given a training set of collected frequency
# spectrum data.

import os
from keras import layers

from micmon.dataset import Dataset
from micmon.model import Model

# This is a directory that contains the saved .npz dataset files
datasets_dir = os.path.expanduser('~/datasets/baby-monitor/datasets')

# This is the output directory where the model will be saved
model_dir = os.path.expanduser(os.path.join('~', 'models', 'baby-monitor'))

# This is the number of training epochs for each dataset sample
epochs = 2

# This value establishes the share of the dataset to be used for cross-validation
validation_split = 0.3

# Load the datasets from the compressed files
datasets = Dataset.scan(datasets_dir, validation_split=0.3)

# Get the number of frequency bins
freq_bins = len(datasets[0].samples[0])

# Create a network with 4 layers (one input layer, two intermediate layers and one output layer).
# The first intermediate layer in this example will have twice the number of units as the number
# of input units, while the second intermediate layer will have as many units as the number of
# input units. We also specify the names for the labels and the low and high frequency range
# used when sampling.
model = Model(
    [
        layers.Input(shape=(freq_bins,)),
        layers.Dense(int(2.0 * freq_bins), activation='relu'),
        layers.Dense(int(freq_bins), activation='relu'),
        layers.Dense(len(datasets[0].labels), activation='softmax'),
    ],
    labels=['negative', 'positive'],
    low_freq=datasets[0].low_freq,
    high_freq=datasets[0].high_freq,
)

# Train the model
for epoch in range(epochs):
    for i, dataset in enumerate(datasets):
        print(f'[epoch {epoch+1}/{epochs}] [audio sample {i+1}/{len(datasets)}]')
        model.fit(dataset)
        evaluation = model.evaluate(dataset)
        print(f'Validation set loss and accuracy: {evaluation}')

# Save the model
model.save(model_dir, overwrite=True)
