micmon
======

*micmon* is a ML-powered library to detect sounds in an audio stream,
either from a file or from an audio input. The use case for its development
has been the creation of a self-built baby monitor to detect the cries
of my new born through a RaspberryPi + USB microphone, but it should be
good enough to detect any type of noise or audio if used with a well trained
model.

It works by splitting an audio stream into short segments, it calculates the
FFT and spectrum bins for each of these segments, and it uses such spectrum
data to train a model to detect the audio. It works well with sounds that are
loud enough to stand out of the background (it's good at detecting e.g. the
sound of an alarm clock, not the sound of flying mosquitto), that are long
enough compared to the size of the chunks (very short sounds will leave a
very small trace in the spectrum of an audio chunk) and, even better, if
their frequency bandwidth doesn't overlap a lot with other sounds (it's good
at detecting the cries of your baby, since his/her voice has a higher pitch
than yours, but it may not detect difference in the spectral signature of
the voice of two adult men in the same age group). It's not going to perform
very well if instead you are trying to use to detect speech - since it operates
on time-agnostic frequency data from chunks of audio it's not granular enough
for proper speech-to-text applications, and it wouldn't be robust enough to
detect differences in voice pitch, tone or accent.

Dependencies
------------

The software uses *ffmpeg* to record and decode audio - check instructions for
your OS on how to get it installed. It also requires *lame* or any other mp3
encoder to encode captured audio to mp3.

Python dependencies:

```bash
pip install numpy tensorflow keras

# Optional, for graphs
pip install matplotlib
```

Installation
------------

```bash
git clone https://github.com/BlackLight/micmon
cd micmon
python setup.py install
```

Audio capture
-------------

Once the software is installed, you can proceed with recording some audio that
will be used for training the model. First create a directory for your audio
samples dataset:

```bash
# This folder will store our audio samples
mkdir -p ~/datasets/sound-detect/audio

# This folder will store the datasets
# generated from the labelled audio samples
mkdir -p ~/datasets/sound-detect/data

# This folder will store the generated
# Tensorflow models
mkdir -p ~/models

cd ~/datasets/sound-detect/audio
```

Then create a new sub-folder for your first audio sample and start recording.
Example:

```bash
mkdir sample_1
cd sample_1
arecord -D plughw:0,1 -f cd | lame - audio.mp3
```

In the example above we are using *arecord* to record from the second channel
of the first audio device (check a list of available recording devices with
*arecord -l*) in WAV format, and we are then using the *lame* encoder to
convert the raw audio to mp3. When done with recording, just Ctrl-C the
application and your audio file will be ready.

Audio labelling
---------------

In the same directory as your sample (in the example above it will be
`~/datasets/sound-detect/audio/sample_1`) create a new file named
`labels.json`. Now open your audio file in Audacity or any audio player
and identify the audio segments that match your criteria - for example
when your baby is crying, when the alarm starts, when your neighbour
starts drilling the wall, or whatever the criteria is. `labels.json`
should contain a key-value mapping in the form of `start_time -> label`.
Example:

```json
{
  "00:00": "negative",
  "02:13": "positive",
  "04:57": "negative",
  "15:41": "positive",
  "18:24": "negative"
}
```

In the example above, all the audio segments between 00:00 and 02:12 will
be labelled as negative, all the segments between 02:13 and 04:56 as
positive, and so on.

You can now use *micmon* to generate a frequency spectrum dataset out of
your labelled audio. You can do it either through the `micmon-datagen`
script or with your own script.

### micmon-datagen

Type `micmon-datagen --help` to get a full list of the available options.
In general, `micmon-datagen` requires a directory that contains the labelled
audio samples sub-directories as input and a directory where the calculated
numpy-compressed datasets will be stored. If you want to generate the dataset
for the audio samples captured on the previous iteration then the command
will be something like this:

```bash
micmon-datagen --low 250 --high 7500 --bins 100 --sample-duration 2 --channels 1 \
    ~/datasets/sound-detect/audio  ~/models
```

The `--low` and `--high` options respectively identify the lowest and highest
frequencies that should be taken into account in the output spectrum. By default
these values are 20 Hz and 20 kHz (respectively the lowest and highest frequency
audible to a healthy and young human ear), but you can narrow down the frequency
space to only detect the frequencies that you're interested in and to remove
high-frequency harmonics that may spoil your data. A good way to estimate the
frequency space is to use e.g. Audacity or any audio equalizer to select the
segments of your audio that contain the sounds that you want to detect and
check their dominant frequencies - you definitely want those frequencies to be
included in your range.

`--bins` specifies in how many segments/buckets the frequency spectrum should
be split - 100 bins is the default value. `--sample-duration` specifies the
duration in seconds for each spectrum data point - 2 seconds is the default
value, i.e. the audio samples will be read in chunks of 2 seconds each and the
spectrum will be calculated for each of these chunks. If the sounds you want to
detect are shorter then you may want to reduce this value.

### Generate the dataset via script

The other way to generate the dataset from the audio is through the *micmon* API
itself. This option also enables you to take a peek at the audio data to better
calibrate the parameters. For example:

```python
import os

from micmon.audio import AudioDirectory, AudioPlayer, AudioFile
from micmon.dataset import DatasetWriter

basedir = os.path.expanduser('~/datasets/sound-detect')
audio_dir = os.path.join(basedir, 'audio/sample_1')
datasets_dir = os.path.join(basedir, 'data')
cutoff_frequencies = [250, 7500]

# Scan the base audio_dir for labelled audio samples
audio_dirs = AudioDirectory.scan(audio_dir)

# Play some audio samples starting from 01:00
for audio_dir in audio_dirs:
    with AudioFile(audio_dir, start='01:00', duration=5) as reader, \
            AudioPlayer() as player:
        for sample in reader:
            player.play(sample)

# Plot the audio and spectrum of the audio samples in the first 10 seconds
# of each audio file.
for audio_dir in audio_dirs:
    with AudioFile(audio_dir, start=0, duration=10) as reader:
        for sample in reader:
            sample.plot_audio()
            sample.plot_spectrum(low_freq=cutoff_frequencies[0],
                                 high_freq=cutoff_frequencies[1])

# Save the spectrum information and labels of the samples to a
# different compressed file for each audio file.
for audio_dir in audio_dirs:
    dataset_file = os.path.join(datasets_dir, os.path.basename(audio_dir.path) + '.npz')
    print(f'Processing audio sample {audio_dir.path}')

    with AudioFile(audio_dir) as reader, \
            DatasetWriter(dataset_file,
                          low_freq=cutoff_frequencies[0],
                          high_freq=cutoff_frequencies[1]) as writer:
        for sample in reader:
            writer += sample

```

Training the model
------------------

Once you have some `.npz` datasets saved under `~/datasets/sound-detect/data`, you can
use those datasets to train a Tensorflow+Keras model to classify an audio segment. A full
example is available under `examples/train.py`:

```python
import os
from keras import layers

from micmon.dataset import Dataset
from micmon.model import Model

# This is a directory that contains the saved .npz dataset files
datasets_dir = os.path.expanduser('~/datasets/sound-detect/data')

# This is the output directory where the model will be saved
model_dir = os.path.expanduser('~/models/sound-detect')

# This is the number of training epochs for each dataset sample
epochs = 2

# Load the datasets from the compressed files.
# 70% of the data points will be included in the training set,
# 30% of the data points will be included in the evaluation set
# and used to evaluate the performance of the model.
datasets = Dataset.scan(datasets_dir, validation_split=0.3)
labels = ['negative', 'positive']
freq_bins = len(datasets[0].samples[0])

# Create a network with 4 layers (one input layer, two intermediate layers and one output layer).
# The first intermediate layer in this example will have twice the number of units as the number
# of input units, while the second intermediate layer will have 75% of the number of
# input units. We also specify the names for the labels and the low and high frequency range
# used when sampling.
model = Model(
    [
        layers.Input(shape=(freq_bins,)),
        layers.Dense(int(2 * freq_bins), activation='relu'),
        layers.Dense(int(0.75 * freq_bins), activation='relu'),
        layers.Dense(len(labels), activation='softmax'),
    ],
    labels=labels,
    low_freq=datasets[0].low_freq,
    high_freq=datasets[0].high_freq
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
```

At the end of the process you should find your Tensorflow model saved under `~/models/sound-detect`.
You can use it in your scripts to classify audio samples from audio sources.

Classifying audio samples
-------------------------

One use case is to analyze an audio file and use the model to detect specific sounds. Example:

```python
import os

from micmon.audio import AudioFile
from micmon.model import Model

model_dir = os.path.expanduser('~/models/sound-detect')
model = Model.load(model_dir)
cur_seconds = 60
sample_duration = 2

with AudioFile('/path/to/some/audio.mp3',
               start=cur_seconds, duration='10:00',
               sample_duration=sample_duration) as reader:
    for sample in reader:
        prediction = model.predict(sample)
        print(f'Audio segment at {cur_seconds} seconds: {prediction}')
        cur_seconds += sample_duration
```

Another is to analyze live audio samples imported from an audio device - e.g. a USB microphone.
Example:

```python
import os

from micmon.audio import AudioDevice
from micmon.model import Model

model_dir = os.path.expanduser('~/models/sound-detect')
model = Model.load(model_dir)
audio_system = 'alsa'        # Supported: alsa and pulse
audio_device = 'plughw:1,0'  # Get list of recognized input devices with arecord -l

with AudioDevice(audio_system, device=audio_device) as source:
    for sample in source:
        source.pause()  # Pause recording while we process the frame
        prediction = model.predict(sample)
        print(prediction)
        source.resume() # Resume recording
```

You can use these two examples as blueprints to set up your own automation routines
with sound detection.
