import os

from micmon.audio import AudioFile
from micmon.model import Model

model_dir = os.path.expanduser('~/models/sound-detect')
model = Model.load(model_dir)
cur_seconds = 60
sample_duration = 2

with AudioFile('/path/to/some/audio.mp3', start=cur_seconds, duration='10:00',
               sample_duration=sample_duration) as reader:
    for sample in reader:
        prediction = model.predict(sample)
        print(f'Audio segment at {cur_seconds} seconds: {prediction}')
        cur_seconds += sample_duration
