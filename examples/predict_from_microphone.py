import os

from micmon.audio import AudioDevice
from micmon.model import Model

# Path to a previously saved sound detection Tensorflow model
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
