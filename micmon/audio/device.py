from micmon.audio import AudioSource


class AudioDevice(AudioSource):
    def __init__(self, system: str = 'alsa', device: str = 'plughw:0,1', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ffmpeg_args = (
            self.ffmpeg_bin, '-f', system, '-i', device, *self.ffmpeg_base_args
        )
