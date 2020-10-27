import os


class AudioDirectory:
    _audio_file_name = 'audio.mp3'
    _labels_file_name = 'labels.json'

    def __init__(self, path: str):
        self.path = os.path.abspath(os.path.expanduser(path))
        self.audio_file = os.path.join(self.path, self._audio_file_name)
        self.labels_file = os.path.join(self.path, self._labels_file_name)
        assert os.path.isfile(self.audio_file) and os.path.isfile(self.audio_file), \
            f'{self._audio_file_name} or {self._labels_file_name} missing from {self.path}'

    @classmethod
    def scan(cls, path: str) -> list:
        path = os.path.abspath(os.path.expanduser(path))
        return [
            cls(os.path.join(path, d))
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
            and os.path.isfile(os.path.join(path, d, cls._audio_file_name))
            and os.path.isfile(os.path.join(path, d, cls._labels_file_name))
        ]
