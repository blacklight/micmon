import json
from typing import Optional, List, Tuple, Union

from micmon.audio import AudioDirectory, AudioSegment, AudioSource


class AudioFile(AudioSource):
    def __init__(self, path: AudioDirectory,
                 start: Union[str, int, float] = 0,
                 duration: Optional[Union[str, int, float]] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ffmpeg_args = (
            self.ffmpeg_bin, '-i', path.audio_file, *(('-ss', str(start)) if start else ()),
            *(('-t', str(duration)) if duration else ()), *self.ffmpeg_base_args
        )

        self.start = self.convert_time(start)/1000
        self.duration = self.convert_time(duration)/1000
        self.segments = self.parse_labels_file(path.labels_file) \
            if path.labels_file else []

        self.labels = sorted(list(set(label for timestamp, label in self.segments)))
        self.cur_time = self.start
        self.cur_label = None

    @classmethod
    def parse_labels_file(cls, labels_file: str) -> List[Tuple[int, Union[int, bool, str]]]:
        with open(labels_file, 'r') as f:
            segments = {
                cls.convert_time(timestamp): label
                for timestamp, label in json.load(f).items()
            }

        return [
            (timestamp, segments[timestamp])
            for timestamp in sorted(segments.keys())
        ]

    def __next__(self) -> AudioSegment:
        if not self.ffmpeg or self.ffmpeg.poll() is not None:
            raise StopIteration

        data = self.ffmpeg.stdout.read(self.bufsize)
        if data:
            while self.segments and self.cur_time * 1000 >= self.segments[0][0]:
                self.cur_label = self.segments.pop(0)[1]

            audio = AudioSegment(data, sample_rate=self.sample_rate, channels=self.channels,
                                 label=self.labels.index(self.cur_label))

            self.cur_time += audio.duration
            return audio

        raise StopIteration

