import logging
import os
import signal
import subprocess
from abc import ABC
from typing import Optional, Union, IO

from micmon.audio.segment import AudioSegment


class AudioSource(ABC):
    def __init__(self,
                 sample_duration: float = 2.0,
                 sample_rate: int = 44100,
                 channels: int = 1,
                 ffmpeg_bin: str = 'ffmpeg',
                 debug: bool = False):
        self.ffmpeg_bin = ffmpeg_bin
        self.ffmpeg_base_args = (
            '-f', 's16le',
            '-acodec', 'pcm_s16le', '-ac', str(channels), '-r', str(sample_rate), '-')

        self.ffmpeg_args = self.ffmpeg_base_args

        # bufsize = sample_duration * rate * width * channels
        self.bufsize = int(sample_duration * sample_rate * 2 * 1)
        self.ffmpeg: Optional[subprocess.Popen] = None
        self.sample_duration = sample_duration
        self.sample_rate = sample_rate
        self.channels = channels
        self.debug = debug
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        self.devnull: Optional[IO] = None

    def __iter__(self):
        return self

    def __next__(self) -> AudioSegment:
        if not self.ffmpeg or self.ffmpeg.poll() is not None:
            raise StopIteration

        data = self.ffmpeg.stdout.read(self.bufsize)
        if data:
            return AudioSegment(data, sample_rate=self.sample_rate, channels=self.channels)

        raise StopIteration

    def __enter__(self):
        kwargs = dict(stdout=subprocess.PIPE)
        if not self.debug:
            self.devnull = open(os.devnull, 'w')
            kwargs['stderr'] = self.devnull

        self.ffmpeg = subprocess.Popen(self.ffmpeg_args, **kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ffmpeg:
            self.ffmpeg.terminate()
            try:
                self.ffmpeg.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger.warning('FFmpeg process termination timeout')

            if self.ffmpeg.poll() is None:
                self.ffmpeg.kill()

            self.ffmpeg.wait()
            self.ffmpeg = None

        if self.devnull:
            self.devnull.close()
            self.devnull = None

    def pause(self):
        if not self.ffmpeg:
            return

        self.ffmpeg.send_signal(signal.SIGSTOP)

    def resume(self):
        if not self.ffmpeg:
            return

        self.ffmpeg.send_signal(signal.SIGCONT)

    @staticmethod
    def convert_time(t: Union[int, float, str]) -> int:
        if not isinstance(t, str):
            return int(t * 1000) if t else 0

        parts = t.split(':')
        hh = int(parts.pop(0)) if len(parts) == 3 else 0
        mm = int(parts.pop(0)) if len(parts) == 2 else 0
        parts = parts[0].split('.')
        msec = int(parts.pop()) if len(parts) > 1 else 0
        ss = int(parts[0])
        return (hh * 60 * 60 * 1000) + (mm * 60 * 1000) + (ss * 1000) + msec
