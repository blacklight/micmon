import subprocess
from typing import Optional

from micmon.audio import AudioSegment


class AudioPlayer:
    def __init__(self, sample_rate: int = 44100, channels: int = 1, ffplay_bin: str = 'ffplay'):
        self.sample_rate = sample_rate
        self.channels = channels
        self.ffplay_bin = ffplay_bin
        self.process: Optional[subprocess.Popen] = None

    def __enter__(self):
        self.process = subprocess.Popen([
            self.ffplay_bin, '-f', 's16le', '-ar', str(self.sample_rate),
            '-ac', str(self.channels), '-nodisp', '-'
        ], stdin=subprocess.PIPE)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            if self.process.poll is None:
                self.process.kill()

            self.process.wait()
            self.process = None

    def play(self, audio: AudioSegment):
        assert self.process, 'Player is not running'
        self.process.stdin.write(audio.data)
