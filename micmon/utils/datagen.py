import argparse
import logging
import os
import sys

from micmon.audio import AudioDirectory, AudioFile, AudioSegment
from micmon.dataset import DatasetWriter

logger = logging.getLogger(__name__)
defaults = {
    'sample_duration': 2.0,
    'sample_rate': 44100,
    'channels': 1,
    'ffmpeg_bin': 'ffmpeg',
}


def create_dataset(audio_dir: str, dataset_dir: str,
                   low_freq: int = AudioSegment.default_low_freq,
                   high_freq: int = AudioSegment.default_high_freq,
                   bins: int = AudioSegment.default_bins,
                   sample_duration: float = defaults['sample_duration'],
                   sample_rate: int = defaults['sample_rate'],
                   channels: int = defaults['channels'],
                   ffmpeg_bin: str = defaults['ffmpeg_bin']):
    audio_dir = os.path.abspath(os.path.expanduser(audio_dir))
    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    audio_dirs = AudioDirectory.scan(audio_dir)

    for audio_dir in audio_dirs:
        dataset_file = os.path.join(dataset_dir, os.path.basename(audio_dir.path) + '.npz')
        logger.info(f'Processing audio sample {audio_dir.path}')

        with AudioFile(audio_dir.audio_file, audio_dir.labels_file,
                       sample_duration=sample_duration, sample_rate=sample_rate, channels=channels,
                       ffmpeg_bin=os.path.expanduser(ffmpeg_bin)) as reader, \
                DatasetWriter(dataset_file, low_freq=low_freq, high_freq=high_freq, bins=bins) as writer:
            for sample in reader:
                writer += sample


def main():
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        description='''
Tool to create numpy dataset files with audio spectrum data from a set of labelled raw audio files.''',

        epilog='''
- audio_dir should contain a list of sub-directories, each of which represents a labelled audio sample.
  audio_dir should have the following structure:

  audio_dir/
    -> train_sample_1
      -> audio.mp3
      -> labels.json
    -> train_sample_2
      -> audio.mp3
      -> labels.json
  ...

- labels.json is a key-value JSON file that contains the labels for each audio segment. Example:

   {
     "00:00": "negative",
     "02:13": "positive",
     "04:57": "negative",
     "15:41": "positive",
     "18:24": "negative"
   }

  Each entry indicates that all the audio samples between the specified timestamp and the next entry or
  the end of the audio file should be applied the specified label.

- dataset_dir is the directory where the generated labelled spectrum dataset in .npz format will be saved.
  Each dataset file will be named like its associated audio samples directory.''',

        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('audio_dir', help='Directory containing the raw audio samples directories to be scanned.')
    parser.add_argument('dataset_dir', help='Destination directory for the compressed .npz files containing the '
                                            'frequency spectrum datasets.')
    parser.add_argument('--low', help='Specify the lowest frequency to be considered in the generated frequency '
                                      'spectrum. Default: 20 Hz (lowest possible frequency audible to a human ear).',
                        required=False, default=AudioSegment.default_low_freq, dest='low_freq', type=int)

    parser.add_argument('--high', help='Specify the highest frequency to be considered in the generated frequency '
                                       'spectrum. Default: 20 kHz (highest possible frequency audible to a human ear).',
                        required=False, default=AudioSegment.default_high_freq, dest='high_freq', type=int)

    parser.add_argument('-b', '--bins', help=f'Specify the number of frequency bins to be used for the spectrum '
                                             f'analysis (default: {AudioSegment.default_bins})',
                        required=False, default=AudioSegment.default_bins, dest='bins', type=int)

    parser.add_argument('-d', '--sample-duration', help=f'The script will calculate the spectrum of audio segments of '
                                                        f'this specified length in seconds (default: '
                                                        f'{defaults["sample_duration"]}).',
                        required=False, default=defaults['sample_duration'], dest='sample_duration', type=float)

    parser.add_argument('-r', '--sample-rate', help=f'Audio sample rate (default: {defaults["sample_rate"]} Hz)',
                        required=False, default=defaults['sample_rate'], dest='sample_rate', type=int)

    parser.add_argument('-c', '--channels', help=f'Number of destination audio channels (default: '
                                                 f'{defaults["channels"]})',
                        required=False, default=defaults['channels'], dest='channels', type=int)

    parser.add_argument('--ffmpeg', help=f'Absolute path to the ffmpeg executable (default: {defaults["ffmpeg_bin"]})',
                        required=False, default=defaults['ffmpeg_bin'], dest='ffmpeg_bin', type=str)

    opts, args = parser.parse_known_args(sys.argv[1:])
    return create_dataset(audio_dir=opts.audio_dir, dataset_dir=opts.dataset_dir, low_freq=opts.low_freq,
                          high_freq=opts.high_freq, bins=opts.bins, sample_duration=opts.sample_duration,
                          sample_rate=opts.sample_rate, channels=opts.channels, ffmpeg_bin=opts.ffmpeg_bin)


if __name__ == '__main__':
    main()
