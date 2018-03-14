"""Data preparation"""
import logging
import pathlib
import subprocess
import zipfile

import PIL.Image

_logger = logging.getLogger(__name__)
root_path = pathlib.Path(__file__).resolve().parent / 'data'
known_extensions = {'.mp4', '.mkv', '.avi', '.webm'}


def main():
    raw_path = root_path / 'raw'
    video_path = root_path / 'videos'
    frames_path = root_path / 'frames'

    for child in raw_path.iterdir():
        if child.suffix not in known_extensions:
            continue

        resize_video(child, video_path / (child.stem + '.mp4'))

    for child in video_path.iterdir():
        if child.suffix not in known_extensions:
            continue

        # export different sizes to compare
        try:
            encode_video(child, frames_path, get_result_id(child), [50, 80])

        except Exception as e:
            _logger.info('error encoding %s: %s', child, e, exc_info=True)


def resize_video(source_path, target_path):
    if target_path.exists():
        _logger.info('%s already exists', target_path.name)
        return

    if not target_path.parent.is_dir():
        target_path.parent.mkdir()

    _logger.info('rescale %s', source_path.name)

    subprocess.run([
        'ffmpeg',
        '-hide_banner', '-loglevel', 'panic',
        '-i', source_path, '-vf', 'scale=320:-1', target_path
    ], check=True)

    _logger.info('remove %s', source_path)
    source_path.unlink()


def encode_video(path, frames_path, result_id, sizes):
    result_paths = [
        frames_path / f'{result_id}_{size}.zip'
        for size in sizes
    ]

    if all(p.exists() for p in result_paths):
        _logger.info('%s already exists', [p.name for p in result_paths])
        return

    if not frames_path.is_dir():
        frames_path.mkdir()

    _logger.info('encode %s -> %s', path.name, sizes)
    res = subprocess.run([
        'ffmpeg',
        '-i', path,
        '-r', '0.3',  # generate 3 frames every 10 seconds
        frames_path / f'{result_id}-%05d.png',
    ], stderr=subprocess.PIPE)

    if res.returncode != 0:
        _logger.error(res.stderr.decode('utf8'))
        raise RuntimeError('Process error')

    frames = list(frames_path.glob(f'{result_id}*.png'))

    for result_path, size in zip(result_paths, sizes):
        if result_path.exists():
            _logger.info('%s already exists', result_path.name)
            continue

        _logger.info('create %s', result_path.name)
        with zipfile.ZipFile(result_path, 'w', zipfile.ZIP_STORED) as z:
            for frame in frames:
                img = PIL.Image.open(frame)
                img = img.resize((size, size), resample=PIL.Image.BICUBIC).convert('L')

                with z.open(frame.name, 'w') as fobj:
                    img.save(fobj, format='png')

    for frame in frames:
        frame.unlink()


def get_result_path(path):
    return root_path / 'data' / 'frames' / (get_result_id(path) + '.zip')


def get_result_id(path):
    return path.stem[-11:]


def get_frame_index(path):
    *_, frame_idx = path.stem.rpartition('-')
    return int(frame_idx)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
