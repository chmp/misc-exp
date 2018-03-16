"""Data loading"""
import logging
import random
import zipfile

import numpy as np
import tensorflow as tf
import PIL.Image

_logger = logging.getLogger(__name__)


def get_data_fnames(root_path, pattern='*_50.zip'):
    return [
        (video_fname, frame_fname)
        for video_fname in root_path.glob(pattern)
        for frame_fname in list_zip_contents(video_fname)
    ]


def list_zip_contents(fname):
    with zipfile.ZipFile(fname, 'r') as z:
        return z.namelist()


def open_image(archive_fname, fname):
    with zipfile.ZipFile(archive_fname, 'r') as z:
        with z.open(fname) as fobj:
            return PIL.Image.open(fobj)


def input_fn(data_fnames, batch_size, n_epochs=None, intensity_threshold=75, shuffle=True):
    generate, types, shapes = make_input_generator(
        data_fnames, batch_size,
        intensity_threshold=intensity_threshold,
        shuffle=shuffle,
    )
    ds = tf.data.Dataset.from_generator(generate, types, shapes)
    ds = ds.repeat(n_epochs)
    return ds


# TODO: switch to text-line dataset, for reproducibility and to remove limitations of from_generator
# NOTE: still has problem of requiring py_func ...
def make_input_generator(data_fnames, batch_size, intensity_threshold=75, shuffle=True):
    """Create a generator compatible with :class:`tensorflow.data.Dataset`"""
    # Explicitly cache the data_fnames. Following TF docs
    data_fnames = list(data_fnames)
    num_batches = len(data_fnames) // batch_size

    def generate():
        if shuffle:
            random.shuffle(data_fnames)
        batch = np.empty((batch_size, 50, 50, 1))

        for i in range(num_batches):
            start = batch_size * i
            end = batch_size * (i + 1)

            for j, (video_fname, frame_fname) in enumerate(data_fnames[start:end]):
                img = open_image(video_fname, frame_fname)

                for _ in range(10):
                    if np.mean(img) >= intensity_threshold:
                        break

                    video_fname, frame_fname = random.choice(data_fnames)
                    img = open_image(video_fname, frame_fname)

                else:
                    _logger.error('could not find replacement image for %s / %s', video_fname, frame_fname)

                batch[j, ..., 0] = np.asarray(open_image(video_fname, frame_fname)) / 128 - 1.0

            yield {'input': batch}, {}

    types = ({'input': tf.float32}, {})
    shapes = ({'input': [batch_size, 50, 50, 1]}, {})

    return generate, types, shapes