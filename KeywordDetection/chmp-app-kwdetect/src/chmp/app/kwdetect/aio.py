"""Programmatic detection engine
"""
import logging

import janus
import numpy as np
import sounddevice as sd
import soundfile as sf

from .util import DEFAULT_SAMPLERATE, label_decoding as default_label_decoding, unique_filename
from .segmentation import StreamProcessor

_logger = logging.getLogger()


class unset:
    pass


async def detect(
        model,
        samplerate=DEFAULT_SAMPLERATE,
        label_decoding=None,
        sample_target=None,
        session=None,
        start_token=unset,
):
    import tensorflow as tf

    if label_decoding is None:
        label_decoding = default_label_decoding

    if session is None:
        session = tf.get_default_session()

    async for sample in record(samplerate=samplerate, start_token=start_token):
        if sample is start_token:
            yield sample

        else:
            label = predict_label(model, sample, session=session, label_decoding=label_decoding)
            save_sample(sample_target, sample, samplerate=samplerate)
            yield label


def predict_label(model, sample, *, session, label_decoding):
    from .model import prepare_samples

    if model is None:
        return '<no model>'

    padded, lengths = prepare_samples([sample])
    pred = model.predict({'inputs': padded, 'lengths': lengths}, session=session)
    return label_decoding[pred[0]]


def save_sample(sample_target, sample, samplerate=DEFAULT_SAMPLERATE):
    if sample_target is None:
        return

    fname = unique_filename(sample_target, '{}.ogg')
    _logger.info('export to %s ', fname)
    sf.write(fname, sample, samplerate=samplerate)


async def record(samplerate=DEFAULT_SAMPLERATE, start_token=unset):
    """An asynchronous generator yield recorded samples"""
    queue = janus.Queue()

    processor = StreamProcessor(queue=queue.sync_q, samplerate=samplerate)

    def callback(indata, _outdata, _frames, _time, _status):
        indata = np.mean(indata, axis=1)
        processor.process(indata)

    try:
        with sd.Stream(
                samplerate=samplerate,
                blocksize=(2 * samplerate) // 1,
                channels=1,
                callback=callback
        ):
            _logger.info('enter recording loop')
            if start_token is not unset:
                yield start_token

            while True:
                yield await queue.async_q.get()

    finally:
        _logger.info('exit recording loop')
        processor.finish()

