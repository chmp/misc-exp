import time
import glob
import json
import os
import os.path

import numpy as np
import sounddevice as sd
import soundfile as sf

from .segmentation import StreamProcessor
from .util import DEFAULT_SAMPLERATE


def play_file(fname):
    data, sr = sf.read(fname)
    sd.play(data, sr, blocking=True)


def listdata(path):
    fnames = glob.glob(os.path.join(path, '*.ogg'))
    result = []

    for fname in fnames:
        label_fname = get_label_fname(fname)

        if os.path.exists(label_fname):
            with open(label_fname) as fobj:
                d = json.load(fobj)

            result.append(dict(
                d,
                file=os.path.abspath(fname),
            ))

        else:
            result.append(dict(
                label='<unlabeled>',
                file=os.path.abspath(fname),
            ))

    return result


def load_sample(fname):
    sample, _ = sf.read(fname)

    if sample.ndim == 2:
        sample = np.mean(sample, axis=1)

    return sample


def get_label_fname(fname):
    label_fname, _ = os.path.splitext(fname)
    label_fname = label_fname + '.label'

    return label_fname


def record_continuous(q, samplerate=DEFAULT_SAMPLERATE):
    """Collect audio continuously and put blocks of speech into the queue.

    :param queue.Queue q:
        a queue collecting detected blocks

    :param int samplerate:
        the desired sample rate of the results
    """
    processor = StreamProcessor(queue=q, samplerate=samplerate)

    def callback(indata, outdata, frames, time, status):
        indata = np.mean(indata, axis=1)
        processor.process(indata)

    try:
        with sd.Stream(
                samplerate=samplerate,
                blocksize=(2 * samplerate) // 1,
                channels=1,
                callback=callback
        ):
            while True:
                time.sleep(1000)

    finally:
        processor.finish()
