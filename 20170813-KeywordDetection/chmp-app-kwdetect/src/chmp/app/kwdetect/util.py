import os
import os.path
import uuid

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch


DEFAULT_SAMPLERATE = 44100

labels = ['noise', 'wait', 'stop', 'explain', 'continue']

label_encoding = {
    'noise': 0,
    'explain': 1,
    'wait': 2,
    'stop': 3,
    'continue': 4
}

label_decoding = {v: k for k, v in label_encoding.items()}


def play_file(fname):
    data, sr = sf.read(fname)
    sd.play(data, sr, blocking=True)


def load_sample(fname):
    sample, _ = sf.read(fname)

    if sample.ndim == 2:
        sample = np.mean(sample, axis=1)

    return sample


def load_optional_model(model):
    if model is None:
        return None

    return torch.load(model)


def unique_filename(*p):
    *tail, head = p

    while True:
        fname = os.path.join(*tail, head.format(uuid.uuid4()))

        if not os.path.exists(fname):
            return fname


def fit(s, l):
    return s.ljust(l)[:l]
