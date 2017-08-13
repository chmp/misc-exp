import glob
import json
import os
import os.path
import pickle

import numpy as np
import sounddevice as sd
import soundfile as sf


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


def load_optional_model(model, session):
    if model is None:
        return None

    with open(model, 'rb') as fobj:
        restorable = pickle.load(fobj)
        return restorable.restore(session=session)
