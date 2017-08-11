import glob
import json
import logging
import queue as _queue
import os.path
import pickle
import random
import threading

import click
import sounddevice as sd
import soundfile as sf

from kwdetect.io import record_continuous, get_label_fname
from kwdetect.model import prepare_samples
from kwdetect.util import DEFAULT_SAMPLERATE, label_decoding, unique_filename

label_decoding = label_decoding.copy()
label_decoding[-1] = '<repeat>'

_logger = logging.getLogger(__name__)


@click.group()
def main():
    pass


@main.command()
@click.argument('target')
@click.option('--model')
def detect(target, model):
    """Continuously detect keywords and save extracted samples to disk."""
    queue = _queue.Queue()
    stop_event = threading.Event()
    t = threading.Thread(
        target=detect_thread,
        kwargs=dict(
            q=queue,
            ev=stop_event,
            target=target,
            model=model,
        ),
    )
    t.start()

    try:
        _logger.info('start recording loop')
        record_continuous(queue)

    finally:
        queue.put(None)
        stop_event.set()
        t.join()


def detect_thread(q, ev, target, model, samplerate=DEFAULT_SAMPLERATE):
    import tensorflow as tf

    with tf.Session().as_default() as sess:

        if model is not None:
            with open(model, 'rb') as fobj:
                model = pickle.load(fobj).restore(session=sess)

        _logger.info('start detection loop')
        while not ev.is_set():
            sample = q.get()

            if sample is None:
                continue

            if model is not None:
                padded, lengths = prepare_samples([sample])
                pred = model.predict({'inputs': padded, 'lengths': lengths}, session=sess)

                label = label_decoding[pred[0]]

            else:
                label = '<no model>'

            fname = unique_filename(target, '{}.ogg')

            _logger.info('detected as %s, export to %s ', label, fname)
            sf.write(fname, sample, samplerate=samplerate)


@main.command()
@click.argument('path')
def label(path):
    """Generate labels in an interactive fashion."""
    unlabeled_files = [
        fname
        for fname in glob.glob(os.path.join(path, '*.ogg'))
        if not os.path.exists(get_label_fname(fname))
    ]

    if not unlabeled_files:
        print('No files to label :)')
        return

    random.shuffle(unlabeled_files)

    print(f'Found {len(unlabeled_files)} unlabeled files')
    print('Start labelling ...')

    while unlabeled_files:
        try:
            fname = unlabeled_files.pop()
            _label_example(fname)

        except KeyboardInterrupt:
            print('Stop labelling ...')
            raise SystemExit(0)

    print('No more files to label :)')


def _label_example(fname):
    label_fname = get_label_fname(fname)
    print(f'Processing: {fname}')

    sample, _ = sf.read(fname)

    while True:
        sd.play(sample, blocking=True)

        label = _get_label_from_user()

        if label == '<skip>':
            print('Skip sample')
            return

        elif label == '<repeat>':
            continue

        else:
            print(f'Assign label {label} (write {label_fname})')
            with open(label_fname, 'w') as fobj:
                json.dump(dict(label=label, file=os.path.basename(fname)), fobj)
            return


def _get_label_from_user():
    print('Chose label:', ' '.join(f'{code}: {label!r}' for code, label in label_decoding.items()))
    while True:
        user_input = input('Label [empty to skip]: > ')

        if not user_input.strip():
            return '<skip>'

        try:
            user_input = int(user_input)

        except ValueError:
            print('Invalid input ...')

        else:
            if user_input not in label_decoding:
                print('Invalid input ...')
                continue

            return label_decoding[user_input]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
