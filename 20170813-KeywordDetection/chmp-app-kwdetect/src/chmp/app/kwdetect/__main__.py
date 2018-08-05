import asyncio
import logging
import os.path
import random

import click
import sounddevice as sd
import soundfile as sf
from chmp.label import write_label, find_unlabeled
from chmp.app.kwdetect.aio import detect as _async_detect
from chmp.app.kwdetect.util import label_decoding, load_optional_model

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
    loop = asyncio.get_event_loop()

    # TODO: add better exception handler
    loop.set_exception_handler(print)
    loop.run_until_complete(_detect(target, model))


async def _detect(target, model):
    import tensorflow as tf

    with tf.Session().as_default() as sess:
        _logger.info('load model')
        model = load_optional_model(model, session=sess)

        _logger.info('enter detection loop')
        async for label in _async_detect(model, sample_target=target, session=sess):
            print('detected: ', label)


@main.command()
@click.argument('path')
def label(path):
    """Generate labels in an interactive fashion."""
    unlabeled_files = find_unlabeled(os.path.join(path, '*.ogg'))

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
            write_label(fname, label=label, file=os.path.basename(fname))
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
