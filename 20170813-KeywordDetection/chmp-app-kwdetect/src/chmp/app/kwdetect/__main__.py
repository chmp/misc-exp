import asyncio
import json
import logging
import os.path
import random

import click
import sounddevice as sd
import soundfile as sf
from chmp.label import write_label, find_unlabeled
from chmp.app.kwdetect.aio import detect as _async_detect
from chmp.app.kwdetect.util import load_optional_model


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
    _logger.info('load model')
    model = load_optional_model(model)

    _logger.info('enter detection loop')
    async for label in _async_detect(model, sample_target=target):
        print('detected: ', label)


@main.command()
@click.argument('path')
@click.option('--labels')
def label(path, labels):
    """Generate labels in an interactive fashion."""
    with open(labels, 'rt') as fobj:
        labels = json.load(fobj)

    label_decoding = {int(key): label for label, key in labels.items()}
    label_decoding[-1] = '<repeat>'

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
            _label_example(fname, label_decoding)

        except KeyboardInterrupt:
            print('Stop labelling ...')
            raise SystemExit(0)

    print('No more files to label :)')


def _label_example(fname, label_decoding):
    print(f'Processing: {fname}')

    sample, _ = sf.read(fname)

    while True:
        sd.play(sample, blocking=True)

        label = _get_label_from_user(label_decoding)

        if label == '<skip>':
            print('Skip sample')
            return

        elif label == '<repeat>':
            continue

        else:
            write_label(fname, label=label, file=os.path.basename(fname))
            return


def _get_label_from_user(label_decoding):
    print('Chose label:', ' '.join(f'{label!r} ({code})' for code, label in label_decoding.items()))
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
