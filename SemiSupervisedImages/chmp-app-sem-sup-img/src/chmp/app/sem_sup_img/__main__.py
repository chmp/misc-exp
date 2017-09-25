import functools as ft
import glob
import json
import logging
import math
import os.path
import statistics
import typing

import imageio
import numpy as np
import tensorflow as tf

from chmp import experiment
from chmp.app.sem_sup_img.model import ConvolutionalAutoEncoder

_logger = logging.getLogger(__name__)


@typing.no_type_check
class Config(experiment.Config):
    version: int = 0
    data_dir: str
    seed: str = '42'

    model: ConvolutionalAutoEncoder.Config


@experiment.experiment(Config)
def train_model(path, config):
    experiment.write(config, (path, 'config.json'))
    _logger.info('run in %s', path)

    model = ConvolutionalAutoEncoder(config=config.model)
    train_fnames, test_fnames = _get_files(config.seed, config.data_dir)

    _logger.info(f'found {len(train_fnames)} training examples, {len(test_fnames)} testing examples')
    with tf.Session() as session:
        _do_train(
            path=path,
            session=session,
            model=model,
            config=config,
            train_fnames=train_fnames,
            test_fnames=test_fnames,
        )


def _get_files(seed, data_dir):
    fnames = glob.glob(os.path.join(data_dir, '*.png'))
    fnames = sorted(fnames)
    fnames = experiment.shuffled(seed, fnames)

    split = int(0.95 * len(fnames))

    train_fnames = fnames[:split]
    test_fnames = fnames[split:]

    return train_fnames, test_fnames


def _do_train(*, path, session, config, model, train_fnames, test_fnames):
    prepare_test_batch = ft.partial(prepare_batch, fnames=test_fnames, n_batch_size=50, width=config.model.width)
    prepare_train_batch = ft.partial(
        prepare_batch, fnames=train_fnames, n_batch_size=config.model.batch_size, width=config.model.width,
    )

    n_batches = len(train_fnames) // config.model.batch_size

    losses = []
    iteration = 0
    loop = experiment.Loop()

    saver = tf.train.Saver()
    session.run(tf.global_variables_initializer())

    test_loss = math.nan
    for epoch in range(config.model.epochs):
        _logger.info(f'epoch {epoch}')
        experiment.shuffle((config.seed, epoch), train_fnames)

        for batch in loop(range(n_batches)):
            loss = model.fit_partial(prepare_train_batch(batch), session=session)

            if iteration % 500 == 0:
                test_loss = statistics.mean([
                    float(model.eval_loss(prepare_test_batch(i), session=session))
                    for i in range(len(test_fnames) // 50)
                ])

                losses.append((epoch, batch, float(loss), test_loss))

            else:
                losses.append((epoch, batch, float(loss), math.nan))

            if iteration and iteration % 500 == 0:
                saver.save(session, os.path.join(path, 'state'), global_step=iteration)

            if iteration and iteration % 50 == 0:
                experiment.write(losses, (path, 'loss.json'))

            print(
                f' {loop} {batch}@{batch / n_batches:.2%} loss: {loss:.5f} / {test_loss:.5f}'.ljust(120),
                end='\r',
            )
            iteration += 1
        print()


def prepare_batch(idx, fnames, n_batch_size, width):
    batch_data = np.zeros((n_batch_size, width, width, 1), dtype=np.float32)

    for i, fname in enumerate(fnames[idx * n_batch_size:(idx + 1) * n_batch_size]):
        batch_data[i, ..., 0] = imageio.imread(fname, format='png') / 256

    return batch_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model.main()
