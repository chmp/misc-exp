import glob
import json
import logging
import pathlib

import click

from chmp.app.sem_sup_img.data import input_fn, list_zip_contents
from chmp.app.sem_sup_img.model import Estimator, default_params


_logger = logging.getLogger(__name__)


@click.group()
def main():
    pass


@main.command(name='update-data-file')
@click.argument('pattern')
@click.argument('file')
def update_data_file(pattern, file):
    data_files = []
    for zip_path in glob.iglob(pattern):
        for frame_fname in list_zip_contents(zip_path):
            data_files.append((str(zip_path), str(frame_fname)))

    with open(file, 'wt') as fobj:
        json.dump(data_files, fobj, indent=2, sort_keys=True)


@main.command()
@click.option('--model-dir', required=True)
@click.option('--data-file', required=True)
def run(model_dir, data_file):
    data_file = pathlib.Path(data_file)
    model_dir = pathlib.Path(model_dir)
    config_path = model_dir / 'config.json'

    config = get_config(config_path, default_params)

    _logger.info('use config %s', config)
    _logger.info('use data file %s', data_file)

    with data_file.open('rt') as fobj:
        data_fnames = json.load(fobj)

    est = Estimator.from_config(config, model_dir=model_dir)

    _logger.info('run until %s steps', config['max_steps'])
    est.train(
        lambda: input_fn(data_fnames, batch_size=config['batch_size']),
        max_steps=config['max_steps'],
    )


def get_config(path, default_params):
    path = pathlib.Path(path)

    if path.exists():
        _logger.info('load config from %s', path)
        with path.open('rt') as fobj:
            config = json.load(fobj)

    else:
        config = {}

    config = dict(default_params, **config)

    _logger.info('write config to %s', path)
    with path.open('wt') as fobj:
        json.dump(config, fobj, indent=2, sort_keys=True)

    return config



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # if tf logger already has a handler remove it to prevent double logging
    handlers = list(logging.getLogger('tensorflow').handlers)
    for handler in handlers:
        logging.getLogger('tensorflow').removeHandler(handler)

    main()
