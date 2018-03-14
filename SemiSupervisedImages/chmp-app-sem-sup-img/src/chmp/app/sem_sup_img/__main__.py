import argparse
import json
import logging
import pathlib

from chmp.app.sem_sup_img.data import input_fn
from chmp.app.sem_sup_img.model import Estimator, default_params


_logger = logging.getLogger(__name__)


def main(*, model_dir, data_file):
    data_file = pathlib.Path(data_file)
    model_dir = pathlib.Path(model_dir)
    config_path = model_dir / 'config.json'

    config = get_config(config_path, default_params)

    with data_file.open('rt') as fobj:
        data_fnames = json.load(fobj)

    est = Estimator(
        model_dir='./run/experiment',
        structure=config['structure'],
        latent_structure=config['latent_structure'],
    )

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


_parser = argparse.ArgumentParser()
_parser.add_argument('--data-file', required=True)
_parser.add_argument('--model-dir', required=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # if tf logger already has a handler remove it to prevent double logging
    handlers = list(logging.getLogger('tensorflow').handlers)
    for handler in handlers:
        logging.getLogger('tensorflow').removeHandler(handler)

    args = _parser.parse_args()
    main(model_dir=args.model_dir, data_file=args.data_file)
