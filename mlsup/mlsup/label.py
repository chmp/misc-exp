"""Helpers to label files.

For any data file, i.e., ``image.png``, the corresponding label files has the
same base, but the extension ``.label``, i.e., ``image.label``.
"""
import datetime
import glob
import json
import logging
import os.path

_logger = logging.getLogger(__name__)


def listdata(pattern, valid_lables=None):
    fnames = glob.glob(pattern)
    result = []

    for fname in fnames:
        if has_label(fname):
            d = dict(
                get_label(fname),
                file=os.path.abspath(fname),
            )

        else:
            d = dict(
                label='<unlabeled>',
                file=os.path.abspath(fname),
            )

        if valid_lables is None or d['label'] in valid_lables:
            result.append(d)

    return result


def get_label(fname):
    fname = get_label_fname(fname)
    with open(fname, 'rt') as fobj:
        return json.load(fobj)


def find_labelled(pattern, recursive=False):
    return _find_predicate(pattern, recursive, lambda fname: has_label(fname))


def find_unlabelled(pattern, recursive=False):
    return _find_predicate(pattern, recursive, lambda fname: not has_label(fname))


def _find_predicate(pattern, recursive, predicate):
    return [
        fname
        for fname in glob.iglob(pattern, recursive=recursive)
        if predicate(fname)
    ]


def has_label(fname):
    return os.path.exists(get_label_fname(fname))


def write_label(*args, keep_existing=True, **kwargs):
    if len(args) == 1:
        fname, = args
        d = {}

    elif len(args) == 2:
        fname, d = args
        d = dict(d)

    else:
        raise ValueError('invalid number of position arguments: {}'.format(len(args)))

    d.update(kwargs)
    label_fname = get_label_fname(fname)

    if keep_existing:
        if os.path.exists(label_fname):
            _logger.info('found existing label file %s', label_fname)
            with open(label_fname, 'rt') as fobj:
                old_label = json.load(fobj)

            d['old_label'] = old_label

    if 'timestamp' not in d:
        d['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    _logger.info('write file %s', label_fname)
    with open(label_fname, 'w') as fobj:
        json.dump(d, fobj, indent=2, sort_keys=True)


def get_label_fname(fname):
    label_fname, _ = os.path.splitext(fname)
    label_fname = label_fname + '.label'

    return label_fname
