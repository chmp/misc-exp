import fnmatch
import itertools as it
import os
import os.path
import pickle
import subprocess
import uuid

import numpy as np
import sounddevice as sd
import soundfile as sf


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


def load_optional_model(model, session):
    if model is None:
        return None

    with open(model, 'rb') as fobj:
        restorable = pickle.load(fobj)
        return restorable.restore(session=session)


# TODO: add support to expand nodes in show_graph
def show_graph(graph=None, *, exclude=frozenset(), style=None, skip_recurrence=False):
    """Show a high-level overview of a tensorflow graph.

    .. note::

        This function requires graphviz to be installed and its binaries
        to be included in the PATH.

    """
    from IPython.core.display import SVG

    src = source_show_graph(
        graph=graph,
        exclude=exclude,
        style=style,
        skip_recurrence=skip_recurrence,
    )
    image_data = subprocess.check_output(['dot', '-Tsvg'], input=src.encode('utf8'))
    return SVG(image_data)


def source_show_graph(graph=None, *, exclude=frozenset(), style=None, skip_recurrence=False):
    if graph is None:
        import tensorflow as tf
        graph = tf.get_default_graph()

    if style is None:
        style = {}

    style = {
        c: v
        for k, v in style.items()
        for c in ([k] if not isinstance(k, tuple) else k)
    }

    conns = set()
    nodes = set()

    for op in graph.get_operations():
        for i, o in it.product(op.inputs, op.outputs):
            io = get_toplevel_scope(i)
            oo = get_toplevel_scope(o)

            if skip_recurrence and io == oo:
                continue

            if any(fnmatch.fnmatch(io, pat) for pat in exclude):
                continue

            if any(fnmatch.fnmatch(oo, pat) for pat in exclude):
                continue

            conns |= {(io, oo)}
            nodes |= {io, oo}

    nodes = {n: f'node{i}' for i, n in enumerate(nodes)}

    return '\n'.join(
        ['digraph{'] +
        [f'{node} [label="{label}" {style.get(label, "")}]' for label, node in nodes.items()] +
        [f'{nodes[src]} -> {nodes[dst]}' for src, dst in conns] +
        ['}']
    )


def get_toplevel_scope(op):
    scope, *_ = op.name.partition(':')
    scope, *_ = scope.split('/')
    return scope


def as_confusion_matrix(true_column, pred_column):
    """Compute a confusion matrix from a df"""
    return lambda df: (
        df
        .groupby([true_column, pred_column])
        .size()
        .unstack()
        .fillna(0)
        .astype(int)
    )


def encoding_to_category(label_decoding, column=None):
    if column is not None:
        column = [column] if not isinstance(column, list) else column

    def _series_impl(s):
        return (
            s
            .astype('category', categories=sorted(label_decoding))
            .cat.rename_categories([label_decoding[k] for k in sorted(label_decoding)])
        )

    # NOTE: use extra function to avoid scoping issues for c
    def _col_impl(c):
        return lambda df: _series_impl(df[c])

    def _df_impl(obj):
        return obj.assign(**{c: _col_impl(c) for c in column})

    return _series_impl if column is None else _df_impl


def unique_filename(*p):
    *tail, head = p

    while True:
        fname = os.path.join(*tail, head.format(uuid.uuid4()))

        if not os.path.exists(fname):
            return fname


def fit(s, l):
    return s.ljust(l)[:l]
