import fnmatch
import importlib
import itertools as it
import os.path
import queue as _queue
import subprocess
import time
import uuid

DEFAULT_SAMPLERATE = 44100

labels = ['noise', 'wait', 'stop', 'explain']

label_encoding = {
    'noise': 0,
    'explain': 1,
    'wait': 2,
    'stop': 3,
}

label_decoding = {v: k for k, v in label_encoding.items()}


class PickableTFModel:
    __params__ = ()

    def to_pickable(self, session=None):
        if session is None:
            import tensorflow as tf
            session = tf.get_default_session()

        init_kwargs = {k: getattr(self, k) for k in self.__params__}
        variables = {v.name: v.eval(session) for v in self.variables}
        return PickableWrapper(type(self), init_kwargs, variables)


class PickableWrapper:
    def __init__(self, cls, init_kwargs, variables):
        self.cls = cls
        self.init_kwargs = init_kwargs
        self.variables = variables

    def restore(self, session):
        import tensorflow as tf

        if session is None:
            session = tf.get_default_session()

        model = self.cls(graph=session.graph, **self.init_kwargs)

        session.run(tf.global_variables_initializer())

        for v in model.variables:
            session.run(v.assign(self.variables[v.name]))

        return model


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


def caption(s, size=13, strip=True):
    """Add captions to matplotlib graphs."""
    import matplotlib.pyplot as plt

    if strip:
        s = s.splitlines()
        s = (i.strip() for i in s)
        s = (i for i in s if i)
        s = ' '.join(s)

    plt.figtext(0.5, 0, s, wrap=True, size=size, va='bottom', ha='center')


def get_color_cycle():
    import matplotlib as mpl
    return mpl.rcParams['axes.prop_cycle'].by_key()['color']


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


def iter_queue(q):
    """Helper to iterate a queue."""
    while True:
        try:
            yield q.get(block=False)

        except _queue.Empty:
            break


def reload(bootstrap=True):
    """helper for interactive development"""
    def _reload(name):
        return importlib.reload(importlib.import_module(name))

    if bootstrap:
        reload = _reload('kwdetect.util').reload
        return reload(False)

    _reload('kwdetect.util')
    _reload('kwdetect.segmentation')
    _reload('kwdetect.io')
    _reload('kwdetect.model')
    _reload('kwdetect')


def unique_filename(*p):
    *tail, head = p

    while True:
        fname = os.path.join(*tail, head.format(uuid.uuid4()))

        if not os.path.exists(fname):
            return fname


class Loop(object):
    """Helper to track the status of a long-running loops."""

    def __init__(self):
        self.idx = 0
        self.length = None
        self.start = 0
        self.current = 0
        self.expected = None

    def iterate(self, iterable):
        try:
            self.length = len(iterable)

        except TypeError:
            self.length = 1

        self.start = time.time()
        for self.idx, item in enumerate(iterable):
            yield item
            self.expected = (time.time() - self.start) / (self.idx + 1) * self.length

    @property
    def status(self):
        total = time.time() - self.start

        if self.expected is None:
            expected = total / (self.idx + 1) * self.length

        else:
            expected = self.expected

        l = ((self.idx + 1) * 10) // self.length
        bar = '#' * l + '.' * (10 - l)

        return '{} [{:.1f}s / {:.1f}s]'.format(bar, total, expected)

    @property
    def fraction(self):
        return '{} / {}'.format(self.idx + 1, self.length)

    @property
    def summary(self):
        total = time.time() - self.start
        return 'done {:.1f} s'.format(total)


def fit(s, l):
    return s.ljust(l)[:l]