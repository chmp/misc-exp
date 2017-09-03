"""Helpers to label files.

For any data file, i.e., ``image.png``, the corresponding label files has the
same base, but the extension ``.label``, i.e., ``image.label``.

IPython widgtes for annotating data sets:

* :class:`TextAnnotator`
* :class:`ImageAnnotator`
* :class:`AudioAnnotator`
* :class:`Annotator`

"""
import base64
import collections.abc
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


def find_labeled(pattern, recursive=False):
    return _find_predicate(pattern, recursive, lambda fname: has_label(fname))


def find_unlabeled(pattern, recursive=False):
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


def write_latest_labels(annotator, skip_class='skip', label_key='label', fname_key='item'):
    """Write the latest labels added to an annotator.
    """
    def unpack(*dict_and_keys):
        d, *keys = dict_and_keys
        return tuple(d[k] for k in keys)

    for item in annotator.get_latest():
        fname = item[fname_key]
        label = item[label_key]

        if label == skip_class:
            continue

        kwargs = dict(item, fname=os.path.basename(fname), label=label)
        write_label(fname, **kwargs)


class Annotator:
    """IPython widget to quickly annotate data sets.
    """
    def __init__(self, classes, history_length=10, context_size=1):
        self.current_annotations = []
        self.current_idx = None
        self.current_reason = None
        self.annotations = []
        self.data = []
        self.order = []
        self.history_length = int(history_length)
        self.context_size = int(context_size)

        self._build(classes)

    def annotate(self, key, data, order=None):
        if order is None:
            self.order = list(range(0, len(data)))

        else:
            self.order = list(order)

        self.current_annotations = []
        self.data = data
        self.annotations.append((key, self.current_annotations))

        self._next()

    def get_latest(self, index=-1):
        if self.annotations:
            _, data = self.annotations[index]

        else:
            data = []

        result = []
        added = set()

        for item in reversed(data):
            if item['index'] in added:
                continue

            added.add(item['index'])
            result.append(item)

        return result

    def build_display_value(self):
        """Build the display of the item being annotated.

        This class has to be overwritten in subclasses.

        :returns:
            an HTML reprensetation of the item to display.

        """
        raise NotImplementedError()

    def _build(self, classes):
        from ipywidgets import HTML, VBox, Layout, Box

        self._html = HTML(value='No data to annotate')
        self._history = VBox(layout=Layout(margin='1em 0em 0em 0em'))

        self._widget = VBox([
            self._html,
            Box([
                self._build_label_button(label)
                for label in classes
            ], layout=Layout(flex_flow='row wrap')),
            self._history,
        ])

    def _build_label_button(self, label):
        from ipywidgets import Button

        if not isinstance(label, collections.abc.Mapping):
            label = {'label': label, 'style': ''}

        else:
            label = dict(label)
            label.setdefault('style', '')

        b = Button(description=label['label'], button_style=label['style'])
        b.on_click(lambda b: self._annotate(label))
        return b

    def _annotate(self, label):
        self.current_annotations.append(dict(
            index=self.current_idx,
            reason=self.current_reason,
            label=label['label'],
            item=self.data[self.current_idx],
        ))

        if self.current_idx is not None:
            self._add_repeat(self.current_idx, label)

        self._next()

    def _next(self, idx=None, reason='order'):
        if idx is None:
            if not self.order:
                self._html.value = 'No data to annotate'
                self.current_idx = None
                return

            self.current_idx = self.order.pop(0)

        else:
            self.current_idx = idx

        self.current_reason = reason
        self._html.value = self.build_display_value()

    def _add_repeat(self, idx, label):
        from ipywidgets import Button, Layout

        repeat_button = Button(
            description=f'{label["label"]} - {self.data[idx]}',
            layout=Layout(width='50%'),
            button_style=label['style'],
        )
        repeat_button.on_click(lambda b: self._repeat(idx))

        history = [repeat_button] + list(self._history.children)
        history = history[:self.history_length]
        self._history.children = history

    def _repeat(self, idx):
        self.order.insert(0, self.current_idx)
        self._next(idx, reason='repeat')

    def _ipython_display_(self, **kwargs):
        return self._widget._ipython_display_(**kwargs)


class ImageAnnotator(Annotator):
    """IPython widget to annotate image files.
    """
    def build_display_value(self):
        fname = self.data[self.current_idx]
        return '<img src="{url}"/>'.format(url=build_data_url(fname))


class AudioAnnotator(Annotator):
    """IPython widget to annotate audio files.
    """
    def build_display_value(self):
        fname = self.data[self.current_idx]
        return '<audio src="{url}" controls autoplay/>'.format(url=build_data_url(fname))


class TextAnnotator(Annotator):
    """IPython widget to annotate a text document line by line.

    Usage::

        classes = [
            {'label': 'Skip', 'style': 'primary'},
            {'label': 'Pose', 'style': 'success'},
            {'label': 'Background', 'style': 'danger'},
        ]
        annotator = TextAnnotator(classes)
        annotator.annotate('my-key', lines)

        display(annotator)


    To limit the number or change the order of the lines to display, pass the
    order argument to ``annotate``::

        annotator.annotate('my-key', lines, order=[10, 5, 6])

    """
    def build_display_value(self):
        result = []
        for i in range(self.current_idx - self.context_size, self.current_idx + self.context_size + 1):
            if i < 0 or i >= len(self.data):
                l = '&nbsp;'

            else:
                l = self.data[i]

            if i == self.current_idx:
                result += [f'<span style="color:red">{l}</span>']

            else:
                result += [l]

        return '<pre>' + '\n'.join(result) + '</pre>'


def build_data_url(fname):
    _, ext = os.path.splitext(fname)
    ext = ext.lower()

    try:
        mime_type = _mime_types[ext]

    except AttributeError:
        raise ValueError('unknown file extension {}'.format(ext))

    with open(fname, 'rb') as fobj:
        data = fobj.read()

    data = base64.b64encode(data)
    data = data.decode('ascii')

    return 'data:{mime_type};base64,{data}'.format(mime_type=mime_type, data=data)


_mime_types = {
    '.png': 'image/png',
    '.ogg': 'audio/ogg',
}
