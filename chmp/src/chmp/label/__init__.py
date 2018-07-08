"""Helpers to label files.

For any data file, i.e., ``image.png``, the corresponding label files has the
same base, but the extension ``.label``, i.e., ``image.label``.

IPython widgtes for annotating data sets:

* :class:`ImageAnnotator`
* :class:`AudioAnnotator`
* :class:`Annotator`

Distributed as part of ``https://github.com/chmp/misc-exp`` under the MIT
license, (c) 2017 Christopher Prohm.
"""
import base64
import collections.abc
import datetime
import glob
import html
import json
import logging
import os.path

from ipywidgets import DOMWidget, Button, Text, HBox, VBox, jslink
from IPython.display import display_javascript, Javascript
from traitlets import Unicode, Float

_logger = logging.getLogger(__name__)


def annotate(
        items, *,
        classes,
        history_length=5,
        display_value=None,
        cls=None,
        kwargs=None
):
    """Annotate data inside the ipython notebook.

    This function constructs an IPython widget and displays it to the user. The
    returned list will be filled with the labels as the user interacts with the
    widget.

    :param Sequence[Any] items:
        the collection of items to label. The interpretation of the items
        depends on the :class:`Annotator` chosen by the ``cls`` argument.  For
        image and audio, each item should be a filename. For text, the items
        should be the lines of the text. For custom ``display_value`` each item
        is passed as is.

    :param Sequence[str] classes:
        the classes to choose from.

    :param int history_length:
        the number of previously labeled items to show for changes in case of
        errors.

    :param Optional[Callable[str,Any]] display_value:
        if given, a callable that accepts an item and returns a HTML
        representation to show to the user.

    :param Optional[Union[str,class]] cls:
        the type of annotator to use. Can be either a class directly or one
        of ``'image'``, ``'audio'``. If not given, but ``display_value`` is
        given, it will be used to display the result to the user. If neither
        one is given, the ``repr`` will be shown to the user.

    :param Optional[Mapping[Str,Any]] kwargs:
        if given, additional keyword arguments passed on constucting the
        annotator object. Note, ``history_length`` and ``display_value``
        are set by the parameters of this function.

    :returns:
        a list that is filled with feedback supplied by the user. In case of
        corrections both the old and the new label will be returned. With the
        new label having a higher index. To only retain the latest labels, use
        the additional ``get_latest`` method on the returned object.
    """
    from IPython.core.display import display

    kwargs = {} if kwargs is None else dict(kwargs)

    kwargs.update(history_length=history_length)
    if display_value is not None:
        kwargs.update(display_value=display_value)

    if cls == 'audio':
        annotator = AudioAnnotator(classes, **kwargs)

    elif cls == 'image':
        annotator = ImageAnnotator(classes, **kwargs)

    elif cls is not None:
        annotator = cls(classes, **kwargs)

    elif display_value is not None:
        annotator = FunctionalAnnotator(classes, **kwargs)

    else:
        annotator = FunctionalAnnotator(
            classes,
            lambda item: '<pre>{}</pre>'.format(html.escape(repr(item))),
        )

    annotator.annotate(items)
    display(annotator)

    return annotator.annotations


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


class BaseAnnotator:
    """Abstract annotator without ties to IPython.
    """
    def __init__(self):
        self.data = None
        self.annotations = None
        self.current_item = None

    def get_latest(self):
        if self.annotations is None:
            return []

        return self.annotations.get_latest()

    def update_display(self):
        """Hook that is called, when new information should be shown.

        The default does nothing, to add custom behavior overwrite it in a
        subclass.
        """
        pass

    def clear(self):
        self.annotations = None
        self.data = None
        self.current_item = None

    def annotate(self, data):
        if self.annotations is not None:
            raise RuntimeError('call clear before annotating again')

        self.annotations = Annotations()
        self.data = [('order', idx, item) for idx, item in enumerate(data)]
        self.next()

    def next(self, item=None):
        if item is not None:
            self.current_item = item

        elif self.data:
            self.current_item = self.data.pop(0)

        else:
            self.current_item = None
            self.update_display()
            return

        self.update_display()

    def annotate_current(self, label):
        if self.annotations is None or self.current_item is None:
            return

        reason, index, item = self.current_item
        self.annotations.append(dict(index=index, reason=reason, item=item, label=label))
        self.next()

    def repeat(self, idx):
        if idx < 0 or idx >= len(self.annotations):
            return

        if self.current_item is not None:
            self.data.insert(0, self.current_item)

        index, item = self.annotations[idx]['index'], self.annotations[idx]['item']
        item = 'repeat', index, item

        self.next(item)


class Annotator(BaseAnnotator):
    """IPython widget to quickly annotate data sets.
    """
    def __init__(self, classes, history_length=10):
        super().__init__()
        self.history_length = int(history_length)
        self.last_repeat = 0
        self._build(classes)

    def build_display_value(self, item):
        """Build the display of the item being annotated.

        This class has to be overwritten in subclasses.

        :returns:
            an HTML reprensetation of the item to display.

        """
        return html.escape(repr(item))

    def clear(self):
        super().clear()
        self.last_repeat = 0

    def update_display(self):
        if self.current_item is None:
            self._html.value = 'No data to annotate'
            self._info.value = '&nbsp;'

        else:
            reason, index, item = self.current_item
            self._html.value = self.build_display_value(item)
            self._info.value = html.escape('index: {}, reason: {}'.format(index, reason))

        if len(self.annotations) > self.last_repeat:
            from ipywidgets import Button, Layout

            repeats = list(self._history.children)

            for idx in range(self.last_repeat, len(self.annotations)):
                annotation = self.annotations[idx]

                repeat_button = Button(
                    description=f'{annotation["label"]} - {annotation["item"]!r}',
                    layout=Layout(width='50%'),
                )
                repeat_button.on_click(lambda b: self.repeat(idx))

                repeats = [repeat_button] + repeats

            self.last_repeat = len(self.annotations)

            repeats = repeats[:self.history_length]
            self._history.children = repeats

    def _build(self, classes):
        from ipywidgets import HTML, VBox, Layout, Box

        self._html = HTML(value='No data to annotate')
        self._info = HTML(value='&nbsp;')
        self._history = VBox(layout=Layout(margin='1em 0em 0em 0em'))

        self._widget = VBox([
            self._html,
            self._info,
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
        b.on_click(lambda b: self.annotate_current(label['label']))
        return b

    def _ipython_display_(self, **kwargs):
        return self._widget._ipython_display_(**kwargs)


class ImageAnnotator(Annotator):
    """IPython widget to annotate image files.

    The widget expects a list of filenames.
    """
    def build_display_value(self, item):
        return '<img src="{url}"/>'.format(url=build_data_url(item))


class AudioAnnotator(Annotator):
    """IPython widget to annotate audio files.

    The widget expects a list of filenames.
    """
    def build_display_value(self, item):
        return '<audio src="{url}" controls autoplay/>'.format(url=build_data_url(item))


class FunctionalAnnotator(Annotator):
    def __init__(self, classes, display_value, *, history_length=5, kwargs=None):
        if kwargs is None:
            kwargs = {}

        super().__init__(classes=classes, history_length=history_length)
        self.display_value = display_value
        self.kwargs = kwargs

    def build_display_value(self, item):
        return self.display_value(item, **self.kwargs)


class Annotations(list):
    def get_latest(self):
        result = []
        added = set()

        for item in reversed(self):
            if item['index'] in added:
                continue

            added.add(item['index'])
            result.append(item)

        return result


def build_data_url(fname, mime_type=None):
    """Helper to build a data-url with data read from a file.

    :param str fname:
        the file to load the content from

    :param Optional[str] mime_type:
        if given the mime type to use, otherwise it's auto-detected from the
        file extension.

    :returns:
        the data url.
    """
    _, ext = os.path.splitext(fname)
    ext = ext.lower()

    if mime_type is None:
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
    '.gif': 'image.gif',
    '.jpg': 'image/jpeg',
    '.mp3': 'audio/mpeg',
    '.ogg': 'audio/ogg',
    '.png': 'image/png',
    '.wav': 'audio/wav',
}


# ########################################################################## #
#                                                                            #
#                                 Bounding Boxes                             #
#                                                                            #
# ########################################################################## #
def annotate_bounding_boxes(images):
    annotations = []
    position = 0

    bounding_boxer = BoundingBoxer()
    annotation_display = AnnotationDisplay()
    label = Text(placeholder='label')
    prev_button = Button(description='prev')
    next_button = Button(description='next')

    widget = VBox([
        HBox([label, prev_button, next_button]),
        bounding_boxer,
        annotation_display,
    ])

    def show():
        if bounding_boxer.url:
            annotations.append((bounding_boxer.url, bounding_boxer.annotations))

        if position < len(images):
            url = images[position]

        else:
            url = ''

        if callable(url):
            url = url()
        
        bounding_boxer.url = url
        bounding_boxer.annotations = []

    def _advance(delta):
        nonlocal position
        # NOTE: allow to walk out of the images
        position = max(0, min(len(images), position + delta))
        show()

    show()
    jslink((bounding_boxer, '_annotations'), (annotation_display, '_annotations'))
    jslink((bounding_boxer, 'current_tag'), (label, 'value'))

    @bounding_boxer.on_msg
    def _(widget, ev, _):
        if ev['type'] == 'keydown' and ev['which'] == 32:
            _advance(+1)

    prev_button.on_click(lambda *_: _advance(-1))
    next_button.on_click(lambda *_: _advance(+1))

    return widget, annotations


class BoundingBoxer(DOMWidget):
    """Allow to annotate an image with bounding boxes."""
    _view_name = Unicode('BoundingBoxer').tag(sync=True)
    _view_module = Unicode('de/cprohm/label/boundingBoxer').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)

    url = Unicode(None, allow_none=True).tag(sync=True)
    current_tag = Unicode().tag(sync=True)
    scale = Float(1.0).tag(sync=True)

    _annotations = Unicode('[]').tag(sync=True)

    def __init__(self):
        super().__init__()
        inject_bounding_boxer_js()

    @property
    def annotations(self):
        return json.loads(self._annotations)

    @annotations.setter
    def annotations(self, value):
        self._annotations = json.dumps(value)


class AnnotationDisplay(DOMWidget):
    """Widget to show and manipulate bounding box annotations."""
    _view_name = Unicode('Annotations').tag(sync=True)
    _view_module = Unicode('de/cprohm/label/boundingBoxer').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)

    _annotations = Unicode('[]').tag(sync=True)

    def __init__(self):
        super().__init__()
        inject_bounding_boxer_js()

    @property
    def annotations(self):
        return json.loads(self._annotations)

    @annotations.setter
    def annotations(self, value):
        self._annotations = json.dumps(value)


def inject_bounding_boxer_js():
    if getattr(inject_bounding_boxer_js, 'injected', False) is True:
        return

    inject_bounding_boxer_js.injected = True
    display_javascript(Javascript(_bounding_boxer_js))


_bounding_boxer_js = r'''
require.undef('de/cprohm/label/boundingBoxer');

define('de/cprohm/label/boundingBoxer', ["@jupyter-widgets/base"], function(widgets) {
    "use strict";
    
    const BoundingBoxer = widgets.DOMWidgetView.extend({
        render: function() {
            // hidden element to load the image
            this.img = document.createElement('img');
            this.img.style = 'display: none';
            this.el.appendChild(this.img);
            
            this.canvas = document.createElement('canvas');
            this.canvas.style = 'border: 1px solid black;'
            // make sure key events are fired
            // TODO: forward key events of the canvas to python
            this.canvas.tabIndex = 100;
            
            this.canvas.addEventListener('click', this.onCanvasClick.bind(this), false);
            this.canvas.addEventListener('mousemove', this.onCanvasMouseMove.bind(this), false);
            this.canvas.addEventListener('keydown', this.onCanvasKeyDown.bind(this), false);
            
            this.el.appendChild(this.canvas);
            
            this.model.on('change:_annotations', this.updateClient.bind(this));
            this.model.on('change:url', this.updateClient.bind(this));
            this.model.on('change:scale', this.updateClient.bind(this));
            
            this.updateClient();
        },
        
        updateServer: logerr(function() {
            console.log('update', JSON.stringify(this.annotations));
            this.model.set(
                '_annotations', JSON.stringify(this.annotations),
            );
            this.touch();
        }),
        updateClient: logerr(function() {
            console.log(this.model.get('_annotations'));
            this.annotations = JSON.parse(
                this.model.get('_annotations'),
            );
            this.url = this.model.get('url');
            this.scale = this.model.get('scale');
            
            // TODO: is this operation really synchronous?
            this.img.src = this.url || '';
            
            this.canvas.width = this.scale * this.img.width;
            this.canvas.height = this.scale * this.img.height;
            
            this.nextId = Math.max(0, ...this.annotations.map(a => a.id + 1));
            
            this.redraw();
        }),
        redraw: logerr(function(ctx) {
            if(ctx == null) {
                ctx = this.canvas.getContext('2d');
            }
            ctx.drawImage(
                this.img, 
                0, 0, this.img.width, this.img.height,
                0, 0, 
                this.scale * this.img.width, 
                this.scale * this.img.height,
            );
            
            for(const annotation of this.annotations) {
                ctx.strokeStyle = 'red';
                ctx.strokeRect(
                    annotation.x, annotation.y, 
                    annotation.w, annotation.h,
                );

                ctx.fillStyle = 'red';
                ctx.font = '12px sans-serif';
                ctx.fillText(
                    annotation.id + ':' + annotation.label, 
                    annotation.x, annotation.y + annotation.h,
                );
            }
        }),
        onCanvasClick: logerr(function(ev) {
            const pos = this.position(ev);

            if(this.pos == null) {
                this.pos = pos;
            }
            else {
                const x = ~~Math.min(this.pos.x, pos.x);
                const y = ~~Math.min(this.pos.y, pos.y);
                const w = ~~Math.abs(this.pos.x - pos.x);
                const h = ~~Math.abs(this.pos.y - pos.y);
                const label = this.model.get('current_tag');
                const id = this.nextId++;
                
                this.pos = null;   
                
                this.annotations.push({id, label, x, y, w, h});
                this.updateServer();
            }
            this.redraw();
        }),
        onCanvasMouseMove: logerr(function(ev) {
            if(this.pos != null) {
                var pos = this.position(ev);
                var x = Math.min(this.pos.x, pos.x)
                var y = Math.min(this.pos.y, pos.y);

                var w = Math.abs(this.pos.x - pos.x);
                var h = Math.abs(this.pos.y - pos.y);

                var ctx = this.canvas.getContext('2d');
                this.redraw(ctx);
                ctx.strokeStyle = 'green';
                ctx.strokeRect(x, y, w, h);
            }
        }),
        onCanvasKeyDown: logerr(function(ev) {
            ev.preventDefault();
            this.send({'type': 'keydown', 'which': ev['which']});
        }),
        position: function(event) {
            var rect = this.canvas.getBoundingClientRect();
            return {
              x: event.clientX - rect.left,
              y: event.clientY - rect.top,
            };
        },
    });
    
    const Annotations = widgets.DOMWidgetView.extend({
        render: function() {
            this.model.on('change:_annotations', this.redraw.bind(this));
            this.redraw();
        },
        updateServer: logerr(function() {
            this.model.set(
                '_annotations', JSON.stringify(this.annotations),
            );
            this.touch();
        }),
        redraw: logerr(function() {
            // keep the array around for bidrectional updates
            this.annotations = JSON.parse(this.model.get('_annotations'));
            
            // clear the element
            while(this.el.firstChild) {
                this.el.removeChild(this.el.firstChild);
            }
            
            // add new annotations
            for(const annotation of this.annotations) {
                this.el.appendChild(
                    div(
                        text('' + annotation.id + ' '),
                        synced(annotation, 'label'),
                        text(': '),
                        synced(annotation, 'x'),
                        text('/'),
                        synced(annotation, 'y'),
                        text(' '),
                        synced(annotation, 'w'),
                        text('/'),
                        synced(annotation, 'h'),
                        removeButton(annotation),
                    ),
                );
            }
            
            const self = this;
            function div(...children) {
                const el = document.createElement('div');
                for(const child of children) {
                    el.appendChild(child);
                }
                return el;
            }
            
            function text(s) {
                return document.createTextNode(s);
            }
            
            function synced(obj, key) {
                const el = document.createElement('input');
                el.value = obj[key];
                
                const update = () => {
                    obj[key] = el.value;
                    self.updateServer();
                }

                el.addEventListener('change', update);
                el.addEventListener('keydown', function(ev) {
                    if(ev.which == 13) {
                        ev.preventDefault();
                        update();
                    }
                });

                return el;
            }
            
            function removeButton(obj) {
                const button = document.createElement('button');
                button.innerHTML = 'x';
                
                button.addEventListener('click', () => {
                    const row = button.parentNode;
                    const el = row.parentNode;
                    el.removeChild(row);
                    
                    self.annotations = self.annotations.filter(
                        o => o.id != obj.id
                    );
                    console.log(JSON.stringify(self.annotations));
                    self.updateServer();
                });
                return button;
            }
        }),
    })
    
    function logerr(func) {
        return function() {
            try {
                return func.apply(this, arguments);
            }
            catch(err) {
                console.error(err);
                throw err;
            }
        }
    }
    
    return {BoundingBoxer, Annotations};
});
'''