import json

from ipywidgets import DOMWidget
from traitlets import Unicode

__all__ = ['VegaWidget']


class VegaWidget(DOMWidget):
    """An IPython widget display a vega chart.

    Usage with ``altair``::

        widget = VegaWidget(chart=chart)
        display(widget)

    """
    _view_name = Unicode('VegaWidget').tag(sync=True)
    _view_module = Unicode('vegawidget').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)
    spec_source = Unicode('null').tag(sync=True)

    def __init__(self, *, spec=None, chart=None, **kwargs):
        super().__init__(**kwargs)
        if (spec is not None) and (chart is not None):
            raise ValueError('can only use either spec or chart')

        if spec is None and chart is not None:
            spec = chart.to_dict()

        self.spec_source = json.dumps(spec)

    @property
    def spec(self):
        return json.loads(self.spec_source)

    @spec.setter
    def spec(self, value):
        self.spec_source = json.dumps(value)

    def update(self, key, remove=None, insert=None):
        data = dict(type='update', key=key)

        if remove is not None:
            data['remove'] = remove

        if insert is not None:
            data['insert'] = insert

        self.send(data)
