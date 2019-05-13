"""Helpers for data science.

Distributed as part of ``https://github.com/chmp/misc-exp`` under the MIT
license, (c) 2017 Christopher Prohm.
"""
import base64
import bisect
import bz2
import collections
import datetime
import enum
import functools as ft
import gzip
import hashlib
import importlib
import inspect
import io
import itertools as it
import json
import logging
import math
import os.path
import pathlib
import pickle
import sys
import threading
import time

from types import ModuleType
from typing import Any, Callable, Iterable, NamedTuple, Optional, Union

try:
    from sklearn.base import (
        BaseEstimator,
        TransformerMixin,
        ClassifierMixin,
        RegressorMixin,
    )

except ImportError:
    from ._import_compat import (  # typing: ignore
        BaseEstimator,
        TransformerMixin,
        ClassifierMixin,
        RegressorMixin,
    )

    _HAS_SK_LEARN = False


else:
    _HAS_SK_LEARN = True


try:
    from daft import PGM

except ImportError:
    from ._import_compat import PGM  # typing: ignore

    _HAS_DAFT = False

else:
    _HAS_DAFT = True


def reload(*modules_or_module_names: Union[str, ModuleType]) -> Optional[ModuleType]:
    mod = None
    for module_or_module_name in modules_or_module_names:
        if isinstance(module_or_module_name, str):
            module_or_module_name = importlib.import_module(module_or_module_name)

        mod = importlib.reload(module_or_module_name)

    return mod


def import_object(obj):
    def _import_obj(obj):
        module, _, name = obj.partition(":")
        module = importlib.import_module(module)
        return getattr(module, name)

    return sapply(_import_obj, obj)


def define(func):
    """Execute a function and return its result.

    The idea is to use function scope to prevent pollution of global scope in
    notebooks.

    Usage::

        @define
        def foo():
            return 42

        assert foo == 42

    """
    return func()


def cached(path: str, validate: bool = False):
    """Similar to ``define``, but cache to a file.

    :param path:
        the path of the cache file to use
    :param validate:
        if `True`, always execute the function. The loaded result will be
        passed to the function, when the cache exists. In that case the
        function should return the value to use. If the returned value is
        not identical to the loaded value, the cache is updated with the
        new value.

    Usage::

        @cached('./cache/result')
        def dataset():
            ...
            return result

    or::

        @cached('./cache/result', validate=True)
        def model(result=None):
            if result is not None:
                # running to validate ...

            return result
    """

    def update_cache(result):
        print("save cache", path)
        with open(path, "wb") as fobj:
            pickle.dump(result, fobj)

    def load_cache():
        print("load cache", path)
        with open(path, "rb") as fobj:
            return pickle.load(fobj)

    def decorator(func):
        if os.path.exists(path):
            result = load_cache()

            if not validate:
                return result

            else:
                print("validate")
                new_result = func(result)

                if new_result is not result:
                    update_cache(new_result)

                return new_result

        else:
            print("compute")
            result = func()
            update_cache(result)
            return result

    return decorator


class Object:
    """Dictionary-like namespace object."""

    def __init__(*args, **kwargs):
        self, *args = args

        if len(args) > 1:
            raise ValueError(
                "Object(...) can be called with at " "most one positional argument"
            )

        elif len(args) == 0:
            seed = {}

        else:
            seed, = args
            if not isinstance(seed, collections.Mapping):
                seed = vars(seed)

        for k, v in dict(seed, **kwargs).items():
            setattr(self, k, v)

    def __repr__(self):
        return "Object({})".format(
            ", ".join("{}={!r}".format(k, v) for k, v in vars(self).items())
        )

    def __eq__(self, other):
        return type(self) == type(other) and vars(self) == vars(other)

    def __ne__(self, other):
        return not (self == other)


class daterange:
    """A range of dates."""

    start: datetime.date
    end: datetime.date
    step: datetime.timedelta

    @classmethod
    def around(cls, dt, start, end, step=None):
        if not isinstance(start, datetime.timedelta):
            start = datetime.timedelta(days=start)

        if not isinstance(end, datetime.timedelta):
            end = datetime.timedelta(days=end)

        if step is None:
            step = datetime.timedelta(days=1)

        elif not isinstance(step, datetime.timedelta):
            step = datetime.timedelta(days=step)

        return cls(dt + start, dt + end, step)

    def __init__(
        self,
        start: datetime.date,
        end: datetime.date,
        step: Optional[datetime.timedelta] = None,
    ):
        if step is None:
            step = datetime.timedelta(days=1)

        self.start = start
        self.end = end
        self.step = step

    def __len__(self) -> int:
        return len(self._offset_range)

    def __iter__(self) -> Iterable[datetime.date]:
        for offset in self._offset_range:
            yield self.start + datetime.timedelta(days=offset)

    def __contains__(self, item: datetime.date) -> bool:
        return self._offset(item) in self._offset_range

    def __getitem__(self, index: int) -> datetime.date:
        return self.start + datetime.timedelta(days=self._offset_range[index])

    def count(self, item: datetime.date) -> int:
        return 1 if (item in self) else 0

    def index(self, item):
        return self._offset_range.index(self._offset(item))

    def _offset(self, item: datetime.date) -> int:
        return (item - self.start).days

    @property
    def _offset_range(self) -> range:
        return range(0, (self.end - self.start).days, self.step.days)

    def __repr__(self):
        return f"daterange({self.start}, {self.end}, {self.step})"


class undefined_meta(type):
    def __repr__(self):
        return "<undefined>"


class undefined(metaclass=undefined_meta):
    """Sentinel class"""

    pass


def first(iterable, default=undefined):
    """Return the first item of an iterable"""
    for item in iterable:
        return item

    return default


def last(iterable, default=undefined):
    """Return the last item of an iterable"""
    item = default
    for item in iterable:
        pass

    return item


def item(iterable, default=undefined):
    """Given a single item iterable return this item."""
    found = undefined

    for item in iterable:
        if found is not undefined:
            raise ValueError("More than one value to unpack")

        found = item

    if found is not undefined:
        return found

    if default is not undefined:
        return default

    raise ValueError("Need at least one item or a default")


def collect(iterable):
    result = {}
    for k, v in iterable:
        result.setdefault(k, []).append(v)

    return result


class kvpair(NamedTuple):
    key: Any
    value: Any


class cell:
    """No-op context manager to allow indentation of code"""

    def __init__(self, name=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        pass

    def __call__(self, func):
        with self:
            func()


def colorize(items, cmap=None):
    """Given an iterable, yield ``(color, item)`` pairs.

    :param cmap:
        if None the color cycle is used, otherwise it is interpreted as a
        colormap to color the individual items.

        Note: ``items`` is fully instantiated during the iteration. For any
        ``list`` or ``tuple`` item only its first element is used for
        colomapping.

        This procedure allows for example to colormap a pandas Dataframe
        grouped on a number column::

            for c, (_, g) in colorize(df.groupby("g"), cmap="viridis"):
                ...
    """
    if cmap is None:
        cycle = get_color_cycle()
        return zip(it.cycle(cycle), items)

    else:
        items = list(items)

        if not items:
            return iter(())

        keys = [item[0] if isinstance(item, (tuple, list)) else item for item in items]

        return zip(colormap(keys, cmap=cmap), items)


def get_color_cycle(n=None):
    """Return the matplotlib color cycle.

    :param Optional[int] n:
        if given, return a list with exactly n elements formed by repeating
        the color cycle as necessary.

    Usage::

        blue, green, red = get_color_cycle(3)

    """
    import matplotlib as mpl

    cycle = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

    if n is None:
        return it.cycle(cycle)

    return list(it.islice(it.cycle(cycle), n))


def mpl_set(
    box=None,
    xlabel=None,
    ylabel=None,
    title=None,
    suptitle=None,
    xscale=None,
    yscale=None,
    caption=None,
    xlim=None,
    ylim=None,
    xticks=None,
    yticks=None,
    xformatter: Optional[Callable[[float, float], str]] = None,
    yformatter: Optional[Callable[[float, float], str]] = None,
    left=None,
    top=None,
    bottom=None,
    right=None,
    wspace=None,
    hspace=None,
    subplot=None,
    legend=None,
    colorbar=None,
    invert: Optional[str] = None,
    ax=None,
    grid=None,
    axis=None,
):
    """Set various style related options of MPL.

    :param xformatter:
        if given a formatter for the major x ticks. Should have the
        signature ``(x_value, pos) -> label``.

    :param yformatter:
        See ``xformatter``.

    :param invert:
        if given invert the different axes. Can be `x`, `y`, or `xy`.
    """
    import matplotlib.pyplot as plt

    if ax is not None:
        plt.sca(ax)

    if box is not None:
        plt.box(box)

    if subplot is not None:
        ax = plt.gca()
        plt.subplot(*subplot)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    if suptitle is not None:
        plt.suptitle(suptitle)

    if xscale is not None:
        plt.xscale(xscale)

    if yscale is not None:
        plt.yscale(yscale)

    # TODO: handle min/max, enlarge ...
    if xlim is not None:
        plt.xlim(*xlim)

    if ylim is not None:
        plt.ylim(*ylim)

    if xticks is not None:
        if isinstance(xticks, tuple):
            plt.xticks(*xticks)

        else:
            plt.xticks(xticks)

    if yticks is not None:
        if isinstance(yticks, tuple):
            plt.yticks(*yticks)

        else:
            plt.yticks(yticks)

    if xformatter is not None:
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(xformatter))

    if yformatter is not None:
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(yformatter))

    if caption is not None:
        _caption(caption)

    subplot_kwargs = _dict_of_optionals(
        left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace
    )

    if subplot_kwargs:
        plt.subplots_adjust(**subplot_kwargs)

    if legend is not None and legend is not False:
        if legend is True:
            plt.legend(loc="best")

        elif isinstance(legend, str):
            plt.legend(loc=legend)

        else:
            plt.legend(**legend)

    if subplot is not None:
        plt.sca(ax)

    if colorbar is True:
        plt.colorbar()

    if invert is not None:
        if "x" in invert:
            plt.gca().invert_xaxis()

        if "y" in invert:
            plt.gca().invert_yaxis()

    if grid is not None:
        if not isinstance(grid, list):
            grid = [grid]

        for spec in grid:
            if isinstance(spec, bool):
                b, which, axis = spec, "major", "both"

            elif isinstance(spec, str):
                b, which, axis = True, "major", spec

            elif isinstance(spec, tuple) and len(spec) == 2:
                b, which, axis = True, spec[0], spec[1]

            elif isinstance(spec, tuple):
                b, which, axis = spec

            else:
                raise RuntimeError()

            plt.grid(b, which, axis)

    if axis is not None and axis is not True:
        if axis is False:
            axis = "off"

        plt.axis(axis)


class mpl_axis:
    def __init__(self, ax=None, **kwargs):
        self.ax = ax
        self._prev_ax = None
        self.kwargs = kwargs

    def __enter__(self):
        import matplotlib.pyplot as plt

        if plt.get_fignums():
            self._prev_ax = plt.gca()

        if self.ax is None:
            _, self.ax = plt.subplots()

        plt.sca(self.ax)
        return self.ax

    def __exit__(self, exc_type, exc_value, exc_tb):
        import matplotlib.pyplot as plt

        mpl_set(**self.kwargs)

        if self._prev_ax is not None:
            plt.sca(self._prev_ax)


# fake the mpl_axis signature ...
# TODO: make this a general utility function?
@define
def _():
    import collections
    import inspect

    wrapper_signature = inspect.signature(mpl_axis)
    base_signature = inspect.signature(mpl_set)

    parameters = collections.OrderedDict()
    parameters["ax"] = wrapper_signature.parameters["ax"].replace(
        kind=inspect.Parameter.POSITIONAL_ONLY
    )
    parameters.update(base_signature.parameters)

    mpl_axis.__signature__ = wrapper_signature.replace(parameters=parameters.values())


def diagonal(**kwargs):
    """Draw a diagonal line in the current axis."""
    import matplotlib.pyplot as plt

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    vmin = max(xmin, ymin)
    vmax = min(xmax, ymax)

    plt.plot([vmin, vmax], [vmin, vmax], **kwargs)


def qlineplot(*, x, y, hue, data, ci=0.95):
    """Plot  median as line, quantiles as shading.
    """
    import matplotlib.pyplot as plt

    agg_data = data.groupby([x, hue])[y].quantile([1 - ci, 0.5, ci]).unstack()
    hue_values = data[hue].unique()

    for color, hue_value in colorize(hue_values):
        subset = agg_data.xs(hue_value, level=hue)
        plt.fill_between(subset.index, subset.iloc[:, 0], subset.iloc[:, 2], alpha=0.2)

    for color, hue_value in colorize(hue_values):
        subset = agg_data.xs(hue_value, level=hue)
        plt.plot(subset.index, subset.iloc[:, 1], label=hue_value, marker=".")

    plt.legend(loc="best")
    plt.xlabel(x)
    plt.ylabel(y)


class pgm:
    """Wrapper around :class:`daft.PGM` to allow fluid call chains.

    Usage::

        (
            pgm(observed_style="inner", ax=ax1)
            .node("z", r"$Z$", 1.5, 2)
            .node("x", r"$X$", 1, 1)
            .node("y", r"$Y$", 2, 1)
            .edge("z", "x")
            .edge("x", "y")
            .edge("z", "y")
            .render(xlim=(1, 5), ylim=(1, 5))
        )

    To annotate a node use::

        .annotate(node_name, annotation_text)

    Nodes can also be created without explicit lables (in which case the node
    name is used)::

        .node("z", 1, 1)
        node("z", "label", 1, 1)

    """

    def __init__(self, *, ax=None, nodes=(), edges=(), annotations=(), **kwargs):
        if not _HAS_DAFT:
            raise RuntimeError("daft is required for pgm support.")

        self.ax = ax
        self.daft_kwargs = kwargs

        self._nodes = list(nodes)
        self._edges = list(edges)
        self._annotations = list(annotations)

    def update(self, nodes=None, edges=None, annotations=None):
        """Replace a full set of features."""
        if nodes is None:
            nodes = self._nodes

        if edges is None:
            edges = self._edges

        if annotations is None:
            annotations = self._annotations

        return type(self)(
            nodes=nodes,
            edges=edges,
            annotations=annotations,
            ax=self.ax,
            **self.daft_kwargs,
        )

    def node(self, *args, edgecolor=None, facecolor=None, **kwargs):
        if edgecolor is not None:
            kwargs.setdefault("plot_params", {}).update(edgecolor=edgecolor)

        if facecolor is not None:
            kwargs.setdefault("plot_params", {}).update(facecolor=facecolor)

        node = Object(kwargs=kwargs)
        if len(args) == 3:
            node.name, node.x, node.y = args
            node.label = node.name

        else:
            node.name, node.label, node.x, node.y = args

        return self.update(nodes=self._nodes + [node])

    def edge(self, from_node, to_node, **kwargs):
        edge = Object(from_node=from_node, to_node=to_node, kwargs=kwargs)
        return self.update(edges=self._edges + [edge])

    def edges(self, from_nodes, to_nodes, **kwargs):
        current = self
        for from_node, to_node in it.product(from_nodes, to_nodes):
            current = current.edge(from_node, to_node, **kwargs)
        return current

    def remove(self, incoming=(), outgoing=()):
        """Remove edges that point in or out of a the specified nodes.
        """
        incoming = set(incoming)
        outgoing = set(outgoing)
        edges_to_keep = [
            edge
            for edge in self._edges
            if (edge.from_node not in outgoing and edge.to_node not in incoming)
        ]

        return self.update(edges=edges_to_keep)

    def annotate(self, node, text):
        annotation = Object(node=node, text=text)
        return self.update(annotations=self._annotations + [annotation])

    def render(self, ax=None, axis=False, xlim=None, ylim=None, **kwargs):
        """Render the figure.

        :param ax:
            the axes to draw into. If not given, the axis specified in
            `__init__` or the current axes is used.
        :param xlim:
            the xlim to use. If not given, it is determined from the data.
        :param ylim:
            the ylim to use. If not given, it is determined from the data.
        :param kwargs:
            keyword arguments forward to mpl set.

        :returns:
            the `pgm` object.
        """
        import daft
        import matplotlib.pyplot as plt

        if ax is None:
            if self.ax is not None:
                ax = self.ax

            else:
                ax = plt.gca()

        pgm = _PGM(ax=ax)

        for node in self._nodes:
            daft_node = daft.Node(node.name, node.label, node.x, node.y, **node.kwargs)
            pgm.add_node(daft_node)

        for edge in self._edges:
            pgm.add_edge(edge.from_node, edge.to_node, **edge.kwargs)

        for annot in self._annotations:
            self._render_annotation(pgm, annot)

        if xlim is None or ylim is None:
            data_xlim, data_ylim = pgm.get_limits()
            if xlim is None:
                xlim = expand(*data_xlim, 0.10)

            if ylim is None:
                ylim = expand(*data_ylim, 0.10)

        pgm.render()
        mpl_set(**kwargs, axis=axis, xlim=xlim, ylim=ylim, ax=ax)

        return pgm

    def _render_annotation(self, pgm, annot):
        extent = pgm.get_node_extent(annot.node)
        pgm._ctx._ax.text(
            extent.x, extent.y - 0.5 * extent.height, annot.text, va="top", ha="center"
        )

    def _ipython_display_(self):
        self.render()


class _PGM(PGM):
    def __init__(self, *, ax=None, **kwargs):
        super().__init__([1.0, 1.0], origin=[0.0, 0.0], **kwargs)
        self._ctx._ax = ax
        self._ctx._figure = ax.get_figure()

    def get_node_extent(self, node):
        # TODO: incorporate the complete logic of daft?
        ctx = self._ctx

        if isinstance(node, str):
            node = self._nodes[node]

        aspect = node.aspect if node.aspect is not None else ctx.aspect
        height = node.scale * ctx.node_unit
        width = aspect * height

        center_x = ctx.grid_unit * node.x
        center_y = ctx.grid_unit * node.y

        return Object(
            x=center_x,
            y=center_y,
            width=width,
            height=height,
            xmin=center_x - 0.5 * width,
            xmax=center_x + 0.5 * width,
            ymin=center_y - 0.5 * height,
            ymax=center_y + 0.5 * height,
        )

    def get_limits(self):
        nodes = list(self._nodes.values())

        if not nodes:
            return (0, 1), (0, 1)

        extent = self.get_node_extent(nodes[0])

        xmin = extent.xmin
        xmax = extent.xmax
        ymin = extent.ymin
        ymax = extent.ymax

        for node in nodes[1:]:
            extent = self.get_node_extent(node)
            xmin = min(xmin, extent.xmin)
            xmax = max(xmax, extent.xmax)
            ymin = min(ymin, extent.ymin)
            ymax = max(ymax, extent.ymax)

        return (xmin, xmax), (ymin, ymax)


def edges(x):
    """Create edges for use with pcolor.

    Usage::

        assert x.size == v.shape[1]
        assert y.size == v.shape[0]
        pcolor(edges(x), edges(y), v)

    """
    import numpy as np

    centers = 0.5 * (x[1:] + x[:-1])
    return np.concatenate(
        ([x[0] - 0.5 * (x[1] - x[0])], centers, [x[-1] + 0.5 * (x[-1] - x[-2])])
    )


def center(u):
    """Compute the center between edges."""
    return 0.5 * (u[1:] + u[:-1])


def caption(s, size=13, strip=True):
    """Add captions to matplotlib graphs."""
    import matplotlib.pyplot as plt

    if strip:
        s = s.splitlines()
        s = (i.strip() for i in s)
        s = (i for i in s if i)
        s = " ".join(s)

    plt.figtext(0.5, 0, s, wrap=True, size=size, va="bottom", ha="center")


_caption = caption


def change_vspan(
    x,
    y,
    *,
    data=None,
    color=("w", "0.90"),
    transform_x=None,
    transform_y=None,
    skip_nan=True,
    **kwargs,
):
    """Plot changes in a quantity with vspans.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    x, y = _prepare_xy(
        x,
        y,
        data=data,
        transform_x=transform_x,
        transform_y=transform_y,
        skip_nan=skip_nan,
    )

    if not isinstance(color, (tuple, list)):
        color = [color]

    changes = _find_changes(y)
    changes = np.concatenate([[0], changes, [len(y) - 1]])

    for start, end, c in zip(changes[:-1], changes[1:], it.cycle(color)):
        plt.axvspan(x[start], x[end], color=c, **kwargs)


def change_plot(
    x, y, *, data=None, transform_x=None, transform_y=None, skip_nan=True, **kwargs
):
    """Plot changes in a quantity with pyplot's standard plot function.
    """
    import matplotlib.pyplot as plt

    x, y = _prepare_xy(
        x,
        y,
        data=data,
        transform_x=transform_x,
        transform_y=transform_y,
        skip_nan=skip_nan,
    )
    changes = _find_changes(y)

    x = x[changes]
    y = y[changes]

    plt.plot(x, y, **kwargs)


def axtext(*args, **kwargs):
    """Add a text in axes coordinates (similar ``figtext``).

    Usage::

        axtext(0, 0, 'text')

    """
    import matplotlib.pyplot as plt

    kwargs.update(transform=plt.gca().transAxes)
    plt.text(*args, **kwargs)


def plot_gaussian_contour(x, y, *, data=None, q=(0.99,), ax=None, **kwargs):
    """Plot isocontours of the maximum likelihood Gaussian for ``x, y``.

    :param q:
        the quantiles to show.
    """
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import scipy.special

    if ax is not None:
        plt.sca(ax)

    kwargs.setdefault("facecolor", "none")
    kwargs.setdefault("edgecolor", "k")

    q = np.atleast_1d(q)

    if data is not None:
        x = data[x]
        y = data[y]

    x = np.asarray(x)
    y = np.asarray(y)

    mx = np.mean(x)
    my = np.mean(y)
    xx = np.mean((x - mx) * (x - mx))
    yy = np.mean((y - my) * (y - my))
    xy = np.mean((x - mx) * (y - my))

    cov = np.asarray([[xx, xy], [xy, yy]])
    eigvals, eigvects = np.linalg.eig(cov)

    dx, dy = eigvects[:, 0]
    angle = math.atan2(dy, dx) / (2 * math.pi) * 360

    for _q in q:
        s = (2 ** 0.5) * scipy.special.erfinv(_q)
        artist = mpl.patches.Ellipse((mx, my), *(s * eigvals), angle, **kwargs)
        plt.gca().add_artist(artist)

    return artist


def _prepare_xy(x, y, data=None, transform_x=None, transform_y=None, skip_nan=True):
    if data is not None:
        x = data[x]
        y = data[y]

    x, y = _optional_skip_nan(x, y, skip_nan=skip_nan)

    if transform_x is not None:
        x = transform_x(x)

    if transform_y is not None:
        y = transform_y(y)

    return x, y


def _find_changes(v):
    import numpy as np

    changes, = np.nonzero(np.diff(v))
    changes = changes + 1
    return changes


def _optional_skip_nan(x, y, skip_nan=True):
    import numpy as np

    if not skip_nan:
        return x, y

    s = np.isfinite(y)
    return x[s], y[s]


def _dict_of_optionals(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}


@ft.singledispatch
def get_children(est):
    return []


def to_markdown(df, index=False):
    """Return a string containg the markdown of the table.

    Depends on the ``tabulate`` dependency.
    """
    from tabulate import tabulate

    return tabulate(df, tablefmt="pipe", headers="keys", showindex=index)


def index_query(obj, expression, scalar=False):
    """Execute a query expression on the index and return matching rows.

    :param scalar:
        if True, return only the first item. Setting ``scalar=True``
        raises an error if the resulting object has have more than one
        entry.
    """
    res = obj.loc[obj.index.to_frame().query(expression).index]

    if scalar:
        assert res.size == 1
        return res.iloc[0]

    return res


def fix_categories(
    s, categories=None, other_category=None, inplace=False, groups=None, ordered=False
):
    """Fix the categories of a categorical series.

    :param pd.Series s:
        the series to normalize

    :param Optional[Iterable[Any]] categories:
        the categories to keep. The result will have categories in the
        iteration order of this parameter. If not given but ``groups`` is
        passed, the keys of ``groups`` will be used, otherwise the existing
        categories of ``s`` will be used.

    :param Optional[Any] other_category:
        all categories to be removed wil be mapped to this value, unless they
        are specified specified by the ``groups`` parameter. If given and not
        included in the categories, it is appended to the given categories.
        For a custom order, ensure it is included in ``categories``.

    :param bool inplace:
        if True the series will be modified in place.

    :param Optional[Mapping[Any,Iterable[Any]]] groups:
        if given, specifies which categories to replace by which in the form
        of ``{replacement: list_of_categories_to_replace}``.

    :param bool ordered:
        if True the resulting series will have ordered categories.
    """
    import pandas.api.types as pd_types

    if not inplace:
        s = s.copy()

    if not pd_types.is_categorical(s):
        if inplace:
            raise ValueError("cannot change the type inplace")

        s = s.astype("category")

    if categories is None:
        if groups is not None:
            categories = list(groups.keys())

        else:
            categories = list(s.cat.categories)

    categories = list(categories)
    inital_categories = set(s.cat.categories)

    if other_category is not None and other_category not in categories:
        categories = categories + [other_category]

    additions = [c for c in categories if c not in inital_categories]
    removals = [c for c in inital_categories if c not in categories]

    if groups is None:
        groups = {}

    else:
        groups = {k: set(v) for k, v in groups.items()}

    remapped = {c for group in groups.values() for c in group}

    dangling_categories = {*removals} - {*remapped}
    if dangling_categories:
        if other_category is None:
            raise ValueError(
                "dangling categories %s found, need other category to assign"
                % dangling_categories
            )

        groups.setdefault(other_category, set()).update(set(removals) - set(remapped))

    if additions:
        s.cat.add_categories(additions, inplace=True)

    for replacement, group in groups.items():
        s[s.isin(group)] = replacement

    if removals:
        s.cat.remove_categories(removals, inplace=True)

    s.cat.reorder_categories(categories, inplace=True, ordered=ordered)

    return s


def find_high_frequency_categories(s, min_frequency=0.02, n_max=None):
    """Find categories with high frequency.

    :param float min_frequency:
        the minimum frequency to keep

    :param Optional[int] n_max:
        if given keep at most ``n_max`` categories. If more are present after
        filtering for minimum frequency, keep the highest ``n_max`` frequency
        columns.
    """
    assert 0.0 < min_frequency < 1.0
    s = s.value_counts(normalize=True).pipe(lambda s: s[s > min_frequency])

    if n_max is None:
        return list(s.index)

    if len(s) <= n_max:
        return s

    return list(s.sort_values(ascending=False).iloc[:n_max].index)


def as_frame(**kwargs):
    import pandas as pd

    return pd.DataFrame().assign(**kwargs)


def singledispatch_on(idx):
    """Helper to dispatch on any argument, not only the first one."""

    # It works by wrapping the function to include the relevant
    # argument as first argument as well.
    def decorator(func):
        @ft.wraps(func)
        def wrapper(*args, **kwargs):
            dispatch_obj = args[idx]
            return dispatcher(dispatch_obj, *args, **kwargs)

        def make_call_impl(func):
            @ft.wraps(func)
            def impl(*args, **kwargs):
                _, *args = args
                return func(*args, **kwargs)

            return impl

        def register(type):
            def decorator(func):
                dispatcher.register(type)(make_call_impl(func))
                return func

            return decorator

        wrapper.register = register
        dispatcher = ft.singledispatch(make_call_impl(func))

        return wrapper

    return decorator


def setdefaultattr(obj, name, value):
    """``dict.setdefault`` for attributes"""
    if not hasattr(obj, name):
        setattr(obj, name, value)

    return getattr(obj, name)


# keep for backwards compat
def sapply(func, obj, sequences=(tuple,), mappings=(dict,)):
    return smap(func, obj, sequences=sequences, mappings=mappings)


def szip(
    iterable_of_objects, sequences=(tuple,), mappings=(dict,), return_schema=False
):
    """Zip but for deeply nested objects.

    For a list of nested set of objects return a nested set of list.
    """
    iterable_of_objects = iter(iterable_of_objects)

    try:
        first = next(iterable_of_objects)

    except StopIteration:
        return None

    # build a scaffold into which the results are appended
    # NOTE: the target lists must not be confused with the structure, use a
    # schema object as an unambiguous marker.
    schema = smap(lambda _: None, first, sequences=sequences, mappings=mappings)
    target = smap(lambda _: [], schema, sequences=sequences, mappings=mappings)

    for obj in it.chain([first], iterable_of_objects):
        smap(
            lambda _, t, o: t.append(o),
            schema,
            target,
            obj,
            sequences=sequences,
            mappings=mappings,
        )

    return target if return_schema is False else (target, schema)


def flatten_with_index(obj, sequences=(tuple,), mappings=(dict,)):
    counter = iter(it.count())
    flat = []

    def _build(item):
        flat.append(item)
        return next(counter)

    index = smap(_build, obj, sequences=sequences, mappings=mappings)
    return index, flat


def unflatten(index, obj, sequences=(tuple,), mappings=(dict,)):
    obj = list(obj)
    return smap(lambda idx: obj[idx], index, sequences=sequences, mappings=mappings)


def smap(func, arg, *args, sequences=(tuple,), mappings=(dict,)):
    """A structured version of map.

    The structure is taken from the first arguments.
    """
    return _smap(func, arg, *args, path="$", sequences=sequences, mappings=mappings)


def _smap(func, arg, *args, path, sequences=(tuple,), mappings=(dict,)):
    try:
        if isinstance(arg, sequences):
            return type(arg)(
                _smap(
                    func,
                    *co,
                    path=f"{path}.{idx}",
                    sequences=sequences,
                    mappings=mappings,
                )
                for idx, *co in zip(it.count(), arg, *args)
            )

        elif isinstance(arg, mappings):
            return type(arg)(
                (
                    k,
                    _smap(
                        func,
                        arg[k],
                        *(obj[k] for obj in args),
                        path=f"{path}.k",
                        sequences=sequences,
                        mappings=mappings,
                    ),
                )
                for k in arg
            )

        else:
            return func(arg, *args)

    # pass through any exceptions in smap without further annotations
    except SApplyError:
        raise

    except Exception as e:
        raise SApplyError(f"Error in sappend at {path}: {e}") from e


class SApplyError(Exception):
    pass


def piecewise_linear(x, y, xi):
    return _piecewise(_linear_interpolator, x, y, xi)


def piecewise_logarithmic(x, y, xi=None):
    return _piecewise(_logarithmic_interpolator, x, y, xi)


def _linear_interpolator(u, y0, y1):
    return y0 + u * (y1 - y0)


def _logarithmic_interpolator(u, y0, y1):
    return (y0 ** (1 - u)) * (y1 ** u)


def _piecewise(interpolator, x, y, xi):
    assert len(x) == len(y)
    interval = bisect.bisect_right(x, xi)

    if interval == 0:
        return y[0]

    if interval >= len(x):
        return y[-1]

    u = (xi - x[interval - 1]) / (x[interval] - x[interval - 1])
    return interpolator(u, y[interval - 1], y[interval])


bg_instances = {}


def bgloop(tag, *iterables, runner=None):
    """Run a loop in a background thread."""
    if runner is None:
        runner = run_thread

    def decorator(func):
        if tag in bg_instances and bg_instances[tag].running:
            raise RuntimeError("Already running loop")

        bg_instances[tag] = Object()
        bg_instances[tag].running = True
        bg_instances[tag].handle = runner(_run_loop, tag, func, iterables)

        return func

    def _run_loop(tag, func, iterables):
        try:
            bg_instances[tag].running = True
            for loop, item in Loop.over(
                fast_product(*iterables), length=product_len(*iterables)
            ):
                if not bg_instances[tag].running:
                    break

                func(loop, *item)

        finally:
            bg_instances[tag].running = False

    return decorator


def cancel(tag):
    if tag in bg_instances:
        bg_instances[tag].running = False


def wait(tag):
    if tag in bg_instances and bg_instances[tag].handle is not None:
        bg_instances[tag].handle.join()


def run_direct(*args, **kwargs):
    func, *args = args
    func(*args, **kwargs)


def run_thread(*args, **kwargs):
    func, *args = args
    t = threading.Thread(target=func, args=args, kwargs=kwargs)
    t.start()
    return t


def product_len(*iterables):
    if not iterables:
        return 1

    head, *tail = iterables
    return len(head) * product_len(*tail)


def fast_product(*iterables):
    if not iterables:
        yield ()
        return

    head, *tail = iterables
    for i in head:
        for j in fast_product(*tail):
            yield (i,) + j


class Display:
    """An interactive display for use in background tasks."""

    def __init__(self, obj=None):
        from IPython.core.display import display

        self.handle = display(obj, display_id=True)

    def update(self, obj):
        self.handle.update(obj)

    def print(self, *args, sep=" "):
        from IPython.core.display import Pretty

        self.handle.update(Pretty(sep.join(str(a) for a in args)))

    def figure(self):
        from IPython.core.display import Image
        import matplotlib.pyplot as plt

        with io.BytesIO() as fobj:
            plt.savefig(fobj, format="png")
            plt.close()

            self.handle.update(Image(fobj.getvalue(), format="png"))


def pd_has_ordered_assign():
    import pandas as pd

    py_major, py_minor, *_ = sys.version_info
    pd_major, pd_minor, *_ = pd.__version__.split(".")
    pd_major = int(pd_major)
    pd_minor = int(pd_minor)

    return (py_major, py_minor) >= (3, 6) and (pd_major, pd_minor) >= (0, 23)


def timed(tag=None, level=logging.INFO):
    """Time a codeblock and log the result.

    Usage::

        with timed():
            long_running_operation()

    :param any tag:
        an object used to identify the timed code block. It is printed with
        the time taken.
    """
    return _TimedContext(
        message=("[TIMING] %s s" if tag is None else "[TIMING] {} %s s".format(tag)),
        logger=_get_caller_logger(),
        level=level,
    )


# use a custom contextmanager to control stack level for _get_caller_logger
class _TimedContext(object):
    def __init__(self, logger, message, level):
        self.logger = logger
        self.message = message
        self.level = level

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        self.logger.log(self.level, self.message, end - self.start)


def _get_caller_logger(depth=2):
    stack = inspect.stack()

    if depth >= len(stack):  # pragma: no cover
        return logging.getLogger(__name__)

    # NOTE: python2 returns raw tuples, index 0 is the frame
    frame = stack[depth][0]
    name = frame.f_globals.get("__name__")
    return logging.getLogger(name)


def cast_types(numeric=None, categorical=None):
    """Build a transform to cast numerical / categorical columns.

    All non-cast columns are stripped for the dataframe.

    """
    return transform(_cast_types, numeric=numeric, categorical=categorical)


def _cast_types(df, numeric, categorical):
    numeric = set() if numeric is None else set(numeric)
    categorical = set() if categorical is None else set(categorical)

    for col in numeric:
        try:
            df = df.assign(**{col: df[col].astype(float)})

        except Exception as e:
            raise RuntimeError(f"could not cast {col} to numeric") from e

    for col in categorical:
        try:
            df = df.assign(**{col: df[col].astype("category")})

        except Exception as e:
            raise RuntimeError(f"could not cast {col} to categorical") from e

    return df[sorted({*categorical, *numeric})]


def find_categorical_columns(df):
    """Find all categorical columns in the given dataframe.
    """
    import pandas.api.types as pd_types

    return [k for k, dtype in df.dtypes.items() if pd_types.is_categorical_dtype(dtype)]


def filter_low_frequency_categories(
    columns=None, min_frequency=0.02, other_category=None, n_max=None
):
    """Build a transformer to filter low frequency categories.

    Usage::

        pipeline = build_pipeline[
            categories=filter_low_frequency_categories(),
            predict=lgb.LGBMClassifier(),
        )

    """
    if columns is not None and not isinstance(columns, (list, tuple)):
        columns = [columns]

    return FilterLowFrequencyTransfomer(columns, min_frequency, other_category, n_max)


class FilterLowFrequencyTransfomer(BaseEstimator, TransformerMixin):
    def __init__(
        self, columns=None, min_frequency=0.02, other_category="other", n_max=None
    ):
        self.columns = columns
        self.min_frequency = min_frequency
        self.other_category = other_category
        self.n_max = n_max

        self._columns = columns
        self._to_keep = {}

    def fit(self, df, y=None):
        if self._columns is None:
            self._columns = find_categorical_columns(df)

        for col in self._columns:
            try:
                to_keep = find_high_frequency_categories(
                    df[col],
                    min_frequency=self._get("min_frequency", col),
                    n_max=self._get("n_max", col),
                )

            except Exception as e:
                raise RuntimeError(
                    f"cannot determine high frequency categories for {col} due to {e}"
                )

            self._to_keep[col] = to_keep

        return self

    def transform(self, df, y=None):
        for col in self._columns:
            df = df.assign(
                **{
                    col: fix_categories(
                        df[col],
                        self._to_keep[col],
                        other_category=self._get("other_category", col),
                    )
                }
            )

        return df

    def _get(self, key, col):
        var = getattr(self, key)
        return var[col] if isinstance(var, dict) else var


def column_transform(*args, **kwargs):
    """Build a transformer for a list of columns.

    Usage::

        pipeline = build_pipeline(
            transform=column_transform(['a', 'b'], np.abs),
            classifier=sk_ensemble.GradientBoostingClassifier(),
        ])

    Or::

        pipeline = build_pipeline(
            transform=column_transform(
                a=np.abs,
                b=op.pos,
            ),
            classifier=sk_ensemble.GradientBoostingClassifier(),
        )

    """
    if not args:
        columns = kwargs

    else:
        columns, func, *args = args

        if not isinstance(columns, (list, tuple)):
            columns = [columns]

        func = ft.partial(func, *args, **kwargs)
        columns = {c: func for c in columns}

    return transform(_column_transform, columns=columns)


def _column_transform(x, columns):
    if not hasattr(x, "assign"):
        raise RuntimeError("can only transform objects with an assign method.")

    for c, func in columns.items():
        x = x.assign(**{c: func(x[c])})

    return x


def build_pipeline(**kwargs):
    """Build a pipeline from named steps.

    The order of the keyword arguments is retained. Note, this functionality
    requires python ``>= 3.6``.

    Usage::

        pipeline = build_pipeline(
            transform=...,
            predict=...,
        )

    """
    import sklearn.pipeline as sk_pipeline

    if sys.version_info[:2] < (3, 6):
        raise RuntimeError("pipeline factory requires deterministic kwarg order")

    return sk_pipeline.Pipeline(list(kwargs.items()))


def transform(*args, **kwargs):
    """Build a function transformer with args / kwargs bound.

    Usage::

        pipeline = build_pipeline(
            transform=transform(np.abs)),
            classifier=sk_ensemble.GradientBoostingClassifier()),
        )
    """
    func, *args = args
    return FuncTransformer(ft.partial(func, *args, **kwargs))


class FuncTransformer(TransformerMixin, BaseEstimator):
    """Simple **non-validating** function transformer.

    :param callable func:
        the function to apply on transform
    """

    def __init__(self, func):
        self.func = func

    def fit(self, x, y=None):
        return self

    def partial_fit(self, x, y=None):
        return self

    def transform(self, x):
        return self.func(x)


class FuncClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, func):
        self.func = func

    def fit(self, df, y=None):
        return self

    def predict_proba(self, df):
        return self.func(df)

    def predict(self, df):
        import numpy as np

        return np.argmax(self.predict_proba(df), axis=1)


class FuncRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, func):
        self.func = func

    def fit(self, df, y=None):
        return self

    def predict(self, df):
        return self.func(df)


class DataFrameEstimator(BaseEstimator):
    """Add support for dataframe use to sklearn estimators.
    """

    def __init__(self, est):
        self.est = est

    def fit(self, x, y=None, **kwargs):
        import numpy as np

        x = x.reset_index(drop=True)
        y = np.asarray(x[y])

        self.est.fit(x, y, **kwargs)
        return self

    def predict(self, x, y=None):
        x = x.reset_index(drop=True)
        return self.est.predict(x)

    def predict_proba(self, x, y=None):
        x = x.reset_index(drop=True)
        return self.est.predict_proba(x)


@get_children.register(DataFrameEstimator)
def df_estimator(est):
    return [(0, est.est)]


class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.columns_ = columns
        self.levels_ = collections.OrderedDict()

    def fit(self, x, y=None):
        if self.columns_ is None:
            self.columns_ = find_categorical_columns(x)

        for col in self.columns_:
            try:
                self.levels_[col] = multi_type_sorted(x[col].unique())

            except Exception as e:
                raise RuntimeError(f"cannot fit {col}") from e

        return self

    def transform(self, x, y=None):
        for col in self.columns_:
            try:
                assignments = {}
                for level in self.levels_[col]:
                    assignments[f"{col}_{level}"] = (x[col] == level).astype(float)

                x = x.drop([col], axis=1).assign(**assignments)

            except Exception as e:
                raise RuntimeError(f"cannot transform {col}") from e

        return x


def multi_type_sorted(vals):
    import pandas as pd

    return sorted(
        vals, key=lambda v: (type(v).__module__, type(v).__name__, pd.isnull(v), v)
    )


class FitInfo(BaseEstimator, TransformerMixin):
    """Extract and store meta data of the dataframe passed to fit.
    """

    def __init__(self, extractor, target=None):
        self.extractor = extractor
        self.target = target

        if target is None:
            self.meta_ = {}

        else:
            self.meta_ = target

    def fit(self, x, y=None):
        self.meta_.update(self.extractor(x))
        return self

    def transform(self, x, y=None):
        return x


try:
    import sklearn.pipeline as sk_pipeline

    @get_children.register(sk_pipeline.Pipeline)
    def pipeline_get_children(est):
        return est.steps


except ImportError:
    pass


def search_estimator(predicate, est, key=()):
    return list(_search_estimator(predicate, key, est))


def _search_estimator(predicate, key, est):
    if predicate(key, est):
        yield key, est

    for child_key, child in get_children(est):
        yield from _search_estimator(predicate, key + (child_key,), child)


def waterfall(
    obj,
    col=None,
    base=None,
    total=False,
    end_annot=None,
    end_fmt=".g",
    annot=False,
    fmt="+.2g",
    cmap="coolwarm",
    xmin=0,
    total_kwargs=None,
    annot_kwargs=None,
    **kwargs,
):
    """Plot a waterfall chart.

    Usage::

        series.pipe(waterfall, annot='top', fmt='+.1f', total=True)

    """
    import matplotlib.pyplot as plt
    import numpy as np

    if len(obj.shape) == 2 and col is None:
        raise ValueError("need a column with 2d objects")

    if col is not None:
        top = obj[col] if not callable(col) else col(obj)

    else:
        top = obj

    if base is not None:
        bottom = obj[base] if not callable(base) else base(obj)

    else:
        bottom = top.shift(1).fillna(0)

    if annot is True:
        annot = "top"

    if total_kwargs is None:
        total_kwargs = {}

    if annot_kwargs is None:
        annot_kwargs = {}

    if end_annot is None:
        end_annot = annot is not False

    total_kwargs = {"color": (0.5, 0.75, 0.5), **total_kwargs}

    if annot == "top":
        annot_kwargs = {"va": "bottom", "ha": "center", **annot_kwargs}
        annot_y = np.maximum(top, bottom)
        total_y = max(top.iloc[-1], 0)

    elif annot == "bottom":
        annot_kwargs = {"va": "bottom", "ha": "center", **annot_kwargs}
        annot_y = np.minimum(top, bottom)
        total_y = min(top.iloc[-1], 0)

    elif annot == "center":
        annot_kwargs = {"va": "center", "ha": "center", **annot_kwargs}
        annot_y = 0.5 * (top + bottom)
        total_y = 0.5 * top.iloc[-1]

    elif annot is not False:
        raise ValueError(f"Cannot annotate with {annot}")

    height = top - bottom

    kwargs = {"color": colormap(height, cmap=cmap, center=True), **kwargs}
    plt.bar(xmin + np.arange(len(height)), height, bottom=bottom, **kwargs)

    if annot is not False:
        for x, y, v in zip(it.count(xmin), annot_y, height):
            if x == xmin:
                continue

            plt.text(x, y, ("%" + fmt) % v, **annot_kwargs)

    if end_annot is not False:
        plt.text(xmin, annot_y.iloc[0], ("%" + end_fmt) % top.iloc[0], **annot_kwargs)

        if total:
            plt.text(
                xmin + len(annot_y),
                total_y,
                ("%" + end_fmt) % top.iloc[-1],
                **annot_kwargs,
            )

    for idx, p in zip(it.count(xmin), bottom):
        if idx == xmin:
            continue

        plt.plot([idx - 1 - 0.4, idx + 0.4], [p, p], ls="--", color="0.5")

    plt.xticks(xmin + np.arange(len(height)), list(height.index))

    if total:
        plt.bar([xmin + len(bottom)], [top.iloc[-1]], **total_kwargs)
        plt.plot(
            [xmin + len(bottom) - 1 - 0.4, xmin + len(bottom) + 0.4],
            [top.iloc[-1], top.iloc[-1]],
            ls="--",
            color="0.5",
        )


def colormap(x, cmap="coolwarm", center=True, vmin=None, vmax=None, norm=None):
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    x = np.asarray(x)

    if norm is None:
        norm = colors.NoNorm()

    if vmin is None:
        vmin = np.min(x)

    if vmax is None:
        vmax = np.max(x)

    if center:
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax

    x = norm(x)
    x = np.clip((x - vmin) / (vmax - vmin), 0, 1)

    return cm.get_cmap(cmap)(x)


def bar(s, cmap="viridis", color=None, norm=None, orientation="vertical"):
    import matplotlib.colors
    import matplotlib.pyplot as plt

    if norm is None:
        norm = matplotlib.colors.NoNorm()

    if color is None:
        color = colormap(s, cmap=cmap, norm=norm)

    indices = range(len(s))

    if orientation == "vertical":
        plt.bar(indices, s, color=color)
        plt.xticks(indices, s.index)

    else:
        plt.barh(indices, s, color=color)
        plt.yticks(indices, s.index)


# TODO: make sureit can be called with a dataframe
def qplot(
    x=None,
    y=None,
    data=None,
    alpha=1.0,
    fill_alpha=0.8,
    color=None,
    ax=None,
    **line_kwargs,
):
    import numpy as np
    import matplotlib.pyplot as plt

    if y is None and x is not None:
        x, y = y, x

    if y is None:
        raise ValueError("need data to plot")

    if isinstance(y, tuple) or (isinstance(y, np.ndarray) and y.ndim == 2):
        y = tuple(y)

    else:
        y = (y,)

    if data is not None:
        y = tuple(data[c] for c in y)

    # TODO: use index if data a dataframe
    if x is None:
        x = np.arange(len(y[0]))

    elif data is not None:
        x = data[x]

    if ax is None:
        ax = plt.gca()

    if color is None:
        color = ax._get_lines.get_next_color()

    n = len(y) // 2
    fill_alpha = (1 / n) if fill_alpha is None else (fill_alpha / n)

    for i in range(n):
        plt.fill_between(x, y[i], y[-(i + 1)], alpha=fill_alpha * alpha, color=color)

    if len(y) % 2 == 1:
        plt.plot(x, y[n], alpha=alpha, color=color, **line_kwargs)


def render_poyo(obj, params):
    """Lighweight POYO templating.

    Any callable in the tree will be called with params. Example::

        template = {
            "key": lambda params: params['value'],
        }

        render_poyo(template, {'value': 20})

    """
    return sapply(lambda o: o if not callable(o) else o(params), obj)


def dashcb(app, output, *inputs, figure=False):
    """Construct a dash callback using function annotations.

    :param dash.Dash app:
        the dash app to build the callback for

    :param str output:
        the output, as a string of the form ``{component}:{property}``

    :param str inputs:
        the inputs, as strings of the form ``{component}:{property}``

    :param bool figure:
        if True, the current matplotlib figure will be captured and returned as
        a data URL. This allows to use matplotlib with dash. See the examples
        below

    Consider the following dash callback::


        @app.callback(dash.dependencies.Output('display', 'children'),
                      [dash.dependencies.Input('dropdown', 'value')])
        def update_display(value):
            return 'Selected: "{}"'.format(value)

    With dashcb, it can be written as::

        @dashcb(app, 'display:children', 'dropdown:value')
        def update_display(value):
            return 'Selected: "{}"'.format(value)

    To use dash with matplotlib figure, define an ``html.Img`` element. For
    example with id ``my_image``. Then the plot can be updated via::

        @dashcb(app, 'my_image:src', 'dropdown:value', figure=True)
        def update_display(value):
            plt.plot([1, 2, 3])
            plt.title(value)

    """
    import dash.dependencies

    def decorator(func):
        dash_inputs = [
            _dash_cb_parse_annotation(dash.dependencies.Input, arg) for arg in inputs
        ]
        dash_output = _dash_cb_parse_annotation(dash.dependencies.Output, output)

        if figure:
            return app.callback(dash_output, dash_inputs)(dashmpl(func))

        return app.callback(dash_output, dash_inputs)(func)

    return decorator


def dashmpl(func):
    """Capture the current matplotlib figure.
    """
    import matplotlib.pyplot as plt

    @ft.wraps(func)
    def impl(*args, **kwargs):
        func(*args, **kwargs)

        img = io.BytesIO()
        plt.savefig(img, format="png")
        plt.close()

        img = base64.b64encode(img.getvalue())
        img = img.decode("ascii")
        return "data:image/png;base64," + img

    return impl


def _dash_cb_parse_annotation(cls, s):
    element, _, property = s.partition(":")
    return cls(element, property)


def expand(low, high, change=0.05):
    center = 0.5 * (low + high)
    delta = 0.5 * (high - low)
    return (center - (1 + 0.5 * change) * delta, center + (1 + 0.5 * change) * delta)


# ########################################################################## #
#                             I/O Methods                                    #
# ########################################################################## #


def magic_open(p, mode, *, compression=None, atomic=False):
    # return file-like objects unchanged
    if not isinstance(p, (pathlib.Path, str)):
        return p

    assert atomic is False, "Atomic operations not yet supported"
    opener = _get_opener(p, compression)
    return opener(p, mode)


def _get_opener(p, compression):
    if compression is None:
        sp = str(p)

        if sp.endswith(".bz2"):
            compression = "bz2"

        elif sp.endswith(".gz"):
            compression = "gz"

        else:
            compression = "none"

    openers = {"bz2": bz2.open, "gz": gzip.open, "gzip": gzip.open, "none": open}
    return openers[compression]


def write_json(
    obj, p, *, compression=None, atomic=False, lines=False, json=json, **kwargs
):
    with magic_open(p, "wt", compression=compression, atomic=atomic) as fobj:
        if not lines:
            json.dump(obj, fobj, **kwargs)

        else:
            for item in obj:
                fobj.write(json.dumps(item))
                fobj.write("\n")


def read_json(p, *, lines=False, compression=None, json=json):
    with magic_open(p, "rt", compression=compression) as fobj:
        if not lines:
            return json.load(fobj)

        else:
            return [json.loads(l) for l in fobj]


# ########################################################################## #
#                               Looping                                      #
# ########################################################################## #


status_characters = ["\u25AB", " "] + [
    chr(ord("\u2800") + v) for v in it.accumulate([64, 128, 4, 32, 2, 16, 1, 8])
]

running_characters = ["-", "\\", "|", "/"]

current_loop: Optional["Loop"] = None
current_label = None


def loop_over(iterable, label: Union[str, Callable[[], str]] = None, keep=False):
    """Simplified interface to Loop.over.

    :param label:
        if a callable, it should return a str that is used as the loop label.
    """
    global current_loop, current_label
    if label is not None:
        current_label = label

    for current_loop, item in Loop.over(iterable):
        yield item

        assert current_loop is not None
        current_loop.print(lambda: "{}{}".format(current_loop, get_current_label()))

    assert current_loop is not None
    current_loop.print(
        "{}{}".format(current_loop, get_current_label()),
        force=True,
        end="\n" if keep else "\r",
    )
    current_loop = None


def loop_nest(iterable, label: Union[str, Callable[[], str]] = None):
    global current_loop, current_label
    if current_loop is None:
        raise RuntimeError("Can only nest within an existing loop")

    if label is not None:
        current_label = label

    for item in current_loop.nest(iterable):
        yield item
        current_loop.print(lambda: "{}{}".format(current_loop, get_current_label()))


def get_current_label():
    if current_label is None:
        return ""

    if callable(current_label):
        return " {}".format(current_label())

    return " {}".format(current_label)


class LoopState(enum.Enum):
    pending = "pending"
    running = "running"
    done = "done"
    aborted = "aborted"


class LoopPrintDispatch:
    def __get__(self, instance, owner):
        if instance is None:
            return owner._static_print

        else:
            return instance._print


class Debouncer:
    def __init__(self, interval, *, now=time.time):
        self.last_invocation = 0
        self.interval = interval
        self.now = now

    def should_run(self, now=None):
        if self.interval is False:
            return True

        if now is None:
            now = self.now()

        return now > self.last_invocation + self.interval

    def invoked(self, now=None):
        if now is None:
            now = self.now()

        self.last_invocation = now


class Loop:
    """A flexible progressbar indicator.

    It's designed to make printing custom messages and customizing the loop
    style easy.

    The following format codes are recognized:

    * ``[``: in the beginning, indicates that the progress bar will be
      surrounded by brackets.
    * ``-``: in the beginning, indicates that the parts of the progress bar
      will be printed without automatic spaces
    * ``B``: a one character bar
    * ``b``: the full bar
    * ``t``: the total runtime so far
    * ``e``: the expected total runtime
    * ``r``: the expected remaining runtime
    * ``f``: the fraction of work performed so far
    * additional characters will be included verbatim

    To access nested loop use the getitem notation, e.g. ``loop[1]``.

    """

    @classmethod
    def range(cls, *range_args, time=time.time, debounce=0.1):
        return cls.over(range(*range_args), time=time, debounce=debounce)

    @classmethod
    def over(cls, iterable, length=None, time=time.time, debounce=0.1):
        loop = cls(time=time, debounce=debounce)

        for item in loop.nest(iterable, length):
            yield loop, item

    print = LoopPrintDispatch()

    @staticmethod
    def _static_print(
        str: Union[str, Callable[[], str]],
        width=120,
        end="\r",
        file=None,
        flush=False,
        force=False,
    ):
        if callable(str):
            str = str()

        print(str.ljust(width)[:width], end=end, file=file, flush=flush)

    def _print(
        self, str: str, width=120, end="\r", file=None, flush=False, force=False
    ):
        now = self.now()
        if not force and not self.debouncer.should_run(now=now):
            return

        self.debouncer.invoked(now=now)
        self._static_print(str, width=width, end=end, file=file, flush=flush)

    def will_print(self, now=None):
        """Check whether the print invocation will be debounced."""
        return self.debouncer.should_run(now)

    def __init__(self, time=time.time, stack=None, root=None, debounce=0.1):
        if stack is None:
            stack = []

        self.now = time
        self._stack = stack
        self._root = root
        self._last_print = 0
        self.debouncer = Debouncer(debounce)

    def __getitem__(self, idx):
        return Loop(time=self.now, stack=self._stack[idx:], root=self._stack[idx])

    def push(self, length=None, idx=0):
        frame = LoopFrame(start=self.now(), idx=idx, length=length)

        self._stack.append(frame)
        if self._root is None:
            self._root = frame

        return frame

    def pop(self, frame):
        if frame.state not in {LoopState.aborted, LoopState.done}:
            frame.finish()

        self._stack = [s for s in self._stack if s is not frame]

    def nest(self, iterable, length=None):
        if length is None:
            try:
                length = len(iterable)

            except TypeError:
                pass

        frame = self.push(length=length)

        for item in iterable:
            try:
                yield item

            # NOTE: this is reached, when the generator is not fully consumed
            except GeneratorExit:
                frame.abort()
                self.pop(frame)
                raise

            frame.finish_item()

        self.pop(frame)

    def get_info(self):
        now = self.now()

        info = dict(
            fraction=self._get_fraction(self._stack),
            total=now - self._root.start,
            state=self._root.state,
            idx=self._root.idx,
        )

        if info["fraction"] is not None:
            info["expected"] = info["total"] / info["fraction"]

        else:
            info["expected"] = None

        return info

    @classmethod
    def _get_fraction(cls, stack):
        if not stack:
            return 1

        root, *stack = stack

        if root.length is None:
            return None

        return min(1.0, (root.idx + cls._get_fraction(stack)) / root.length)

    def __str__(self):
        return format(self)

    def __format__(self, format_spec):
        status = self.get_info()

        if status["state"] is LoopState.pending:
            return "[pending]"

        elif status["state"] is LoopState.aborted:
            return f'[aborted. took {tdformat(status["total"])}]'

        elif status["state"] is LoopState.done:
            return f'[done. took {tdformat(status["total"])}]'

        elif status["state"] is not LoopState.running:
            raise RuntimeError("unknown state")

        if not format_spec:
            format_spec = "[bt/e"

        if format_spec[:1] == "[":
            outer = "[", "]"
            format_spec = format_spec[1:]

        else:
            outer = "", ""

        if format_spec[:1] == "-":
            join_char = ""
            format_spec = format_spec[1:]

        else:
            join_char = " "

        result = [self._loop_formats.get(c, lambda _: c)(status) for c in format_spec]
        return outer[0] + join_char.join(result) + outer[1]

    _loop_formats = {
        "B": lambda status: loop_bar(status, n=1),
        "b": lambda status: loop_bar(status),
        "t": lambda status: tdformat(status["total"]),
        "e": lambda status: tdformat(status["expected"]),
        "r": lambda status: tdformat(status["expected"] - status["total"]),
        "f": lambda status: f"{status['fraction']:.1%}",
    }


class LoopFrame:
    def __init__(self, start, idx=0, length=None, state=LoopState.running):
        if length is not None:
            length = int(length)

        self.start = start
        self.idx = idx
        self.length = length
        self.state = state

    def copy(self):
        return LoopFrame(
            start=self.start, idx=self.idx, length=self.length, state=self.state
        )

    def finish_item(self):
        self.idx += 1

    def abort(self):
        self.state = LoopState.aborted

    def finish(self):
        self.state = LoopState.done

    def __repr__(self):
        return "LoopFrame(...) <state={!r}, start={!r}>".format(self.state, self.start)


def tdformat(time_delta):
    """Format a timedelta given in seconds.
    """
    if time_delta is None:
        return "?"

    # TODO: handle negative differences?
    time_delta = abs(time_delta)

    d = dict(
        weeks=int(time_delta // (7 * 24 * 60 * 60)),
        days=int(time_delta % (7 * 24 * 60 * 60) // (24 * 60 * 60)),
        hours=int(time_delta % (24 * 60 * 60) // (60 * 60)),
        minutes=int(time_delta % (60 * 60) // 60),
        seconds=time_delta % 60,
    )

    if d["weeks"] > 0:
        return "{weeks}w {days}d".format(**d)

    elif d["days"] > 0:
        return "{days}d {hours}h".format(**d)

    elif d["hours"] > 0:
        return "{hours}h {minutes}m".format(**d)

    elif d["minutes"] > 0:
        return "{minutes}m {seconds:.0f}s".format(**d)

    else:
        return "{seconds:.2f}s".format(**d)


def loop_bar(status, n=10):
    if status["fraction"] is not None:
        return ascii_bar(status["fraction"], n=n)

    return running_characters[status["idx"] % len(running_characters)]


def ascii_bar(u, n=10):
    """Format a ASCII progressbar"""
    u = max(0.00, min(0.99, u))

    done = int((n * u) // 1)
    rest = max(0, n - done - 1)

    c = int(((n * u) % 1) * len(status_characters))
    return (
        status_characters[-1] * done
        + status_characters[c]
        + status_characters[0] * rest
    )


# ###################################################################### #
# #                                                                    # #
# #                 Deterministic Random Number Generation             # #
# #                                                                    # #
# ###################################################################### #

maximum_15_digit_hex = float(0xFFF_FFFF_FFFF_FFFF)
max_32_bit_integer = 0xFFFF_FFFF


def sha1(obj):
    """Create a hash for a json-encode-able object
    """
    return int(str_sha1(obj)[:15], 16)


def str_sha1(obj):
    s = json.dumps(obj, indent=None, sort_keys=True, separators=(",", ":"))
    s = s.encode("utf8")
    return hashlib.sha1(s).hexdigest()


def random(obj):
    """Return a random float in the range [0, 1)"""
    return min(sha1(obj) / maximum_15_digit_hex, 0.999_999_999_999_999_9)


def uniform(obj, a, b):
    return a + (b - a) * random(obj)


def randrange(obj, *range_args):
    r = range(*range_args)
    # works up to a len of 9007199254749999, rounds down afterwards
    i = int(random(obj) * len(r))
    return r[i]


def randint(obj, a, b):
    return randrange(obj, a, b + 1)


def np_seed(obj):
    """Return a seed usable by numpy.
    """
    return [randrange((obj, i), max_32_bit_integer) for i in range(10)]


def tf_seed(obj):
    """Return a seed usable by tensorflow.
    """
    return randrange(obj, max_32_bit_integer)


def std_seed(obj):
    """Return a seed usable by python random module.
    """
    return str_sha1(obj)


def shuffled(obj, l):
    l = list(l)
    shuffle(obj, l)
    return l


def shuffle(obj, l):
    """Shuffle ``l`` in place using Fisher–Yates algorithm.

    See: https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
    """
    n = len(l)
    for i in range(n - 1):
        j = randrange((obj, i), i, n)
        l[i], l[j] = l[j], l[i]


# ########################################################################### #
#                                                                             #
#                     Helper for datetime handling in pandas                  #
#                                                                             #
# ########################################################################### #
def timeshift_index(obj, dt):
    """Return a shallow copy of ``obj`` with its datetime index shifted by ``dt``."""
    obj = obj.copy(deep=False)
    obj.index = obj.index + dt
    return obj


def to_start_of_day(s):
    """Return the start of the day for the datetime given in ``s``."""
    import pandas as pd

    s = pd.to_datetime({"year": s.dt.year, "month": s.dt.month, "day": s.dt.day})
    s = pd.Series(s)
    return s


def to_time_in_day(s, unit=None):
    """Return the timediff relative to the start of the day of ``s``."""
    import pandas as pd

    s = s - to_start_of_day(s)
    return s if unit is None else s / pd.to_timedelta(1, unit=unit)


def to_start_of_week(s):
    """Return the start of the week for the datetime given ``s``."""
    s = to_start_of_day(s)
    return s - s.dt.dayofweek * datetime.timedelta(days=1)


def to_time_in_week(s, unit=None):
    """Return the timedelta relative to weekstart for the datetime given in ``s``.
    """
    import pandas as pd

    s = s - to_start_of_week(s)
    return s if unit is None else s / pd.to_timedelta(1, unit=unit)


def to_start_of_year(s):
    """Return the start of the year for the datetime given in ``s``."""
    import pandas as pd

    s = pd.to_datetime({"year": s.dt.year, "month": 1, "day": 1})
    s = pd.Series(s)
    return s


def to_time_in_year(s, unit=None):
    """Return the timediff relative to the start of the year for ``s``."""
    import pandas as pd

    s = s - to_start_of_year(s)
    return s if unit is None else s / pd.to_timedelta(1, unit=unit)
