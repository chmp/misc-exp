"""Helpers for data science.

Distributed as part of ``https://github.com/chmp/misc-exp`` under the MIT
license, (c) 2017 Christopher Prohm.
"""
import base64
import collections
import enum
import functools as ft
import hashlib
import importlib
import io
import itertools as it
import json
import math
import os.path
import pickle
import sys
import time

try:
    from sklearn.base import (
        BaseEstimator,
        TransformerMixin,
        ClassifierMixin,
        RegressorMixin,
    )

except ImportError:
    _HAS_SK_LEARN = False

    class BaseEstimator:
        pass

    class TransformerMixin:
        pass

    class RegressorMixin:
        pass

    class ClassifierMixin:
        pass


else:
    _HAS_SK_LEARN = True


try:
    from daft import PGM

except ImportError:

    class PGM:
        def __init__(self, *args, **kwargs):
            pass

    _HAS_DAFT = False

else:
    _HAS_DAFT = True


def reload(*modules_or_module_names):
    mod = None
    for module_or_module_name in modules_or_module_names:
        if isinstance(module_or_module_name, str):
            module_or_module_name = importlib.import_module(module_or_module_name)

        mod = importlib.reload(module_or_module_name)

    return mod


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


def cached(path):
    """Similar to ``define``, but cache to a file.
    """

    def decorator(func):
        if os.path.exists(path):
            print("load cache", path)
            with open(path, "rb") as fobj:
                return pickle.load(fobj)

        else:
            print("compute")
            result = func()

            print("save cache", path)
            with open(path, "wb") as fobj:
                pickle.dump(result, fobj)

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


def colorize(items):
    cycle = get_color_cycle()
    return zip(it.cycle(cycle), items)


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
        return cycle

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
    xformatter=None,
    yformatter=None,
    left=None,
    top=None,
    bottom=None,
    right=None,
    wspace=None,
    hspace=None,
    subplot=None,
    legend=None,
    colorbar=None,
    invert=None,
    ax=None,
    grid=None,
    axis=None,
):
    """Set various style related options of MPL.

    :param Optional[callable] xformatter:
        if given a formatter for the major x ticks. Should have the
        signature ``(x_value, pos) -> label``.

    :param Optional[callable] yformatter:
        See ``xformatter``.
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


def path(x, y, close=True, **kwargs):
    """Plot a path given as a list of vertices.

    Usage::

        path([0, 1, 0], [0, 1, 1], facecolor='r')

    """
    import numpy as np

    from matplotlib import pyplot as plt
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    vertices = np.stack([np.asarray(x), np.asarray(y)], axis=1)

    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 1)
    if close:
        codes += [Path.CLOSEPOLY]
        vertices = np.concatenate([vertices, [vertices[0]]])

    p = Path(vertices, codes)
    p = PathPatch(p, **kwargs)

    plt.gca().add_patch(p)


def axtext(*args, **kwargs):
    """Add a text in axes coordinates (similar ``figtext``).

    Usage::

        axtext(0, 0, 'text')

    """
    import matplotlib.pyplot as plt

    kwargs.update(transform=plt.gca().transAxes)
    plt.text(*args, **kwargs)


def plot_gaussian_contour(df, x, y, *, q=(0.99,), ax=None, **kwargs):
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

    x = np.asarray(df[x])
    y = np.asarray(df[y])

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
        s = (2 ** 0.5) * scipy.special.erfinv(q)
        artist = mpl.patches.Ellipse((mx, my), *(s * eigvals), angle, **kwargs)
        plt.gca().add_artist(artist)


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


def pd_has_ordered_assign():
    import pandas as pd

    py_major, py_minor, *_ = sys.version_info
    pd_major, pd_minor, *_ = pd.__version__.split(".")
    pd_major = int(pd_major)
    pd_minor = int(pd_minor)

    return (py_major, py_minor) >= (3, 6) and (pd_major, pd_minor) >= (0, 23)


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
                    f"cannot determine high frequency categories for {col}"
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
#                               Looping                                      #
# ########################################################################## #


status_characters = it.accumulate([64, 128, 4, 32, 2, 16, 1, 8])
status_characters = [chr(ord("\u2800") + v) for v in status_characters]
status_characters = ["\u25AB", " "] + status_characters

running_characters = ["-", "\\", "|", "/"]

current_loop = None
current_label = None


def loop_over(iterable, label=None):
    global current_loop, current_label
    if label is not None:
        current_label = label

    for current_loop, item in Loop.over(iterable):
        yield item

        if current_loop.will_print():
            label = " {}".format(current_label) if current_label is not None else ""
            current_loop.print("{}{}".format(current_loop, label))


def loop_nest(iterable, label=None):
    global current_loop, current_label
    if label is not None:
        current_label = label

    for item in current_loop.nest(iterable):
        yield item

        if current_loop.will_print():
            label = " {}".format(current_label) if current_label is not None else ""
            current_loop.print("{}{}".format(current_loop, label))


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
    def _static_print(str: str, width=120, end="\r", file=None, flush=False):
        print(str.ljust(width)[:width], end=end, file=file, flush=flush)

    def _print(self, str: str, width=120, end="\r", file=None, flush=False):
        now = self.now()
        if not self.debouncer.should_run(now=now):
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

    def nest(self, iterable, length=None):
        frame = LoopFrame(self.now(), iterable, 0, length)

        self._stack.append(frame)
        if self._root is None:
            self._root = frame

        for item in frame.iterable:
            try:
                yield item

            # NOTE: this is reached, when the generator is not fully consumed
            except GeneratorExit:
                frame.abort()
                raise

            frame.finish_item()

        frame.finish()
        self._stack.pop()

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
    def __init__(self, start, iterable, idx=0, length=None, state=LoopState.running):
        if length is None:
            try:
                length = len(iterable)

            except TypeError:
                length = None

        else:
            length = int(length)

        self.start = start
        self.iterable = iterable
        self.idx = idx
        self.length = length
        self.state = state

    def copy(self):
        return LoopFrame(
            start=self.start,
            iterable=self.iterable,
            idx=self.idx,
            length=self.length,
            state=self.state,
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
    return min(sha1(obj) / maximum_15_digit_hex, 0.9999999999999999)


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
