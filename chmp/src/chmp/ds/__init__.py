"""Helper for data science.

Distributed as part of ``https://github.com/chmp/misc-exp`` under the MIT
license, (c) 2017 Christopher Prohm.
"""
import functools as ft
import importlib
import itertools as it
import warnings

try:
    from sklearn.base import BaseEstimator, TransformerMixin

except ImportError:
    _HAS_SK_LEARN = False
    BaseEstimator = TransformerMixin = object

else:
    _HAS_SK_LEARN = True



def notebook_preamble():
    """Add common code
    """
    from IPython import get_ipython

    get_ipython().set_next_input(_notebook_preamble, replace=True)


_notebook_preamble = '''# from chmp.ds import notebook_preamble; notebook_preamble()

%matplotlib inline
# disable rescaling the figure, to gain tighter control over the result
%config InlineBackend.print_figure_kwargs = {'bbox_inches':None}

import logging
logging.basicConfig(level=logging.INFO)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
'''.strip()


def reload(module_or_module_name):
    if isinstance(module_or_module_name, str):
        module_or_module_name = importlib.import_module(module_or_module_name)

    return importlib.reload(module_or_module_name)


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


def get_color_cycle(n=None):
    """Return the matplotlib color cycle.

    :param Optional[int] n:
        if given, return a list with exactly n elements formed by repeating
        the color cycle as necessary.

    Usage::

        blue, green, red = get_color_cycle(3)

    """
    import matplotlib as mpl

    cycle = mpl.rcParams['axes.prop_cycle'].by_key()['color']

    if n is None:
        return cycle

    return list(it.islice(it.cycle(cycle), n))


def mpl_set(
        box=None,
        xlabel=None, ylabel=None,
        title=None, suptitle=None,
        xscale=None, yscale=None, caption=None,
        xlim=None, ylim=None,
        xticks=None, yticks=None,
        left=None, top=None, bottom=None, right=None, wspace=None, hspace=None,
        subplot=None,
        legend=None,
        colorbar=None,
):
    """Set various style related options of MPL.
    """
    import matplotlib.pyplot as plt

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
        plt.yscale(xscale)

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

    if caption is not None:
        _caption(caption)

    subplot_kwargs = _dict_of_optionals(
        left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace
    )

    if subplot_kwargs:
        plt.subplots_adjust(**subplot_kwargs)

    if legend is not None and legend is not False:
        if legend is True:
            plt.legend(loc='best')

        else:
            plt.legend(**legend)

    if subplot is not None:
        plt.sca(ax)

    if colorbar is True:
        plt.colorbar()


def edges(x):
    """Create edges for use with pcolor.

    Usage::

        assert x.size == v.shape[1]
        assert y.size == v.shape[0]
        pcolor(edges(x), edges(y), v)

    """
    centers = 0.5 * (x[1:] + x[:-1])
    return np.concatenate((
        [x[0] - 0.5 * (x[1] - x[0])],
        centers,
        [x[-1] + 0.5 * (x[-1] - x[-2])]
    ))


def caption(s, size=13, strip=True):
    """Add captions to matplotlib graphs."""
    import matplotlib.pyplot as plt

    if strip:
        s = s.splitlines()
        s = (i.strip() for i in s)
        s = (i for i in s if i)
        s = ' '.join(s)

    plt.figtext(0.5, 0, s, wrap=True, size=size, va='bottom', ha='center')


_caption = caption


def change_vspan(
        x, y, *,
        data=None,
        color=('w', '0.90'),
        transform_x=None,
        transform_y=None,
        skip_nan=True,
        **kwargs
):
    """Plot changes in a quantity with vspans.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    x, y = _prepare_xy(
        x, y,
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
        x, y, *,
        data=None,
        transform_x=None,
        transform_y=None,
        skip_nan=True,
        **kwargs
):
    """Plot changes in a quantity with pyplot's standard plot function.
    """
    import matplotlib.pyplot as plt
    x, y = _prepare_xy(
        x, y,
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


def fix_categories(s, categories=None, other_category=None, inplace=False, groups=None, ordered=False):
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
        included in the categories, it is added.

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
            warnings.warn('cannot change the type inplace')
        s = s.astype('category', )

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

    if set(removals) - set(remapped):
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

    :param float min_min_frequency:
        the minimum frequency to keep

    :param Optional[int] n_max:
        if given keep at most ``n_max`` categories. If more are present after
        filtering for minimum frequency, keep the highest ``n_max`` frequency
        columns.
    """
    s = (
        s
        .value_counts(normalize=True)
        .pipe(lambda s: s[s > min_frequency])
    )

    if n_max is None:
        return list(s.index)

    if len(s) <= n_max:
        return s

    return list(
        s
        .sort_values(ascending=False)
        .iloc[:n_max]
        .index
    )


def as_frame(**kwargs):
    import pandas as pd
    return pd.DataFrame(kwargs)


def column_transform(*args, **kwargs):
    """Build a transformer for a list of columns.

    Usage::

        pipeline = sk_pipeline.Pipeline([
            ('transform', column_transform(['a', 'b'], np.abs)),
            ('classifier', sk_ensemble.GradientBoostingClassifier()),
        ])

    """
    columns, func, *args = args

    if not isinstance(columns, (list, tuple)):
        columns = [columns]

    func = ft.partial(
        _column_transform,
        columns=columns, func=func, args=args, kwargs=kwargs,
    )

    return FuncTransformer(func)


def _column_transform(x, columns, func, args, kwargs):
    if not hasattr(x, 'assign'):
        raise RuntimeError('can only transform objects with an assign method.')

    for c in columns:
        x = x.assign(**{c: func(x[c], *args, **kwargs)})

    return x


def transform(*args, **kwargs):
    """Build a function transformer with args / kwargs bound.

    Usage::

        pipeline = sk_pipeline.Pipeline([
            ('transform', transform(np.abs)),
            ('classifier', sk_ensemble.GradientBoostingClassifier()),
        ])
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


def waterfall(
        obj,
        col=None, base=None, total=False,
        end_annot=None, end_fmt='.g',
        annot=False, fmt='+.2g',
        cmap='coolwarm',
        xmin=0,
        total_kwargs=None, annot_kwargs=None, **kwargs
):
    """Plot a waterfall chart.

    Usage::

        series.pipe(waterfall, annot='top', fmt='+.1f', total=True)

    """
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import numpy as np

    if len(obj.shape) == 2 and col is None:
        raise ValueError('need a column with 2d objects')

    if col is not None:
        top = obj[col] if not callable(col) else col(obj)

    else:
        top = obj

    if base is not None:
        bottom = obj[base] if not callable(base) else base(obj)

    else:
        bottom = top.shift(1).fillna(0)

    if annot is True:
        annot = 'top'

    if total_kwargs is None:
        total_kwargs = {}

    if annot_kwargs is None:
        annot_kwargs = {}

    if end_annot is None:
        end_annot = annot is not False

    total_kwargs = {'color': (0.5, 0.75, 0.5), **total_kwargs}

    if annot == 'top':
        annot_kwargs = {'va': 'bottom', 'ha': 'center', **annot_kwargs}
        annot_y = np.maximum(top, bottom)
        total_y = max(top.iloc[-1], 0)

    elif annot == 'bottom':
        annot_kwargs = {'va': 'bottom', 'ha': 'center', **annot_kwargs}
        annot_y = np.minimum(top, bottom)
        total_y = min(top.iloc[-1], 0)

    elif annot == 'center':
        annot_kwargs = {'va': 'center', 'ha': 'center', **annot_kwargs}
        annot_y = 0.5 * (top + bottom)
        total_y = 0.5 * top.iloc[-1]

    elif annot is not False:
        raise ValueError(f'Cannot annotate with {annot}')

    height = top - bottom

    kwargs = {'color': colormap(height, cmap=cmap, center=True), **kwargs}
    plt.bar(xmin + np.arange(len(height)), height, bottom=bottom, **kwargs)

    if annot is not False:
        for x, y, v in zip(it.count(xmin), annot_y, height):
            if x == xmin:
                continue

            plt.text(x, y, ('%' + fmt) % v, **annot_kwargs)

    if end_annot is not False:
        plt.text(xmin, annot_y.iloc[0], ('%' + end_fmt) % top.iloc[0], **annot_kwargs)

        if total:
            plt.text(xmin + len(annot_y), total_y, ('%' + end_fmt) % top.iloc[-1], **annot_kwargs)

    for idx, p in zip(it.count(xmin), bottom):
        if idx == xmin:
            continue

        plt.plot([idx - 1 - 0.4, idx + 0.4], [p, p], ls='--', color='0.5')

    plt.xticks(xmin + np.arange(len(height)), list(height.index))

    if total:
        plt.bar([xmin + len(bottom)], [top.iloc[-1]], **total_kwargs)
        plt.plot(
            [xmin + len(bottom) - 1 - 0.4, xmin + len(bottom) + 0.4],
            [top.iloc[-1], top.iloc[-1]],
            ls='--', color='0.5',
        )


def colormap(x, cmap='coolwarm', center=True, vmin=None, vmax=None, norm=None):
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.colors as colors

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


def bar(s, cmap='viridis', color=None, norm=None, orientation='vertical'):
    import matplotlib.colors
    import matplotlib.pyplot as plt

    if norm is None:
        norm = matplotlib.colors.NoNorm()

    if color is None:
        color = colormap(s, cmap=cmap, norm=norm)

    indices = range(len(s))

    if orientation == 'vertical':
        plt.bar(indices, s, color=color)
        plt.xticks(indices, s.index)

    else:
        plt.barh(indices, s, color=color)
        plt.yticks(indices, s.index)
