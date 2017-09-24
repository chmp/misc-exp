"""Helper for data science.

Distributed as part of ``https://github.com/chmp/misc-exp`` under the MIT
license, (c) 2017 Christopher Prohm.
"""
import importlib
import itertools as it
import warnings


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


def get_color_cycle():
    import matplotlib as mpl

    return mpl.rcParams['axes.prop_cycle'].by_key()['color']


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
        iteration order of this parameter. If not given but groups is passed,
        the keys of `groups` will be used, otherwise the existing categories
        of ``s`` will be used.

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
