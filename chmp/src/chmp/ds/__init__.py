"""Helper for data science.
"""

import importlib
import itertools as it


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
        xlabel=None, ylabel=None,
        title=None, suptitle=None,
        xscale=None, yscale=None, caption=None,
        xlim=None, ylim=None,
        left=None, top=None, bottom=None, right=None, wspace=None, hspace=None,
        subplot=None,
        legend=None,
):
    """Set various style related options of MPL.
    """
    import matplotlib.pyplot as plt

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

    if caption is not None:
        _caption(caption)

    subplot_kwargs = _dict_of_optionals(
        left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace
    )

    if subplot_kwargs:
        plt.subplots_adjust(**subplot_kwargs)

    if legend is not None:
        plt.legend(**legend)

    if subplot is not None:
        plt.sca(ax)


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
