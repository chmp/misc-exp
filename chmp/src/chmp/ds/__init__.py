"""Helpers for data science.

Distributed as part of ``https://github.com/chmp/misc-exp`` under the MIT
license, (c) 2017 Christopher Prohm.
"""
import base64
import collections
import functools as ft
import importlib
import io
import itertools as it
import sys

try:
    from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin

except ImportError:
    _HAS_SK_LEARN = False
    BaseEstimator = TransformerMixin = object

else:
    _HAS_SK_LEARN = True


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
        xformatter=None, yformatter=None,
        left=None, top=None, bottom=None, right=None, wspace=None, hspace=None,
        subplot=None,
        legend=None,
        colorbar=None,
        invert=None,
        ax=None,
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
            plt.legend(loc='best')

        else:
            plt.legend(**legend)

    if subplot is not None:
        plt.sca(ax)

    if colorbar is True:
        plt.colorbar()

    if invert is not None:
        if 'x' in invert:
            plt.gca().invert_xaxis()

        if 'y' in invert:
            plt.gca().invert_yaxis()


def edges(x):
    """Create edges for use with pcolor.

    Usage::

        assert x.size == v.shape[1]
        assert y.size == v.shape[0]
        pcolor(edges(x), edges(y), v)

    """
    import numpy as np

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


@ft.singledispatch
def get_children(est):
    return []


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
            raise ValueError('cannot change the type inplace')

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

    dangling_categories = {*removals} - {*remapped}
    if dangling_categories:
        if other_category is None:
            raise ValueError(
                'dangling categories %s found, need other category to assign' % dangling_categories,
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
            raise RuntimeError(f'could not cast {col} to numeric') from e

    for col in categorical:
        try:
            df = df.assign(**{col: df[col].astype('category')})

        except Exception as e:
            raise RuntimeError(f'could not cast {col} to categorical') from e

    return df[sorted({*categorical, *numeric})]


def find_categorical_columns(df):
    """Find all categorical columns in the given dataframe.
    """
    import pandas.api.types as pd_types

    return [
        k
        for k, dtype in df.dtypes.items()
        if pd_types.is_categorical_dtype(dtype)
    ]


def filter_low_frequency_categories(columns=None, min_frequency=0.02, other_category=None, n_max=None):
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
    def __init__(self, columns=None, min_frequency=0.02, other_category='other', n_max=None):
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
                    min_frequency=self._get('min_frequency', col),
                    n_max=self._get('n_max', col),
                )

            except Exception as e:
                raise RuntimeError(f'cannot determine high frequency categories for {col}')

            self._to_keep[col] = to_keep

        return self

    def transform(self, df, y=None):
        for col in self._columns:
            df = df.assign(**{
                col: fix_categories(
                    df[col], self._to_keep[col],
                    other_category=self._get('other_category', col),
                )
            })

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
    if not hasattr(x, 'assign'):
        raise RuntimeError('can only transform objects with an assign method.')

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
        raise RuntimeError('pipeline factory requires deterministic kwarg order')

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


class FuncRegressor(BaseEstimator, ClassifierMixin):
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
                raise RuntimeError(f'cannot fit {col}') from e

        return self

    def transform(self, x, y=None):
        for col in self.columns_:
            try:
                assignments = {}
                for level in self.levels_[col]:
                    assignments[f'{col}_{level}'] = (x[col] == level).astype(float)

                x = x.drop([col], axis=1).assign(**assignments)

            except Exception as e:
                raise RuntimeError(f'cannot transform {col}') from e

        return x


def multi_type_sorted(vals):
    import pandas as pd
    return sorted(vals, key=lambda v: (type(v).__module__, type(v).__name__, pd.isnull(v), v))


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


# TODO: make sureit can be called with a dataframe
def qplot(
    x=None, y=None, data=None, alpha=1.0, fill_alpha=0.8, color=None, ax=None,
    **line_kwargs
):
    import numpy as np
    import matplotlib.pyplot as plt

    if y is None and x is not None:
        x, y = y, x

    if y is None:
        raise ValueError('need data to plot')

    if isinstance(y, tuple) or (isinstance(y, np.ndarray) and y.ndim == 2):
        y = tuple(y)

    else:
        y = y,

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

    Consider the following das callback::


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
            _dash_cb_parse_annotation(dash.dependencies.Input, arg)
            for arg in inputs
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
        plt.savefig(img, format='png')
        plt.close()

        img = base64.b64encode(img.getvalue())
        img = img.decode('ascii')
        return 'data:image/png;base64,' + img

    return impl


def _dash_cb_parse_annotation(cls, s):
    element, _, property = s.partition(':')
    return cls(element, property)
