## `chmp.ds`

Helpers for data science.

Distributed as part of `https://github.com/chmp/misc-exp` under the MIT
license, (c) 2017 Christopher Prohm.


### `chmp.ds.define`
`chmp.ds.define()`

Execute a function and return its result.

The idea is to use function scope to prevent pollution of global scope in
notebooks.

Usage:

```
@define
def foo():
    return 42

assert foo == 42
```


### `chmp.ds.Object`
`chmp.ds.Object(*args, **kwargs)`

Dictionary-like namespace object.


### `chmp.ds.get_color_cycle`
`chmp.ds.get_color_cycle(n, n=None)`

Return the matplotlib color cycle.

#### Parameters

* **n** (*Optional[int]*):
  if given, return a list with exactly n elements formed by repeating
  the color cycle as necessary.

Usage:

```
blue, green, red = get_color_cycle(3)
```


### `chmp.ds.mpl_set`
`chmp.ds.mpl_set(box, xlabel, ylabel, title, suptitle, xscale, yscale, caption, xlim, ylim, xticks, yticks, xformatter, yformatter, left, top, bottom, right, wspace, hspace, subplot, legend, colorbar, invert, ax, grid, box=None, xlabel=None, ylabel=None, title=None, suptitle=None, xscale=None, yscale=None, caption=None, xlim=None, ylim=None, xticks=None, yticks=None, xformatter=None, yformatter=None, left=None, top=None, bottom=None, right=None, wspace=None, hspace=None, subplot=None, legend=None, colorbar=None, invert=None, ax=None, grid=None)`

Set various style related options of MPL.

#### Parameters

* **xformatter** (*Optional[callable]*):
  if given a formatter for the major x ticks. Should have the
  signature `(x_value, pos) -> label`.
* **yformatter** (*Optional[callable]*):
  See `xformatter`.


### `chmp.ds.edges`
`chmp.ds.edges()`

Create edges for use with pcolor.

Usage:

```
assert x.size == v.shape[1]
assert y.size == v.shape[0]
pcolor(edges(x), edges(y), v)
```


### `chmp.ds.caption`
`chmp.ds.caption(s, size, size=13, strip=True)`

Add captions to matplotlib graphs.


### `chmp.ds._caption`
`chmp.ds._caption(s, size, size=13, strip=True)`

Add captions to matplotlib graphs.


### `chmp.ds.change_vspan`
`chmp.ds.change_vspan(**kwargs)`

Plot changes in a quantity with vspans.


### `chmp.ds.change_plot`
`chmp.ds.change_plot(**kwargs)`

Plot changes in a quantity with pyplot's standard plot function.


### `chmp.ds.path`
`chmp.ds.path(x, close=True, **kwargs)`

Plot a path given as a list of vertices.

Usage:

```
path([0, 1, 0], [0, 1, 1], facecolor='r')
```


### `chmp.ds.axtext`
`chmp.ds.axtext(*args, **kwargs)`

Add a text in axes coordinates (similar `figtext`).

Usage:

```
axtext(0, 0, 'text')
```


### `chmp.ds.fix_categories`
`chmp.ds.fix_categories(s, categories, other_category, inplace, groups, categories=None, other_category=None, inplace=False, groups=None, ordered=False)`

Fix the categories of a categorical series.

#### Parameters

* **s** (*pd.Series*):
  the series to normalize
* **categories** (*Optional[Iterable[Any]]*):
  the categories to keep. The result will have categories in the
  iteration order of this parameter. If not given but `groups` is
  passed, the keys of `groups` will be used, otherwise the existing
  categories of `s` will be used.
* **other_category** (*Optional[Any]*):
  all categories to be removed wil be mapped to this value, unless they
  are specified specified by the `groups` parameter. If given and not
  included in the categories, it is appended to the given categories.
  For a custom order, ensure it is included in `categories`.
* **inplace** (*bool*):
  if True the series will be modified in place.
* **groups** (*Optional[Mapping[Any,Iterable[Any]]]*):
  if given, specifies which categories to replace by which in the form
  of `{replacement: list_of_categories_to_replace}`.
* **ordered** (*bool*):
  if True the resulting series will have ordered categories.


### `chmp.ds.find_high_frequency_categories`
`chmp.ds.find_high_frequency_categories(s, min_frequency, min_frequency=0.02, n_max=None)`

Find categories with high frequency.

#### Parameters

* **min_frequency** (*float*):
  the minimum frequency to keep
* **n_max** (*Optional[int]*):
  if given keep at most `n_max` categories. If more are present after
  filtering for minimum frequency, keep the highest `n_max` frequency
  columns.


### `chmp.ds.cast_types`
`chmp.ds.cast_types(numeric, categorical, numeric=None, categorical=None)`

Build a transform to cast numerical / categorical columns.

All non-cast columns are stripped for the dataframe.


### `chmp.ds.find_categorical_columns`
`chmp.ds.find_categorical_columns()`

Find all categorical columns in the given dataframe.


### `chmp.ds.filter_low_frequency_categories`
`chmp.ds.filter_low_frequency_categories(columns, min_frequency, other_category, n_max, columns=None, min_frequency=0.02, other_category=None, n_max=None)`

Build a transformer to filter low frequency categories.

Usage:

```
pipeline = build_pipeline[
    categories=filter_low_frequency_categories(),
    predict=lgb.LGBMClassifier(),
)
```


### `chmp.ds.column_transform`
`chmp.ds.column_transform(*args, **kwargs)`

Build a transformer for a list of columns.

Usage:

```
pipeline = build_pipeline(
    transform=column_transform(['a', 'b'], np.abs),
    classifier=sk_ensemble.GradientBoostingClassifier(),
])
```

Or:

```
pipeline = build_pipeline(
    transform=column_transform(
        a=np.abs,
        b=op.pos,
    ),
    classifier=sk_ensemble.GradientBoostingClassifier(),
)
```


### `chmp.ds.build_pipeline`
`chmp.ds.build_pipeline(**kwargs)`

Build a pipeline from named steps.

The order of the keyword arguments is retained. Note, this functionality
requires python `>= 3.6`.

Usage:

```
pipeline = build_pipeline(
    transform=...,
    predict=...,
)
```


### `chmp.ds.transform`
`chmp.ds.transform(*args, **kwargs)`

Build a function transformer with args / kwargs bound.

Usage:

```
pipeline = build_pipeline(
    transform=transform(np.abs)),
    classifier=sk_ensemble.GradientBoostingClassifier()),
)
```


### `chmp.ds.FuncTransformer`
`chmp.ds.FuncTransformer()`

Simple **non-validating** function transformer.

#### Parameters

* **func** (*callable*):
  the function to apply on transform


### `chmp.ds.DataFrameEstimator`
`chmp.ds.DataFrameEstimator()`

Add support for dataframe use to sklearn estimators.


### `chmp.ds.FitInfo`
`chmp.ds.FitInfo(extractor, target=None)`

Extract and store meta data of the dataframe passed to fit.


### `chmp.ds.waterfall`
`chmp.ds.waterfall(obj, col, base, total, end_annot, end_fmt, annot, fmt, cmap, xmin, total_kwargs, col=None, base=None, total=False, end_annot=None, end_fmt='.g', annot=False, fmt='+.2g', cmap='coolwarm', xmin=0, total_kwargs=None, annot_kwargs=None, **kwargs)`

Plot a waterfall chart.

Usage:

```
series.pipe(waterfall, annot='top', fmt='+.1f', total=True)
```


### `chmp.ds.dashcb`
`chmp.ds.dashcb(*inputs)`

Construct a dash callback using function annotations.

#### Parameters

* **app** (*dash.Dash*):
  the dash app to build the callback for
* **output** (*str*):
  the output, as a string of the form `{component}:{property}`
* **inputs** (*str*):
  the inputs, as strings of the form `{component}:{property}`
* **figure** (*bool*):
  if True, the current matplotlib figure will be captured and returned as
  a data URL. This allows to use matplotlib with dash. See the examples
  below

Consider the following dash callback:

```
@app.callback(dash.dependencies.Output('display', 'children'),
              [dash.dependencies.Input('dropdown', 'value')])
def update_display(value):
    return 'Selected: "{}"'.format(value)
```

With dashcb, it can be written as:

```
@dashcb(app, 'display:children', 'dropdown:value')
def update_display(value):
    return 'Selected: "{}"'.format(value)
```

To use dash with matplotlib figure, define an `html.Img` element. For
example with id `my_image`. Then the plot can be updated via:

```
@dashcb(app, 'my_image:src', 'dropdown:value', figure=True)
def update_display(value):
    plt.plot([1, 2, 3])
    plt.title(value)
```


### `chmp.ds.dashmpl`
`chmp.ds.dashmpl()`

Capture the current matplotlib figure.


### `chmp.ds.LoopState`
`chmp.ds.LoopState(*args, **kwargs)`

An enumeration.


### `chmp.ds.Loop`
`chmp.ds.Loop(time, stack, root, debounce, time=<built-in function time>, stack=None, root=None, debounce=0.1)`

A flexible progressbar indicator.

It's designed to make printing custom messages and customizing the loop
style easy.

The following format codes are recognized:

- `[`: in the beginning, indicates that the progress bar will be
  surrounded by brackets.
- `-`: in the beginning, indicates that the parts of the progress bar
  will be printed without automatic spaces
- `B`: a one character bar
- `b`: the full bar
- `t`: the total runtime so far
- `e`: the expected total runtime
- `r`: the expected remaining runtime
- `f`: the fraction of work performed so far
- additional characters will be included verbatim

To access nested loop use the getitem notation, e.g. `loop[1]`.


### `chmp.ds.tdformat`
`chmp.ds.tdformat()`

Format a timedelta given in seconds.


### `chmp.ds.ascii_bar`
`chmp.ds.ascii_bar(u, n=10)`

Format a ASCII progressbar


### `chmp.ds.sha1`
`chmp.ds.sha1()`

Create a hash for a json-encode-able object


### `chmp.ds.random`
`chmp.ds.random()`

Return a random float in the range [0, 1)


### `chmp.ds.np_seed`
`chmp.ds.np_seed()`

Return a seed usable by numpy.


### `chmp.ds.tf_seed`
`chmp.ds.tf_seed()`

Return a seed usable by tensorflow.


### `chmp.ds.std_seed`
`chmp.ds.std_seed()`

Return a seed usable by python random module.


### `chmp.ds.shuffle`
`chmp.ds.shuffle()`

Shuffle `l` in place using Fisher–Yates algorithm.

See: https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle

