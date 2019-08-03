## `chmp.ds`

Helpers for data science.

Distributed as part of `https://github.com/chmp/misc-exp` under the MIT
license, (c) 2017 Christopher Prohm.


### `chmp.ds.define`
`chmp.ds.define(func)`

Execute a function and return its result.

The idea is to use function scope to prevent pollution of global scope in
notebooks.

Usage:

```python
@define
def foo():
    return 42

assert foo == 42
```


### `chmp.ds.cached`
`chmp.ds.cached(path, validate=False)`

Similar to `define`, but cache to a file.

#### Parameters

* **path** (*any*):
  the path of the cache file to use
* **validate** (*any*):
  if True, always execute the function. The loaded result will be
  passed to the function, when the cache exists. In that case the
  function should return the value to use. If the returned value is
  not identical to the loaded value, the cache is updated with the
  new value.

Usage:

```python
@cached('./cache/result')
def dataset():
    ...
    return result
```

or:

```python
@cached('./cache/result', validate=True)
def model(result=None):
    if result is not None:
        # running to validate ...

    return result
```


### `chmp.ds.Object`
`chmp.ds.Object(*args, **kwargs)`

Dictionary-like namespace object.


### `chmp.ds.daterange`
`chmp.ds.daterange(start, end, step=None)`

A range of dates.


### `chmp.ds.undefined`
`chmp.ds.undefined(*args, **kwargs)`

Sentinel class

Initialize self.  See help(type(self)) for accurate signature.


### `chmp.ds.first`
`chmp.ds.first(iterable, default=<undefined>)`

Return the first item of an iterable


### `chmp.ds.last`
`chmp.ds.last(iterable, default=<undefined>)`

Return the last item of an iterable


### `chmp.ds.item`
`chmp.ds.item(iterable, default=<undefined>)`

Given a single item iterable return this item.


### `chmp.ds.kvpair`
`chmp.ds.kvpair(*args, **kwargs)`

kvpair(key, value)

Initialize self.  See help(type(self)) for accurate signature.


### `chmp.ds.cell`
`chmp.ds.cell(name=None)`

No-op context manager to allow indentation of code


### `chmp.ds.colorize`
`chmp.ds.colorize(items, cmap=None)`

Given an iterable, yield `(color, item)` pairs.

#### Parameters

* **cmap** (*any*):
  if None the color cycle is used, otherwise it is interpreted as a
  colormap to color the individual items.
  
  Note: `items` is fully instantiated during the iteration. For any
  `list` or `tuple` item only its first element is used for
  colomapping.
  
  This procedure allows for example to colormap a pandas Dataframe
  grouped on a number column:
  
  ```python
  for c, (_, g) in colorize(df.groupby("g"), cmap="viridis"):
      ...
  ```


### `chmp.ds.get_color_cycle`
`chmp.ds.get_color_cycle(n=None)`

Return the matplotlib color cycle.

#### Parameters

* **n** (*Optional[int]*):
  if given, return a list with exactly n elements formed by repeating
  the color cycle as necessary.

Usage:

```python
blue, green, red = get_color_cycle(3)
```


### `chmp.ds.mpl_set`
`chmp.ds.mpl_set(box=None, xlabel=None, ylabel=None, title=None, suptitle=None, xscale=None, yscale=None, caption=None, xlim=None, ylim=None, xticks=None, yticks=None, xformatter=None, yformatter=None, left=None, top=None, bottom=None, right=None, wspace=None, hspace=None, subplot=None, legend=None, colorbar=None, invert=None, ax=None, grid=None, axis=None)`

Set various style related options of MPL.

#### Parameters

* **xformatter** (*any*):
  if given a formatter for the major x ticks. Should have the
  signature `(x_value, pos) -> label`.
* **yformatter** (*any*):
  See `xformatter`.
* **invert** (*any*):
  if given invert the different axes. Can be x, y, or xy.


### `chmp.ds.diagonal`
`chmp.ds.diagonal(**kwargs)`

Draw a diagonal line in the current axis.


### `chmp.ds.qlineplot`
`chmp.ds.qlineplot(*, x, y, hue, data, ci=0.95)`

Plot  median as line, quantiles as shading.


### `chmp.ds.pgm`
`chmp.ds.pgm(*, ax=None, nodes=(), edges=(), annotations=(), **kwargs)`

Wrapper around [daft.PGM](http://daft-pgm.org/api/#daft.PGM) to allow fluid call chains.

Usage:

```python
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
```

To annotate a node use:

```python
.annotate(node_name, annotation_text)
```

Nodes can also be created without explicit lables (in which case the node
name is used):

```python
.node("z", 1, 1)
node("z", "label", 1, 1)
```


#### `chmp.ds.pgm.update`
`chmp.ds.pgm.update(nodes=None, edges=None, annotations=None)`

Replace a full set of features.


#### `chmp.ds.pgm.remove`
`chmp.ds.pgm.remove(incoming=(), outgoing=())`

Remove edges that point in or out of a the specified nodes.


#### `chmp.ds.pgm.render`
`chmp.ds.pgm.render(ax=None, axis=False, xlim=None, ylim=None, **kwargs)`

Render the figure.

#### Parameters

* **ax** (*any*):
  the axes to draw into. If not given, the axis specified in
  __init__ or the current axes is used.
* **xlim** (*any*):
  the xlim to use. If not given, it is determined from the data.
* **ylim** (*any*):
  the ylim to use. If not given, it is determined from the data.
* **kwargs** (*any*):
  keyword arguments forward to mpl set.

#### Returns

the pgm object.


### `chmp.ds.edges`
`chmp.ds.edges(x)`

Create edges for use with pcolor.

Usage:

```python
assert x.size == v.shape[1]
assert y.size == v.shape[0]
pcolor(edges(x), edges(y), v)
```


### `chmp.ds.center`
`chmp.ds.center(u)`

Compute the center between edges.


### `chmp.ds.caption`
`chmp.ds.caption(s, size=13, strip=True)`

Add captions to matplotlib graphs.


### `chmp.ds.change_vspan`
`chmp.ds.change_vspan(x, y, *, data=None, color=('w', '0.90'), transform_x=None, transform_y=None, skip_nan=True, **kwargs)`

Plot changes in a quantity with vspans.


### `chmp.ds.change_plot`
`chmp.ds.change_plot(x, y, *, data=None, transform_x=None, transform_y=None, skip_nan=True, **kwargs)`

Plot changes in a quantity with pyplot's standard plot function.


### `chmp.ds.axtext`
`chmp.ds.axtext(*args, **kwargs)`

Add a text in axes coordinates (similar `figtext`).

Usage:

```python
axtext(0, 0, 'text')
```


### `chmp.ds.plot_gaussian_contour`
`chmp.ds.plot_gaussian_contour(x, y, *, data=None, q=(0.99,), ax=None, **kwargs)`

Plot isocontours of the maximum likelihood Gaussian for `x, y`.

#### Parameters

* **q** (*any*):
  the quantiles to show.


### `chmp.ds.to_markdown`
`chmp.ds.to_markdown(df, index=False)`

Return a string containg the markdown of the table.

Depends on the `tabulate` dependency.


### `chmp.ds.index_query`
`chmp.ds.index_query(obj, expression, scalar=False)`

Execute a query expression on the index and return matching rows.

#### Parameters

* **scalar** (*any*):
  if True, return only the first item. Setting `scalar=True`
  raises an error if the resulting object has have more than one
  entry.


### `chmp.ds.fix_categories`
`chmp.ds.fix_categories(s, categories=None, other_category=None, inplace=False, groups=None, ordered=False)`

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
`chmp.ds.find_high_frequency_categories(s, min_frequency=0.02, n_max=None)`

Find categories with high frequency.

#### Parameters

* **min_frequency** (*float*):
  the minimum frequency to keep
* **n_max** (*Optional[int]*):
  if given keep at most `n_max` categories. If more are present after
  filtering for minimum frequency, keep the highest `n_max` frequency
  columns.


### `chmp.ds.singledispatch_on`
`chmp.ds.singledispatch_on(idx)`

Helper to dispatch on any argument, not only the first one.


### `chmp.ds.setdefaultattr`
`chmp.ds.setdefaultattr(obj, name, value)`

`dict.setdefault` for attributes


### `chmp.ds.szip`
`chmp.ds.szip(iterable_of_objects, sequences=(<class 'tuple'>,), mappings=(<class 'dict'>,), return_schema=False)`

Zip but for deeply nested objects.

For a list of nested set of objects return a nested set of list.


### `chmp.ds.smap`
`chmp.ds.smap(func, arg, *args, sequences=(<class 'tuple'>,), mappings=(<class 'dict'>,))`

A structured version of map.

The structure is taken from the first arguments.


### `chmp.ds.copy_structure`
`chmp.ds.copy_structure(template, obj, sequences=(<class 'tuple'>,), mappings=(<class 'dict'>,))`

Arrange `obj` into the structure of `template`.

#### Parameters

* **template** (*any*):
  the object of which top copy the structure
* **obj** (*any*):
  the object which to arrange into the structure. If it is
  already structured, the template structure and its structure
  must be the same or a value error is raised


### `chmp.ds.json_numpy_default`
`chmp.ds.json_numpy_default(obj)`

A default implementation for `json.dump` that deals with numpy datatypes.


### `chmp.ds.bgloop`
`chmp.ds.bgloop(tag, *iterables, runner=None)`

Run a loop in a background thread.


### `chmp.ds.Display`
`chmp.ds.Display(obj=None)`

An interactive display for use in background tasks.


### `chmp.ds.timed`
`chmp.ds.timed(tag=None, level=20)`

Time a codeblock and log the result.

Usage:

```python
with timed():
    long_running_operation()
```

#### Parameters

* **tag** (*any*):
  an object used to identify the timed code block. It is printed with
  the time taken.


### `chmp.ds.find_categorical_columns`
`chmp.ds.find_categorical_columns(df)`

Find all categorical columns in the given dataframe.


### `chmp.ds.filter_low_frequency_categories`
`chmp.ds.filter_low_frequency_categories(columns=None, min_frequency=0.02, other_category=None, n_max=None)`

Build a transformer to filter low frequency categories.

Usage:

```python
pipeline = build_pipeline[
    categories=filter_low_frequency_categories(),
    predict=lgb.LGBMClassifier(),
)
```


### `chmp.ds.column_transform`
`chmp.ds.column_transform(*args, **kwargs)`

Build a transformer for a list of columns.

Usage:

```python
pipeline = build_pipeline(
    transform=column_transform(['a', 'b'], np.abs),
    classifier=sk_ensemble.GradientBoostingClassifier(),
])
```

Or:

```python
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

```python
pipeline = build_pipeline(
    transform=...,
    predict=...,
)
```


### `chmp.ds.transform`
`chmp.ds.transform(*args, **kwargs)`

Build a function transformer with args / kwargs bound.

Usage:

```python
pipeline = build_pipeline(
    transform=transform(np.abs)),
    classifier=sk_ensemble.GradientBoostingClassifier()),
)
```


### `chmp.ds.FuncTransformer`
`chmp.ds.FuncTransformer(func)`

Simple **non-validating** function transformer.

#### Parameters

* **func** (*callable*):
  the function to apply on transform


### `chmp.ds.DataFrameEstimator`
`chmp.ds.DataFrameEstimator(est)`

Add support for dataframe use to sklearn estimators.


### `chmp.ds.FitInfo`
`chmp.ds.FitInfo(extractor, target=None)`

Extract and store meta data of the dataframe passed to fit.


### `chmp.ds.waterfall`
`chmp.ds.waterfall(obj, col=None, base=None, total=False, end_annot=None, end_fmt='.g', annot=False, fmt='+.2g', cmap='coolwarm', xmin=0, total_kwargs=None, annot_kwargs=None, **kwargs)`

Plot a waterfall chart.

Usage:

```python
series.pipe(waterfall, annot='top', fmt='+.1f', total=True)
```


### `chmp.ds.render_poyo`
`chmp.ds.render_poyo(obj, params)`

Lighweight POYO templating.

Any callable in the tree will be called with params. Example:

```python
template = {
    "key": lambda params: params['value'],
}

render_poyo(template, {'value': 20})
```


### `chmp.ds.dashcb`
`chmp.ds.dashcb(app, output, *inputs, figure=False)`

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

```python
@app.callback(dash.dependencies.Output('display', 'children'),
              [dash.dependencies.Input('dropdown', 'value')])
def update_display(value):
    return 'Selected: "{}"'.format(value)
```

With dashcb, it can be written as:

```python
@dashcb(app, 'display:children', 'dropdown:value')
def update_display(value):
    return 'Selected: "{}"'.format(value)
```

To use dash with matplotlib figure, define an `html.Img` element. For
example with id `my_image`. Then the plot can be updated via:

```python
@dashcb(app, 'my_image:src', 'dropdown:value', figure=True)
def update_display(value):
    plt.plot([1, 2, 3])
    plt.title(value)
```


### `chmp.ds.dashmpl`
`chmp.ds.dashmpl(func)`

Capture the current matplotlib figure.


### `chmp.ds.read_markdown_list`
`chmp.ds.read_markdown_list(fobj_or_path, *, section, columns, dtype=None, parse_dates=None, compression=None)`

Read a markdown file as a DataFrame.


### `chmp.ds.loop_over`
`chmp.ds.loop_over(iterable, label=None, keep=False)`

Simplified interface to Loop.over.

#### Parameters

* **label** (*any*):
  if a callable, it should return a str that is used as the loop label.


### `chmp.ds.LoopState`
`chmp.ds.LoopState(*args, **kwargs)`

An enumeration.

Initialize self.  See help(type(self)) for accurate signature.


#### `chmp.ds.LoopState.pending`

An enumeration.


#### `chmp.ds.LoopState.running`

An enumeration.


#### `chmp.ds.LoopState.done`

An enumeration.


#### `chmp.ds.LoopState.aborted`

An enumeration.


### `chmp.ds.Loop`
`chmp.ds.Loop(time=<built-in function time>, stack=None, root=None, debounce=0.1)`

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


#### `chmp.ds.Loop.will_print`
`chmp.ds.Loop.will_print(now=None)`

Check whether the print invocation will be debounced.


### `chmp.ds.tdformat`
`chmp.ds.tdformat(time_delta)`

Format a timedelta given in seconds or as a `datetime.timedelta`.


### `chmp.ds.ascii_bar`
`chmp.ds.ascii_bar(u, n=10)`

Format a ASCII progressbar


### `chmp.ds.sha1`
`chmp.ds.sha1(obj)`

Create a hash for a json-encode-able object


### `chmp.ds.random`
`chmp.ds.random(obj)`

Return a random float in the range [0, 1)


### `chmp.ds.np_seed`
`chmp.ds.np_seed(obj)`

Return a seed usable by numpy.


### `chmp.ds.tf_seed`
`chmp.ds.tf_seed(obj)`

Return a seed usable by tensorflow.


### `chmp.ds.std_seed`
`chmp.ds.std_seed(obj)`

Return a seed usable by python random module.


### `chmp.ds.shuffle`
`chmp.ds.shuffle(obj, l)`

Shuffle `l` in place using Fisher–Yates algorithm.

See: https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle


### `chmp.ds.timeshift_index`
`chmp.ds.timeshift_index(obj, dt)`

Return a shallow copy of `obj` with its datetime index shifted by `dt`.


### `chmp.ds.to_start_of_day`
`chmp.ds.to_start_of_day(s)`

Return the start of the day for the datetime given in `s`.


### `chmp.ds.to_time_in_day`
`chmp.ds.to_time_in_day(s, unit=None)`

Return the timediff relative to the start of the day of `s`.


### `chmp.ds.to_start_of_week`
`chmp.ds.to_start_of_week(s)`

Return the start of the week for the datetime given `s`.


### `chmp.ds.to_time_in_week`
`chmp.ds.to_time_in_week(s, unit=None)`

Return the timedelta relative to weekstart for the datetime given in `s`.


### `chmp.ds.to_start_of_year`
`chmp.ds.to_start_of_year(s)`

Return the start of the year for the datetime given in `s`.


### `chmp.ds.to_time_in_year`
`chmp.ds.to_time_in_year(s, unit=None)`

Return the timediff relative to the start of the year for `s`.

