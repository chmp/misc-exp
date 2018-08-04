
## `chmp.ds`

Helpers for data science.

Distributed as part of `https://github.com/chmp/misc-exp` under the MIT
license, (c) 2017 Christopher Prohm.



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



### `chmp.ds.axtext`
`chmp.ds.axtext(*args, **kwargs)`

Add a text in axes coordinates (similar `figtext`).

Usage:

```
axtext(0, 0, 'text')
```



### `chmp.ds.change_plot`
`chmp.ds.change_plot(**kwargs)`

Plot changes in a quantity with pyplot's standard plot function.



### `chmp.ds.change_vspan`
`chmp.ds.change_vspan(**kwargs)`

Plot changes in a quantity with vspans.



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



### `chmp.ds.edges`
`chmp.ds.edges()`

Create edges for use with pcolor.

Usage:

```
assert x.size == v.shape[1]
assert y.size == v.shape[0]
pcolor(edges(x), edges(y), v)
```



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



### `chmp.ds.path`
`chmp.ds.path(x, close=True, **kwargs)`

Plot a path given as a list of vertices.

Usage:

```
path([0, 1, 0], [0, 1, 1], facecolor='r')
```



### `chmp.ds.reload`
`chmp.ds.reload(*modules_or_module_names)`

<undocumented>


