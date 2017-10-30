
## `chmp.ds`

Helper for data science.

Distributed as part of `https://github.com/chmp/misc-exp` under the MIT
license, (c) 2017 Christopher Prohm.



### `chmp.ds.change_plot`
`chmp.ds.change_plot(**kwargs)`

Plot changes in a quantity with pyplot's standard plot function.



### `chmp.ds.change_vspan`
`chmp.ds.change_vspan(**kwargs)`

Plot changes in a quantity with vspans.



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

* **min_min_frequency** (*float*):
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
  included in the categories, it is added.
* **inplace** (*bool*):
  if True the series will be modified in place.
* **groups** (*Optional[Mapping[Any,Iterable[Any]]]*):
  if given, specifies which categories to replace by which in the form
  of `{replacement: list_of_categories_to_replace}`.
* **ordered** (*bool*):
  if True the resulting series will have ordered categories.



### `chmp.ds.get_color_cycle`
`chmp.ds.get_color_cycle()`

<undocumented>



### `chmp.ds.mpl_set`
`chmp.ds.mpl_set(box, xlabel, ylabel, title, suptitle, xscale, yscale, caption, xlim, ylim, xticks, yticks, left, top, bottom, right, wspace, hspace, subplot, legend, colorbar, box=None, xlabel=None, ylabel=None, title=None, suptitle=None, xscale=None, yscale=None, caption=None, xlim=None, ylim=None, xticks=None, yticks=None, left=None, top=None, bottom=None, right=None, wspace=None, hspace=None, subplot=None, legend=None, colorbar=None)`

Set various style related options of MPL.



### `chmp.ds.notebook_preamble`
`chmp.ds.notebook_preamble()`

Add common code



### `chmp.ds.reload`
`chmp.ds.reload()`

<undocumented>


