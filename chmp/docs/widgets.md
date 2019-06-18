# chmp.widgets

To symlink the extension in development mode use:

```python
jupyter nbextension install --sys-prefix --python --symlink chmp.widgets
```

## `chmp.widgets`


### `chmp.widgets.FocusCell`
`chmp.widgets.FocusCell(**kwargs)`

A widget to hide all other cells, but the one containing this widget

Public constructor


### `chmp.widgets.PersistentDatasets`
`chmp.widgets.PersistentDatasets(datasets=None, widget=None)`

Helper to keep data between the front and backend consistent with VegaWidget.


### `chmp.widgets.JSExpr`
`chmp.widgets.JSExpr(argnames, expr)`

Wrapper around javascript exression as a python callable.


### `chmp.widgets.JSObj`
`chmp.widgets.JSObj(*args, **kwargs)`

A wrapper around a dictionary mimicking javascripts getattr behavior.

