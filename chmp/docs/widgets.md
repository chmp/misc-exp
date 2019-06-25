# chmp.widgets

To symlink the extension in development mode use:

```bash
jupyter nbextension install --sys-prefix --python --symlink chmp.widgets
```

## `chmp.widgets`


### `chmp.widgets.FocusCell`
`chmp.widgets.FocusCell(**kwargs)`

A widget to hide all other cells, but the one containing this widget.

Usage:

```
# in a notebook cell
widget = FocusCell()
widget
```

Public constructor


### `chmp.widgets.CommandInput`
`chmp.widgets.CommandInput(on_command=None)`

A entry line of a command-line like application.

When the user preses enter, the current value is sent passed to the
`on_command` callback and the input is cleared.

Usage:

```
widget = CommandInput(on_command=print)

@widget.on_command
def handler(value):
    print(value)

display(widget)
```


### `chmp.widgets.WidgetRegistry`
`chmp.widgets.WidgetRegistry()`

Register an retrieve widgets by name.

Usage:

```
registry = WidgetRegistry()

widget = HBox([
    registry("label", Label()),
])

registry.label.value = "Hello World"
```


### `chmp.widgets.PersistentDatasets`
`chmp.widgets.PersistentDatasets(datasets=None)`

Helper to keep data between the front and backend consistent with VegaWidget.

Usage:

```
# construct the datasets object
datasets = PersistentDatasets()

# notify the widget of any changes in the timeseries dataset
datasets.bind(widget, ["timeseries"])

# update the dataset
datasets.update(
    "timeseries",
    # insert new data points
    insert=[
        {"date": "2019-05-03", "value": 2.0},
        {"date": "2019-05-04", "value": 4.0},
    ],
    # specify a JS expression, it will be rewritten into python and
    # also run in the kernel.
    remove="datum.date <  '2019-04-01'",
)

# get the current python state of the dataset
datasets.get("timeseries")

# clear a single (or all datasets)
datasets.clear("timeseries")
datasets.clear_all()
```


### `chmp.widgets.JSExpr`
`chmp.widgets.JSExpr(argnames, expr)`

Wrapper around javascript exression as a python callable.


### `chmp.widgets.JSObj`
`chmp.widgets.JSObj(*args, **kwargs)`

A wrapper around a dictionary mimicking javascripts getattr behavior.

