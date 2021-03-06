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

```python
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

```python
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

```python
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

```python
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


### `chmp.widgets.run_thread`
`chmp.widgets.run_thread(func=None, *, key=None, registry=None, interval=None, wake_interval=0.1)`

A decorator to run function a background thread.

This function is designed to be run in a notebook and will modify the
`__main__` module per default, i.e., global namespace.

The function is passed a context object, whose `running` attribute will
be set to false, when the function should stop executing:

```python
@run_thread
def func(ctx):
    while ctx.running:
        ...
```

To execute a function in regular intervals, set the `interval` argument of
the decorator. For example to excecute every 5 seconds, use:

```python
@run_thread(interval=5)
def func(ctx):
    ...
```

Any function started with `run_thread` can be stopped via
`stop_thread`.


### `chmp.widgets.stop_thread`
`chmp.widgets.stop_thread(func_or_key, *, registry=None)`

Stop a thread started with `run_thread`.

The argument can either be the function started or the key used when
starting it:

```python
stop_thread(func)
stop_thread("key")
```

The main thread will block until the function has stopped executing.


### `chmp.widgets.JSExpr`
`chmp.widgets.JSExpr(argnames, expr)`

Wrapper around javascript exression as a python callable.


### `chmp.widgets.JSObj`
`chmp.widgets.JSObj(*args, **kwargs)`

A wrapper around a dictionary mimicking javascripts getattr behavior.

