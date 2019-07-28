import json
import functools as ft
import logging

import nbformat

_logger = logging.getLogger(__name__)


def main(input, output=None):
    if output is None:
        _logger.info("transform %s", input)

    else:
        _logger.info("transform %s -> %s", input, output)

    nb = nbformat.read(input, as_version=4)
    source = "\n".join(export_notebook(nb))

    if output is None:
        print(source)

    else:
        with open(output, "wt") as fobj:
            fobj.write(source)


# TODO: move general dispatch functionality into chmp.ds?
def dispatch(func):
    """A general dispatch function.
    """
    registry = {}

    @ft.wraps(func)
    def wrapper(*args, **kwargs):
        key = func(*args, **kwargs)
        handler = resolver(registry, key)
        return handler(*args, **kwargs)

    def resolver(registery, key):
        try:
            return registry[key]

        except KeyError:
            raise DispatchError("Cannot dispatch for key {!r}".format(key))

    def resolve(func):
        nonlocal resolver
        resolver = func
        return func

    def register(key):
        def decorator(func):
            registry[key] = func
            return func

        return decorator

    wrapper.key = func
    wrapper.resolve = resolve
    wrapper.register = register

    return wrapper


class DispatchError(Exception):
    pass


@dispatch
def export_notebook(obj, *args, **kwargs):
    if "nbformat" in obj:
        return "notebook"

    elif "cell_type" in obj:
        return "cell:{}".format(obj["cell_type"])


@export_notebook.register("notebook")
def export_notebook_notebook(obj):
    for cell in obj["cells"]:
        yield from export_notebook(cell)


@export_notebook.register("cell:raw")
def export_notebook_cell_raw(obj):
    tags = {*obj["metadata"].get("tags", ())}
    if "chmp-export-ignore" in tags:
        return

    if "chmp-export-meta" in tags:
        meta = json.loads(obj["source"])

        yield f"Title: {meta['title']}"
        yield f"Date: {meta['date']}"
        yield f"Author: {meta['author']}"
        yield ""
        yield ""

    else:
        raise ValueError("Cannot handle raw cell that is not tagged as meta or ignore")


@export_notebook.register("cell:markdown")
def export_notebook_cell_markdown(obj):
    source = obj["source"].splitlines()
    yield from source
    yield ""


@export_notebook.register("cell:code")
def export_notebook_cell_code(obj):
    tags = {*obj["metadata"].get("tags", ())}
    tags = {tag for tag in tags if tag.startswith("chmp-export-")}

    if "chmp-export-ignore" in tags:
        return

    hide_output = "chmp-export-hide-output" in tags
    tags = tags - {"chmp-export-hide-output"}

    if tags:
        raise ValueError("Cannot handle tags {}".format(tags))

    yield "```python"
    yield from obj["source"].splitlines()
    yield "```"
    yield ""

    if not hide_output and obj["outputs"]:
        raise ValueError("Cannot handle output")
