"""Support for reactive notebooks.
"""
import ast
import hashlib
import inspect
import json


class FunctionalGraph:
    def __init__(self):
        self.definitions = {}
        self.graph = {}

    def update_definition(self, key, func):
        self.definitions[key] = func
        self.graph[key] = get_requirements(func)

    def update_states(self, roots):
        raise NotImplementedError()


def identity(obj):
    return obj


def no_display(obj):
    pass


class IPythonGraph(FunctionalGraph):
    def __init__(self):
        super().__init__()
        self.old_display_ids = {}
        self.module = get_main_module()
        self.display_funcs = {}

    def define(self, display=identity):
        def decorator(func):
            self.display_funcs[func.__name__] = display
            self.update_definition(func.__name__, func)
            self.update_states([func.__name__])
            return getattr(self.module, func.__name__)

        return decorator

    def update_states(self, roots):
        for var in get_outstanding_updates(roots, invert_graph(self.graph)):
            self.update_variable(var, primary=var in roots)

    def update_variable(self, name, primary):
        value = eval_func_in_module(self.module, self.definitions[name])
        display_func = self.display_funcs[name]

        self.update_named_display(name, [display_func(value)] + collect_figures(), primary=primary)

        setattr(self.module, name, value)

    def update_named_display(self, name, displays, primary):
        display_ids = update_named_display(
            name, displays,
            primary=primary,
            old_display_ids=self.old_display_ids.get(name, []),
        )
        self.old_display_ids[name] = display_ids


def eval_func_in_module(module, func):
    requirements = get_requirements(func)
    arguments = {}

    for name in requirements:
        try:
            arguments[name] = getattr(module, name)

        except AttributeError:
            return '<requirements not found>'

    return func(**arguments)


def get_main_module():
    import __main__
    return __main__


def update_named_display(name, displays, primary, old_display_ids):
    from IPython.display import display, update_display

    display_func = display if primary else update_display

    if not primary:
        for display_id in old_display_ids:
            display_func('<removed>', display_id=display_id)

    display_ids = []

    displays = [display for display in displays if display is not None]

    for idx, val in enumerate(displays):
        display_id = f'rx-{name}-{idx}'
        display_ids.append(display_id)
        display_func(val, display_id=display_id)

    return display_ids


def collect_figures():
    from IPython.core.pylabtools import getfigs, print_figure
    from IPython.display import Image
    from matplotlib import pyplot as plt

    figures = []

    for fig in getfigs():
        data = print_figure(fig)
        figures.append(Image(data))
        plt.close(fig)

    return figures


def get_requirements(func):
    spec = inspect.getfullargspec(func)
    return spec.args + spec.kwonlyargs


def get_graph(definitions):
    return {
        key: get_requirements(func)
        for key, func in definitions.items()
    }


def get_outstanding_updates(roots, inverse_graph):
    """Get all graph elements that need to be updated.

    :param Sequence[str] roots:
        the elements that triggered the current re-execution of the graph.

    :param Mapping[str,Iterable[str]] inverse_graph:
        a mapping from element to elements that depend on it. Use
        :func:`invert_graph` to get the inverse graph from the primary graph.

    :rtype: Sequence[str]
    :returns:
        the sequence of updates to perform to update the whole graph.
    """
    roots = strip_transitive(inverse_graph, roots)

    # concatenate any updates, regardless of whether dependencies have been updated
    updates = list(roots)
    dirty = set(updates)
    seen = set()

    while dirty:
        # use min to guarantee deterministic output
        current = min(dirty)
        dependents = inverse_graph.get(current, [])

        seen = seen | {current}
        dirty = (dirty | set(dependents)) - seen
        updates += dependents

    # strip any updates for which dependencies have not been updated
    return keep_last(updates)


def strip_transitive(graph, keys):
    """Strip any key in keys that is a transitive child of any other key in keys.
    """
    transitive = get_transitive(graph)

    keys = set(keys)
    for k in keys:
        # k was already removed, no need to check
        if k not in keys:
            continue

        keys = keys - set(transitive.get(k, []))

    return keys


def get_transitive(graph):
    """Transform a graph ``{key: [child]}`` into ``{key: [transitive-cild]}``.
    """
    transitive = {}
    for k in graph:
        _update_transitive(transitive, graph, k)

    return transitive


def _update_transitive(transitive, graph, key):
    if key in transitive:
        return

    direct = graph.get(key, [])
    transitive[key] = set(direct)

    for k in direct:
        _update_transitive(transitive, graph, k)
        transitive[key].update(transitive[k])


def invert_graph(graph):
    """Transfrom a graph ``{el: [dependency]}`` into ``{dependency: [el]}``.

    :param Mapping[str,Iterable[str]] graph:
        a mapping from element to its dependencies.

    :rtype: Mapping[str,Iterable[str]]
    :returns:
        a mapping from element to elements that depend on it.
    """
    ig = collect(
        (dependency, el)
        for el, dependencies in graph.items()
        for dependency in dependencies
    )
    # sort to make the order deterministic
    return {k: sorted(set(v)) for k, v in ig.items()}


def keep_last(items):
    """Return a list, while only retaining the last instance of any item.
    """
    items = reversed(items)
    items = keep_first(items)
    items = list(items)
    items.reverse()

    return items


def keep_first(items):
    """Return an iterator over the first instances of any item.
    """
    seen = set()

    for item in items:
        if item in seen:
            continue

        seen.add(item)
        yield item


def collect(iterable):
    """For an iterable of ``(key, value)`` pairs return a dict ``{key: [value]}``.
    """
    res = {}
    for key, val in iterable:
        res.setdefault(key, []).append(val)

    return res


def ast_hash(tree):
    h = hashlib.sha1()
    _ast_hash(h, tree)
    return h.hexdigest()


def _ast_hash(h, obj):
    if isinstance(obj, ast.AST):
        _ast_hash(h, type(obj))
        _ast_hash(h, {key: getattr(obj, key) for key in obj._fields})

    elif isinstance(obj, type):
        h.update(b'type:')
        _ast_hash(h, obj.__module__)
        _ast_hash(h, obj.__name__)

    elif type(obj) is str:
        h.update(json.dumps(obj).encode('utf8'))

    elif isinstance(obj, (str, float, int, bool, type(None))):
        _ast_hash(h, type(obj))
        _ast_hash(h, json.dumps(obj))

    elif isinstance(obj, (list, tuple)):
        _ast_hash(h, type(obj))
        _ast_hash(h, len(obj))
        for item in obj:
            _ast_hash(h, item)

    elif isinstance(obj, dict):
        _ast_hash(h, type(obj))
        _ast_hash(h, len(obj))
        for key in sorted(obj):
            _ast_hash(h, key)
            _ast_hash(h, obj[key])

    else:
        raise ValueError(f'cannot hash object of type {type(obj)}')
