"""Helper for ML code.

Distributed as part of ``https://github.com/chmp/misc-exp`` under the MIT
license, (c) 2017 Christopher Prohm.
"""
import contextlib


class PickableTFModel:
    """Helper to build pickable tensorflow models.

    Usage::

        class Model(PickableTFModel):
            __params__ = ('n_layers', 'n_hidden')

            # NOTE: graph is always passed as kw arg on restore
            def __init__(self, *, n_layers, n_hidden, graph=None):
                super().__init__()
                self.n_layers = n_layers
                self.n_hidden = n_hidden

                self._build(graph=graph)

            def _build(self, graph):
                with self.build_context(graph):
                    # build the model graph
                    ...

        model = Model(n_layers=2, n_hidden=32)

        # NOTE: requires a valid session. It can also be passed as an argument
        # to to_pickable / restore.
        with open('model.pickle', 'wb') as fobj:
            pickle.dump(model.to_pickable(), fobj)

        with open('model.pickle', 'rb') as fobj:
            model = pickle.load(fobj).restore()

    """
    __params__ = ()

    def __init__(self):
        self.variables = set()

    def to_pickable(self, session=None):
        if session is None:
            import tensorflow as tf
            session = tf.get_default_session()

        init_kwargs = {k: getattr(self, k) for k in self.__params__}
        variables = {v.name: v.eval(session) for v in self.variables}
        return PickableWrapper(type(self), init_kwargs, variables)

    @contextlib.contextmanager
    def build_context(self, graph):
        """Register the default graph and capture any created variables.
        """
        with self.valid_graph(graph) as graph, self.capture_variables():
            yield graph

    @contextlib.contextmanager
    def capture_variables(self):
        variables = vars_to_save()
        yield
        self.variables = vars_to_save() - variables

    @staticmethod
    def valid_graph(graph=None):
        if graph is None:
            import tensorflow as tf
            graph = tf.get_default_graph()

        return graph.as_default()


def vars_to_save():
    import tensorflow as tf

    return (
        set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) |
        set(tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS))
    )


@contextlib.contextmanager
def capture_variables():
    result = set()
    variables = vars_to_save()
    yield result
    result.update(vars_to_save() - variables)


class PickableWrapper:
    def __init__(self, cls, init_kwargs, variables):
        self.cls = cls
        self.init_kwargs = init_kwargs
        self.variables = variables

    def restore(self, session=None):
        import tensorflow as tf

        if session is None:
            session = tf.get_default_session()

        model = self.cls(graph=session.graph, **self.init_kwargs)

        session.run(tf.global_variables_initializer())

        for v in model.variables:
            session.run(v.assign(self.variables[v.name]))

        return model
