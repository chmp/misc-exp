"""Helpers to handle batched models.
"""

import collections
import enum
import numpy as np

from chmp.ds import loop_over, loop_nest

from ._metrics import ensure_metric

default_batch_size = 32


class BaseTrainer:
    def __init__(self, model, callbacks, metrics):
        if metrics is None:
            metrics = []

        elif not isinstance(metrics, (list, tuple)):
            metrics = [metrics]

        self.model = model
        self.callbacks = CallbackList.make(callbacks)
        self.metrics = [ensure_metric(metric) for metric in metrics]
        self.history = History()
        self.message = FitMessage()

        self.callbacks.append(self.history)
        self.callbacks.append(self.message)
        self.callbacks.set_model(self.model)

    def on_train_begin(self, logs):
        self.callbacks.on_train_begin(logs)

    def on_train_end(self, logs):
        self.callbacks.on_train_end(logs)

    def on_epoch_begin(self, epoch, logs):
        self.callbacks.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs):
        self.callbacks.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs):
        self.callbacks.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs):
        self.callbacks.on_batch_end(batch, logs)

    def on_validation_begin(self, logs):
        self.callbacks.on_validation_begin(logs)

    def on_validation_end(self, logs):
        self.callbacks.on_validation_end(logs)

    def batch_step(self, batch_x, batch_y, logs):
        raise NotImplementedError("Overwrite batch_step in derived class.")

    def batch_validate(self, batch_x, batch_y):
        raise NotImplementedError("Overwrite batch_validate in derived class.")

    def fit_data(
        self,
        data: "sized_generator",
        *,
        validation_data: "sized_generator" = None,
        epochs=1,
        verbose=True,
    ):
        train_logs = {}
        self.callbacks.set_parameters({"epochs": epochs})
        self.on_train_begin(train_logs)

        for epoch in optional_loop(range(epochs), verbose=verbose):
            epoch_logs = {}
            self.on_epoch_begin(epoch, epoch_logs)

            for batch, (batch_x, batch_y) in enumerate(
                optional_loop_nest(data, verbose=verbose, label=self.message.get)
            ):
                batch_logs = {}
                self.on_batch_begin(batch, batch_logs)
                self.batch_step(batch_x, batch_y, batch_logs)
                self.on_batch_end(batch, batch_logs)

            if validation_data is not None:
                self.validate(validation_data)

            self.on_epoch_end(epoch, epoch_logs)

        self.on_train_end(train_logs)

        if verbose:
            print("\n")

    def validate(self, validation_data):
        validation_logs = {}
        self.on_validation_begin(validation_logs)

        current = [metric.zero() for metric in self.metrics]

        for batch_x, batch_y in validation_data:
            updates = self.batch_validate(batch_x, batch_y)
            current = [c + u for (c, u) in zip(current, updates)]

        validation_logs["metrics"] = [
            (m.name(), c.get()) for m, c in zip(self.metrics, current)
        ]
        self.on_validation_end(validation_logs)


class BasePredictor:
    def __init__(self, model):
        self.model = model

    def unpack_batch(self, keys, values):
        return unpack(keys, values)

    def pack_prediction(self, x):
        return pack(x)

    def predict_batch(self, x):
        raise NotImplementedError("Overwrite predict_batch in derived class")

    def predict_data(self, data, *, verbose=False):
        batch_keys_values_pairs = []
        for batch_x in optional_loop(data, verbose=verbose, label="predict"):
            pred = self.predict_batch(batch_x)
            batch_keys_values_pairs.append(self.pack_prediction(pred))

        if verbose:
            print("\n")

        result = unpack_batches(batch_keys_values_pairs)
        return result


class BatchedModel:
    """Helper to add simple batched training for numpy based data."""

    trainer = BaseTrainer
    predictor = BasePredictor

    def fit_data(
        self,
        data: "sized_generator",
        *,
        epochs=1,
        callbacks=None,
        verbose=True,
        metrics=None,
        validation_data: "sized_generator" = None,
    ):
        trainer = self.trainer(model=self, callbacks=callbacks, metrics=metrics)
        trainer.fit_data(
            data, epochs=epochs, verbose=verbose, validation_data=validation_data
        )
        return trainer.history

    def predict_data(self, data: "sized_generator", *, verbose=False):
        return self.predictor(model=self).predict_data(data, verbose=verbose)


def optional_loop(iterable, verbose=False, label=None):
    return iterable if not verbose else loop_over(iterable, label=label)


def optional_loop_nest(iterable, verbose=False, label=None):
    return iterable if not verbose else loop_nest(iterable, label=label)


def batched_numpy(
    obj, batch_size=1, shuffle=True, dtype="float32", drop_last=False, prepack=None
):
    """Construct a data-loader like for numpy arrays."""
    if prepack is None:
        prepack = identity

    obj = apply_dtype(dtype, obj)
    keys, values = pack(obj)

    n_samples = get_number_of_samples(*values)
    n_batches = get_number_of_batches(
        n_samples, batch_size=batch_size, only_complete=drop_last
    )

    def generator():
        indices = None if not shuffle else shuffled_indices(n_samples)

        for batch_values in iter_batched(
            values,
            batch_size=batch_size,
            indices=indices,
            shuffle=shuffle,
            only_complete=drop_last,
        ):
            batch_values = tuple(prepack(val) for val in batch_values)
            batch_obj = unpack(keys, batch_values)
            yield batch_obj

    return sized_generator(generator, length=n_batches)


def batched_transformed(
    base,
    transform=None,
    *,
    batch_size=1,
    shuffle=True,
    dtype="float32",
    drop_last=False,
    prepack=None,
):
    """Construct a data-loader like for transformed data."""
    if prepack is None:
        prepack = identity

    if isinstance(base, range):
        base = np.arange(base.start, base.stop, base.step)

    keys, values = pack(base)
    n_samples = get_number_of_samples(*values)
    n_batches = get_number_of_batches(
        n_samples, batch_size=batch_size, only_complete=drop_last
    )

    def decorator(transform):
        def generator():
            indices = None if not shuffle else shuffled_indices(n_samples)

            for batch_values in iter_batched(
                values,
                batch_size=batch_size,
                indices=indices,
                shuffle=shuffle,
                only_complete=drop_last,
            ):
                batch_obj = unpack(keys, batch_values)
                batch_obj = transform(batch_obj)
                batch_obj = apply_dtype(dtype, batch_obj)

                k, v = pack(batch_obj)
                v = tuple(prepack(i) for i in v)
                batch_obj = unpack(k, v)

                yield batch_obj

        return sized_generator(generator, n_batches)

    return decorator if transform is None else decorator(transform)


def identity(x):
    return x


def unpack_batches(batch_keys_values_pairs):
    """Given batchwise-packed results, concatenate and pack them all."""

    all_keys = set()
    all_values = list()

    for keys, values in batch_keys_values_pairs:
        all_keys.add(keys)
        all_values.append(values)

    try:
        keys, = all_keys

    except ValueError:
        raise ValueError("Inconsistent keys")

    values = parallel_concat(all_values)

    return unpack(keys, values)


def parallel_concat(items):
    if not items:
        raise ValueError()

    n_values, = {len(values) for values in items}
    return tuple(
        np.concatenate([values[i] for values in items], axis=0) for i in range(n_values)
    )


def shuffled_indices(n):
    import numpy as np

    res = np.arange(n)
    np.random.shuffle(res)
    return res


def iter_batched(
    data: tuple,
    *,
    batch_size: int = None,
    indices=None,
    only_complete: bool = True,
    shuffle=False,
):
    """Iterate over data in batches.

    :param data:
        the data to iterate over
    :param only_complete:
        if True yield only batches that have exactly ``batch_size`` items
    """
    n_samples = get_number_of_samples(*data)
    index_generator = iter_batch_indices(
        n_samples,
        batch_size=batch_size,
        indices=indices,
        shuffle=shuffle,
        only_complete=only_complete,
    )

    def generator():
        for batch_indices in index_generator:
            yield tuple(multi_getitem(array, batch_indices) for array in data)

    return sized_generator(generator, length=len(index_generator))


def multi_getitem(obj, indices):
    """Get multiple items from one object with support for various types."""
    if type(obj).__module__.startswith("numpy"):
        return obj[indices]

    if type(obj).__module__.startswith("pandas"):
        return obj.iloc[indices]

    return type(obj)([obj[idx] for idx in indices])


def iter_batch_indices(
    n_samples,
    *,
    batch_size: int = None,
    indices=None,
    shuffle=False,
    only_complete=True,
):
    import numpy as np

    if batch_size is None:
        batch_size = default_batch_size

    n_batches = get_number_of_batches(
        n_samples, batch_size=batch_size, only_complete=only_complete
    )

    if n_batches == 0:
        raise ValueError("Not enought items")

    if indices is None:
        indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    def generator():
        for batch in range(n_batches):
            batch_indices = indices[batch * batch_size : (batch + 1) * batch_size]
            yield batch_indices

    return sized_generator(generator, length=n_batches)


def get_number_of_batches(n_samples, batch_size, only_complete=True):
    n_batches = n_samples // batch_size

    if not only_complete and n_samples % batch_size != 0:
        n_batches += 1

    return n_batches


def get_number_of_samples(*values):
    sample_counts = {len(item) for item in values if item is not None}
    if len(sample_counts) != 1:
        raise ValueError("inconsistent batch sizes")

    n_samples, = sample_counts
    return n_samples


def ensure_tuple(obj, length):
    if not isinstance(obj, tuple):
        return tuple(obj for _ in range(length))

    return obj


def apply_dtype(dtype, arg):
    if arg is None and dtype is None:
        return None

    if arg is None and dtype is not None:
        raise ValueError("cannot convert ...")

    if dtype is None:
        return arg

    # "broadcat" scalar dtypes to structured args
    if isinstance(arg, dict) and not isinstance(dtype, dict):
        dtype = {k: dtype for k in arg}

    if isinstance(arg, tuple) and not isinstance(dtype, tuple):
        dtype = tuple(dtype for _ in arg)

    # NOTE: always use the dtype as the reference, to allow passing
    # non-exact type matches, i.e., pandas.DataFrames for dict.
    if isinstance(dtype, dict):
        return collections.OrderedDict(
            (k, apply_dtype(dtype[k], arg[k])) for k in dtype
        )

    elif isinstance(dtype, tuple):
        return tuple(apply_dtype(dtype[i], arg[i]) for i in range(len(dtype)))

    else:
        return np.asarray(arg, dtype=dtype)


class Pack(enum.Enum):
    object = 1
    tuple = 2
    dict = 3


def pack(obj):
    """Pack arguments of different types into a list.

    The ``pack`` / ``unpack`` pair is used to ensure that even with mixed
    arguments internal the logic always sees a list of arrays.

    :returns:
        a tuple of ``keys, values``. The ``keys`` are an opaque hashable
        object that can be used to unpack the values. ``values`` will be
        a tuple of flattend arguments.
    """
    if isinstance(obj, tuple):
        keys = []
        values = []

        for item in obj:
            item_key, item_values = pack(item)
            keys.append(item_key)
            values.extend(item_values)

        return (Pack.tuple, *keys), tuple(values)

    elif isinstance(obj, dict):
        keys = []
        values = []

        for k, item in obj.items():
            item_key, item_values = pack(item)

            keys.append((k, item_key))
            values.extend(item_values)

        return (Pack.dict, *keys), tuple(values)

    else:
        return (Pack.object,), (obj,)


def unpack(key, values):
    """Unpack previously packed parameters.

    Given ``keys`` and ``values`` as returned by ``pack`` reconstruct
    build objects of the same form as the arguments to ``pack``.

    :returns:
        a tuple of the same structure as the arguments to ``pack``.
    """
    obj, rest = _unpack(key, values)
    assert not rest
    return obj


def _unpack(key, values):
    if key[0] == Pack.object:
        return values[0], values[1:]

    elif key[0] == Pack.tuple:
        _, *item_keys = key

        items = []
        for item_key in item_keys:
            item, values = _unpack(item_key, values)
            items.append(item)

        return tuple(items), values

    elif key[0] == Pack.dict:
        _, *item_keys = key

        items = {}
        for (k, item_key) in item_keys:
            item, values = _unpack(item_key, values)
            items[k] = item

        return items, values

    else:
        raise NotImplementedError()


class sized_generator:
    def __init__(self, generator, length):
        self.generator = generator
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter(self.generator())


class Callback:
    """Event handler to monitor / modify training runs.

    Most event handlers have a ``begin_*``, ``end_*`` structure, with a
    ``logs`` argument. For each ``end_*`` call, the same dictionary as for the
    ``begin_*`` call is passed. This mechanism allows to modify the object to
    collect statistics.
    """

    def __init__(self):
        self.model = None

    def set_model(self, model):
        self.model = model

    def set_parameters(self, params):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_validation_begin(self, logs=None):
        pass

    def on_validation_end(self, logs=None):
        pass


class CallbackList(Callback):
    """Wrapper around a list of callbacks, that are called in turn for events."""

    @classmethod
    def make(cls, obj):
        if isinstance(obj, cls):
            return cls(obj.callbacks)

        elif isinstance(obj, (list, tuple)):
            return cls(obj)

        elif obj is None:
            return cls([])

        else:
            return cls([obj])

    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = list(callbacks)

    def append(self, callback):
        self.callbacks.append(callback)

    def set_parameters(self, params):
        for callback in self.callbacks:
            callback.set_parameters(params)

    def on_epoch_begin(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_validation_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_validation_begin(logs)

    def on_validation_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_validation_end(logs)


class History(Callback):
    """Record any epoch statistics generated during training."""

    def __init__(self):
        super().__init__()

        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


class LossHistory(Callback):
    """Record the loss history per batch."""

    def __init__(self):
        super().__init__()
        self.current_epoch = 0

        self.epoch = []
        self.batch = []
        self.loss = []

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_end(self, batch, logs=None):
        self.epoch.append(self.current_epoch)
        self.batch.append(batch)
        self.loss.append(logs.get("loss"))

    def on_train_end(self, logs=None):
        self.current_epoch = None
        self.epoch = np.asanyarray(self.epoch)
        self.batch = np.asarray(self.batch)
        self.loss = np.asarray(self.loss)


class TerminateOnNaN(Callback):
    """Raise an exception when the loss becomes nan."""

    def __init__(self):
        super().__init__()
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_end(self, batch, logs=None):
        loss = logs.get("loss")
        if loss is not None and not np.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss {loss} in epoch {self.current_epoch}, batch {batch}"
            )


class FitMessage(Callback):
    def __init__(self):
        super().__init__()
        self.current_loss = None
        self.current_epoch = None
        self.epochs = None
        self.current_metrics = None

    def get(self):
        def loss_fragments():
            if self.current_loss is None:
                return

            yield "{:.3g}".format(self.current_loss)
            yield " "

        def metric_fragments():
            if self.current_metrics is None:
                return

            for label, value in self.current_metrics:
                yield "{}: {:.3g}".format(label, value)
                yield " "

        def format_epoch_number(epoch):
            return "{:,d}".format(epoch) if epoch is not None else "?"

        fragments = [
            *loss_fragments(),
            *metric_fragments(),
            "(",
            format_epoch_number(self.current_epoch),
            " / ",
            format_epoch_number(self.epochs),
            ")",
        ]
        return "".join(fragments)

    def set_parameters(self, params):
        self.epochs = params.get("epochs")

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_end(self, batch, logs=None):
        self.current_loss = optional_dict(logs).get("loss")

    def on_validation_end(self, logs=None):
        self.current_metrics = optional_dict(logs).get("metrics", None)


def optional_dict(d=None):
    return d if d is not None else {}
