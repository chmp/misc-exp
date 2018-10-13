import itertools as it
import numpy as np
import torch

from chmp.ds import loop_over, loop_nest

default_batch_size = 32


class TorchModel:
    """Helper to add simple numpy integration to torch model.

    :param module:
        the module that defines the model prediction
    :param optimizer:
        a factory for the optimizer to use
    :param loss:
        the ``loss`` function to use, with signature
        ``(pred, target) -> loss``. If ``None``, the module is assumed to
        return the loss itself.
    :param regularization:
        if given a callable, with signature ``(module) -> loss``, that should
        return a regularization loss

    For all functions ``x`` and ``y`` can not only be ``numpy`` arrays, but
    also structured data, such as dicts or lists / tuples. The former are
    passed to the module as keyword arguments, the latter as varargs.

    For example::

        # NOTE: this module does not define parameters
        class Model(torch.nn.Module):
            def forward(self, a, b):
                return a + b


        model = TorchModel(module=Model, loss=MSELoss())
        model.fit(x={"a": [...], "b": [...]}, y=[...])

    """

    def __init__(
        self,
        module,
        optimizer="Adam",
        loss=None,
        regularization=None,
        optimizer_kwargs=None,
    ):
        if isinstance(optimizer, str):
            optimizer = getattr(torch.optim, optimizer)

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        self.module = module
        self.optimizer_factory = optimizer
        self.loss = loss
        self.regularization = regularization
        self.optimizer_kwargs = optimizer_kwargs

        self.optimizer = None

    def fit(
        self,
        x=None,
        y=None,
        *,
        batch_size=None,
        epochs=1,
        shuffle=True,
        verbose=False,
        callbacks=None,
        dtype="float32",
    ):
        dtype_x, dtype_y = ensure_tuple(dtype, 2)
        x = apply_dtype(dtype_x, x)
        y = apply_dtype(dtype_y, y)
        keys, values = pack(x, y)
        n_samples = get_number_of_samples(*values)

        self.optimizer = self._build_optimizer()

        # NOTE: may require optimizer
        callbacks, history = self._build_callbacks(callbacks)

        train_logs = {}
        callbacks.on_train_begin(train_logs)

        for epoch in optional_loop(range(epochs), verbose=verbose):
            epoch_logs = {}
            callbacks.on_epoch_begin(epoch, epoch_logs)

            indices = None if not shuffle else shuffled_indices(n_samples)

            msg = self._fit_message(epoch, epochs)

            for batch, batch_values in enumerate(
                optional_loop_nest(
                    iter_batched(values, batch_size=batch_size, indices=indices),
                    verbose=verbose,
                    label=lambda: msg,
                )
            ):
                batch_logs = {}

                callbacks.on_batch_begin(batch, batch_logs)
                batch_values = list_as_tensor(batch_values)
                batch_x, batch_y = unpack(keys, batch_values)

                batch_logs["loss"] = self._batch_step(self.optimizer, batch_x, batch_y)
                msg = self._fit_message(epoch, epochs, batch_logs["loss"])
                callbacks.on_batch_end(batch, batch_logs)

            callbacks.on_epoch_end(epoch, epoch_logs)

        callbacks.on_train_end(train_logs)

        if verbose:
            print("\n")

        return history

    def fit_generator(
        self,
        generator,
        *,
        steps_per_epoch=1,
        epochs=None,
        verbose=True,
        callbacks=None,
        dtype="float32",
    ):
        """Fit the model on a dynamically generated dataset.

        :param generator:
            A generator yielding ``batch_x, batch_y`` pairs.
        :param steps_per_epoch:
            The number batches that make up an epoch.
        :param epochs:
            The number of epochs to evaluate. If ``None``, the generator must be
            finite.
        :param verbose:
        :returns:
            itself.
        """
        dtype_x, dtype_y = ensure_tuple(dtype, 2)

        generator = iter(generator)
        epoch_sequence = it.count() if epochs is None else range(epochs)
        self.optimizer = self._build_optimizer()

        # NOTE: may require optimizer
        callbacks, history = self._build_callbacks(callbacks)
        train_logs = {}
        callbacks.on_train_begin(train_logs)

        for epoch in optional_loop(epoch_sequence, verbose=verbose):
            epoch_logs = {}
            callbacks.on_epoch_begin(epoch, epoch_logs)

            msg = self._fit_message(epoch, epochs)

            for batch in optional_loop_nest(
                range(steps_per_epoch), verbose=verbose, label=lambda: msg
            ):
                try:
                    batch_x, batch_y = next(generator)

                except StopIteration:
                    if epochs is None:
                        break

                    else:
                        raise RuntimeError(
                            "Generator did not yield enough batches for fit."
                        )

                batch_logs = {}
                callbacks.on_batch_begin(batch, batch_logs)

                batch_x = apply_dtype(dtype_x, batch_x)
                batch_y = apply_dtype(dtype_y, batch_y)
                batch_x, batch_y = generic_as_tensor(batch_x, batch_y)

                batch_logs["loss"] = self._batch_step(self.optimizer, batch_x, batch_y)
                msg = self._fit_message(epoch, epochs, batch_logs["loss"])

                callbacks.on_batch_end(batch, batch_logs)

            # if the epoch was not aborted
            else:
                callbacks.on_epoch_end(epoch, epoch_logs)
                continue

            # else break
            break

        callbacks.on_train_end(train_logs)

        if verbose:
            print("\n")

        return history

    def _fit_message(self, epoch, epochs, loss=None):
        epochs_format = ",d" if epochs is not None else ""
        epochs = epochs if epochs is not None else "?"

        if loss is None:
            return f"(epoch {epoch:,d} / {epochs:{epochs_format}})"

        else:
            return f"{loss:.3g} (epoch {epoch:,d} / {epochs:{epochs_format}})"

    def _build_callbacks(self, callbacks):
        callbacks = CallbackList.ensure_callback(callbacks)
        history = History()
        callbacks.append(history)
        callbacks.set_model(self)

        return callbacks, history

    def _build_optimizer(self):
        return self.optimizer_factory(self.module.parameters(), **self.optimizer_kwargs)

    def _batch_step(self, optimizer, batch_x, batch_y):
        def closure():
            optimizer.zero_grad()
            batch_pred = self._call_module(batch_x)

            loss = self._compute_loss(batch_pred, batch_y)
            loss = self._add_regularization(loss)

            loss.backward()
            return loss

        loss = optimizer.step(closure)

        return float(loss)

    def _compute_loss(self, pred, y):
        if self.loss is not None:
            return self.loss(pred, y)

        else:
            return pred

    def _add_regularization(self, loss):
        if self.regularization is not None:
            return loss + self.regularization(self.module)

        else:
            return loss

    def predict(self, x=None, batch_size=None, verbose=False, dtype="float32"):
        x = apply_dtype(dtype, x)
        keys, values = pack(x)
        batch_keys_values_pairs = []

        for batch_values in optional_loop(
            iter_batched(values, batch_size=batch_size, only_complete=False),
            verbose=verbose,
            label="predict",
        ):
            batch_values = list_as_tensor(batch_values)
            batch_x, = unpack(keys, batch_values)

            pred = self._call_module(batch_x)

            pred_keys, pred_values = pack(pred)
            pred_values = list_torch_to_numpy(pred_values)

            batch_keys_values_pairs.append((pred_keys, pred_values))

        if verbose:
            print("\n")

        result, = unpack_batches(batch_keys_values_pairs)
        return result

    def predict_generator(
        self, generator, *, steps=None, verbose=True, dtype="float32"
    ):
        """Predict on a generator.

        :param generator:
            an iterable, that will be used to get batches for prediction.
        :param steps:
            the number of times the generator should be called. if ``steps`` is
            ``None``, the generator all items of the generator will be
            processed. Therefore, the generator should only yield a finite
            number of items in this case.
        :param verbose:
        :returns:
            the predictions as a numpy array.
        """
        generator = iter(generator)
        batch_keys_values_pairs = []
        step_sequence = it.count() if steps is None else range(steps)

        for _ in optional_loop(step_sequence, verbose=verbose, label="predict"):
            try:
                batch_x = next(generator)

            except StopIteration:
                if steps is None:
                    break

                else:
                    raise RuntimeError(
                        "Generator did not yield enough batches for predict"
                    )

            batch_x = apply_dtype(dtype, batch_x)
            batch_x, = generic_as_tensor(batch_x)

            pred = self._call_module(batch_x)

            pred_keys, pred_values = pack(pred)
            pred_values = list_torch_to_numpy(pred_values)

            batch_keys_values_pairs.append((pred_keys, pred_values))

        if verbose:
            print("\n")

        result, = unpack_batches(batch_keys_values_pairs)
        return result

    def _call_module(self, x):
        if isinstance(x, (tuple, list)):
            return self.module(*x)

        elif isinstance(x, dict):
            return self.module(**x)

        else:
            return self.module(x)


def generic_as_tensor(*objs):
    keys, values = pack(*objs)
    values = list_as_tensor(values)
    return unpack(keys, values)


def list_as_tensor(values):
    return [torch.tensor(val) if val is not None else None for val in values]


def list_torch_to_numpy(values):
    return [t.detach().numpy() for t in values]


def optional_loop(iterable, verbose=False, label=None):
    return iterable if not verbose else loop_over(iterable, label=label)


def optional_loop_nest(iterable, verbose=False, label=None):
    return iterable if not verbose else loop_nest(iterable, label=label)


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
        the data to iterate over has to be a dicitonary
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
            yield tuple(array[batch_indices] for array in data)

    return sized_generator(generator, length=len(index_generator))


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

    n_batches = n_samples // batch_size

    if not only_complete and n_samples % batch_size != 0:
        n_batches += 1

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


def get_number_of_samples(*values):
    sample_counts = {item.shape[0] for item in values if item is not None}
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
        return {k: np.asarray(arg[k], dtype=dtype[k]) for k in dtype}

    elif isinstance(dtype, tuple):
        return tuple(np.asarray(arg[i], dtype=dtype[i]) for i in range(len(dtype)))

    else:
        return np.asarray(arg, dtype=dtype)


def pack(*args):
    """Pack arguments of different types into a list.

    The ``pack`` / ``unpack`` pair is used to ensure that even with mixed
    arguments internal the logic always sees a list of arrays.

    :returns:
        a tuple of ``keys, values``. The ``keys`` are an opaque hashable
        object that can be used to unpack the values. ``values`` will be
        a tuple of flattend arguments.
    """
    keys = []
    values = []

    for arg in args:
        if isinstance(arg, tuple):
            keys.append(len(arg))
            values.extend(arg)

        elif isinstance(arg, dict):
            k, v = zip(*arg.items())
            keys.append(tuple(k))
            values.extend(v)

        else:
            keys.append(None)
            values.append(arg)

    return tuple(keys), tuple(values)


def unpack(keys, values):
    """Unpack previously packed parameters.

    Given ``keys`` and ``values`` as returned by ``pack`` reconstruct
    build objects of the same form as the arguments to ``pack``.

    :returns:
        a tuple of the same structure as the arguments to ``pack``.
    """
    offset = 0
    result = []

    for key in keys:
        if key is None:
            result.append(values[offset])
            offset += 1

        elif isinstance(key, tuple):
            result.append(
                {k: v for k, v in zip(key, values[offset : offset + len(key)])}
            )
            offset += len(key)

        else:
            result.append(tuple(v for v in values[offset : offset + key]))
            offset += key

    return tuple(result)


class sized_generator:
    def __init__(self, generator, length):
        self.generator = generator
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter(self.generator())


class Callback:
    def __init__(self):
        self.model = None

    def set_model(self, model):
        self.model = model

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


class CallbackList(Callback):
    @classmethod
    def ensure_callback(cls, obj):
        if isinstance(obj, cls):
            return obj

        elif isinstance(obj, (list, tuple)):
            return cls(obj)

        elif obj is None:
            return cls([])

        else:
            return cls([obj])

    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = callbacks

    def append(self, callback):
        self.callbacks.append(callback)

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


class History(Callback):
    def __init__(self):
        super().__init__()

        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


class LossHistory(Callback):
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


class LearningRateScheduler(Callback):
    def __init__(self, cls, **kwargs):
        super().__init__()
        self.cls = cls
        self.kwargs = kwargs

    def on_train_begin(self, logs=None):
        self.scheduler = self.cls(self.model.optimizer, **self.kwargs)

    def on_epoch_begin(self, epoch, logs=None):
        self.scheduler.step()
