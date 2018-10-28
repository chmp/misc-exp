import collections
import numpy as np

import torch

from chmp.ds import loop_over, loop_nest
from .data import data_loader
from ._metrics import ensure_metric
from ._util import pack, unpack, default_batch_size

__all__ = ["Model", "Callback", "History", "LossHistory", "LearningRateScheduler"]


class Trainer:
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

        self.optimizer = None

    def on_train_begin(self, logs):
        self.optimizer = self.model.optimizer(
            self.model.module.parameters(), **self.model.optimizer_kwargs
        )
        self.callbacks.on_train_begin(logs)

    def on_train_end(self, logs):
        self.optimizer = None
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

    def fit_data(self, data, *, validation_data=None, epochs=1, verbose=True):
        train_logs = {}
        self.callbacks.set_parameters({"epochs": epochs})
        self.on_train_begin(train_logs)

        for epoch in optional_loop(range(epochs), verbose=verbose):
            epoch_logs = {"epoch": epoch}
            self.on_epoch_begin(epoch, epoch_logs)

            train_loss = []

            for batch, (batch_x, batch_y) in enumerate(
                optional_loop_nest(data, verbose=verbose, label=self.message.get)
            ):
                batch_logs = {"batch": batch}
                self.on_batch_begin(batch, batch_logs)
                self.batch_step(batch_x, batch_y, batch_logs)
                train_loss.append(batch_logs.get("loss"))
                self.on_batch_end(batch, batch_logs)

            if validation_data is not None:
                validation_logs = self.validate(validation_data)
                epoch_logs.update(validation_logs)

            epoch_logs["loss"] = np.mean(train_loss)

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

        validation_logs["metrics"] = collections.OrderedDict(
            [(m.name(), c.get()) for m, c in zip(self.metrics, current)]
        )
        self.on_validation_end(validation_logs)
        return validation_logs

    def batch_step(self, batch_x, batch_y, logs):
        assert self.optimizer is not None

        def closure():
            self.optimizer.zero_grad()
            batch_pred = call_module(self.model.module, batch_x)

            loss = self._compute_loss(batch_pred, batch_y)
            loss = self._add_regularization(loss)

            loss.backward()
            return loss

        loss = self.optimizer.step(closure)
        logs["loss"] = float(loss)

    def batch_validate(self, batch_x, batch_y):
        pred = call_module(self.model.module, batch_x)
        return [metric.compute(pred, batch_y) for metric in self.metrics]

    def _compute_loss(self, pred, y):
        if self.model.loss is not None:
            return self.model.loss(pred, y)

        else:
            return pred

    def _add_regularization(self, loss):
        if self.model.regularization is not None:
            return loss + self.model.regularization(self.model.module)

        else:
            return loss


class Predictor:
    def __init__(self, model):
        self.model = model

    def unpack_batch(self, keys, values):
        return unpack(keys, values)

    def predict_data(self, data, *, verbose=False):
        batch_keys_values_pairs = []
        for batch_x in optional_loop(data, verbose=verbose, label="predict"):
            pred = self.predict_batch(batch_x)
            batch_keys_values_pairs.append(self.pack_prediction(pred))

        if verbose:
            print("\n")

        result = unpack_batches(batch_keys_values_pairs)
        return result

    def pack_prediction(self, x):
        keys, values = pack(x)
        values = list_as_numpy(values)
        return keys, values

    def predict_batch(self, x):
        return call_module(self.model.module, x)


class Model:
    """Keras-like API around a torch models.

    :param module:
        the module that defines the model prediction
    :param optimizer:
        the optimizer to use. Either a callable or string specifying an
        optimizer in `torch.optim`.
    :param optimizer_kwargs:
        keyword arguments passed to the optimizer before building it.
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

    __members__ = [
        "fit",
        "predict",
        "fit_transformed",
        "predict_transformed",
        "fit_data",
        "predict_data",
    ]

    trainer = Trainer
    predictor = Predictor

    def __init__(
        self,
        module,
        optimizer="Adam",
        loss=None,
        regularization=None,
        optimizer_kwargs=None,
    ):
        super().__init__()
        if isinstance(optimizer, str):
            optimizer = getattr(torch.optim, optimizer)

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        self.module = module
        self.optimizer = optimizer
        self.loss = loss
        self.regularization = regularization
        self.optimizer_kwargs = optimizer_kwargs

    def fit(
        self,
        x,
        y,
        *,
        epochs=1,
        batch_size=default_batch_size,
        dtype="float32",
        verbose=True,
        callbacks=None,
        metrics=None,
        validation_data=None,
    ):
        if validation_data is not None:
            validation_data = data_loader(
                "numpy",
                validation_data,
                batch_size=batch_size,
                dtype=dtype,
                mode="predict",
            )

        return self.fit_data(
            data_loader(
                "numpy", (x, y), batch_size=batch_size, dtype=dtype, mode="fit"
            ),
            epochs=epochs,
            verbose=verbose,
            metrics=metrics,
            validation_data=validation_data,
            callbacks=callbacks,
        )

    def predict(self, x, batch_size=default_batch_size, dtype="float32", verbose=False):
        return self.predict_data(
            data_loader("numpy", x, batch_size=batch_size, dtype=dtype, mode="predict"),
            verbose=verbose,
        )

    def fit_transformed(
        self,
        transform,
        *bases,
        epochs=1,
        batch_size=default_batch_size,
        dtype="float32",
        verbose=True,
        callbacks=None,
        metrics=None,
        collate_fn=None,
        validation_data=None,
    ):
        if validation_data is not None:
            if not isinstance(validation_data, tuple):
                validation_data = (validation_data,)

            validation_data = data_loader(
                "transformed",
                transform,
                *validation_data,
                batch_size=batch_size,
                dtype=dtype,
                mode="predict",
                collate_fn=collate_fn,
            )

        return self.fit_data(
            data_loader(
                "transformed",
                transform,
                *bases,
                batch_size=batch_size,
                dtype=dtype,
                mode="fit",
                collate_fn=collate_fn,
            ),
            epochs=epochs,
            verbose=verbose,
            metrics=metrics,
            validation_data=validation_data,
            callbacks=callbacks,
        )

    def predict_transformed(
        self,
        transform,
        *bases,
        batch_size=default_batch_size,
        dtype="float32",
        verbose=False,
        collate_fn=None,
    ):
        return self.predict_data(
            data_loader(
                "transformed",
                transform,
                *bases,
                batch_size=batch_size,
                dtype=dtype,
                mode="predict",
                collate_fn=collate_fn,
            ),
            verbose=verbose,
        )

    def fit_data(
        self,
        data,
        *,
        epochs=1,
        callbacks=None,
        verbose=True,
        metrics=None,
        validation_data=None,
    ):
        trainer = self.trainer(model=self, callbacks=callbacks, metrics=metrics)
        trainer.fit_data(
            data, epochs=epochs, verbose=verbose, validation_data=validation_data
        )
        return trainer.history

    def predict_data(self, data, *, verbose=False):
        return self.predictor(model=self).predict_data(data, verbose=verbose)


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


class LearningRateScheduler(Callback):
    def __init__(self, cls, **kwargs):
        super().__init__()
        self.cls = cls
        self.kwargs = kwargs

    def on_train_begin(self, logs=None):
        self.scheduler = self.cls(self.model.optimizer, **self.kwargs)

    def on_epoch_begin(self, epoch, logs=None):
        self.scheduler.step()


def call_module(module, x):
    if isinstance(x, (tuple, list)):
        return module(*x)

    elif isinstance(x, dict):
        return module(**x)

    else:
        return module(x)


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
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = optional_dict(logs)
        logs = flatten_dict(logs)

        self.history.append(logs)

    def get(self, key, default=None):
        return [item.get(key, default) for item in self.history]

    def __iter__(self):
        return iter(self.history)


def flatten_dict(d):
    r = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for ck, cv in flatten_dict(v).items():
                ck = (k, *ck) if isinstance(ck, tuple) else (k, ck)
                r[ck] = cv

        else:
            r[k] = v

    return r


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

            for label, value in self.current_metrics.items():
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


def list_as_tensor(values):
    return [torch.as_tensor(val) if val is not None else None for val in values]


def list_as_numpy(values):
    return [t.detach().numpy() for t in values]


def optional_loop(iterable, verbose=False, label=None):
    return iterable if not verbose else loop_over(iterable, label=label)


def optional_loop_nest(iterable, verbose=False, label=None):
    return iterable if not verbose else loop_nest(iterable, label=label)


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
