import torch

from ._batched import (
    BatchedModel,
    Callback,
    BaseTrainer,
    BasePredictor,
    batched_numpy,
    batched_transformed,
    default_batch_size,
)


class TorchTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = None

    def on_train_begin(self, logs):
        super().on_train_begin(logs)

        self.optimizer = self.model.optimizer(
            self.model.module.parameters(), **self.model.optimizer_kwargs
        )

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

    def on_train_end(self, logs):
        super().on_train_end(logs)
        self.optimizer = None


class TorchPredictor(BasePredictor):
    def pack_prediction(self, x):
        keys, values = super().pack_prediction(x)
        values = list_as_numpy(values)
        return keys, values

    def predict_batch(self, x):
        return call_module(self.model.module, x)


class TorchModel(BatchedModel):
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

    trainer = TorchTrainer
    predictor = TorchPredictor

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
    ):
        self.fit_data(
            numpy_fit(x, y, batch_size=batch_size, dtype=dtype),
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
        )

    def fit_transformed(
        self,
        base,
        transform,
        *,
        epochs=1,
        batch_size=default_batch_size,
        dtype="float32",
        verbose=True,
        callbacks=None,
    ):
        return self.fit_data(
            transformed_fit(base, transform, batch_size=batch_size, dtype=dtype),
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
        )

    def predict(self, x, batch_size=default_batch_size, dtype="float32", verbose=False):
        return self.predict_data(
            numpy_predict(x, batch_size=batch_size, dtype=dtype), verbose=verbose
        )

    def predict_transformed(
        self,
        base,
        transform,
        batch_size=default_batch_size,
        dtype="float32",
        verbose=False,
    ):
        return self.predict_data(
            transformed_predict(base, transform, batch_size=batch_size, dtype=dtype),
            verbose=verbose,
        )


def numpy_fit(*objs, batch_size=default_batch_size, dtype="float32"):
    return batched_numpy(
        *objs,
        batch_size=batch_size,
        dtype=dtype,
        shuffle=True,
        drop_last=True,
        prepack=torch.as_tensor,
    )


def numpy_predict(*objs, batch_size=default_batch_size, dtype="float32"):
    return batched_numpy(
        *objs,
        batch_size=batch_size,
        dtype=dtype,
        shuffle=False,
        drop_last=False,
        prepack=torch.as_tensor,
    )


def transformed_fit(
    base, transform=None, *, batch_size=default_batch_size, dtype="float32"
):
    return batched_transformed(
        base,
        transform,
        batch_size=batch_size,
        dtype=dtype,
        shuffle=True,
        drop_last=True,
        prepack=torch.as_tensor,
    )


def transformed_predict(
    base, transform=None, *, batch_size=default_batch_size, dtype="float32"
):
    return batched_transformed(
        base,
        transform,
        batch_size=batch_size,
        dtype=dtype,
        shuffle=False,
        drop_last=False,
        prepack=torch.as_tensor,
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


def call_module(module, x):
    if isinstance(x, (tuple, list)):
        return module(*x)

    elif isinstance(x, dict):
        return module(**x)

    else:
        return module(x)


def list_as_tensor(values):
    return [torch.as_tensor(val) if val is not None else None for val in values]


def list_as_numpy(values):
    return [t.detach().numpy() for t in values]
