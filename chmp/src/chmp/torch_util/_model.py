import torch

from ._batched import BatchedModel, Callback, BaseTrainer, BasePredictor


class TorchTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = None

    def on_train_begin(self, logs):
        super().on_train_begin(logs)

        self.optimizer = self.model.optimizer(
            self.model.module.parameters(), **self.model.optimizer_kwargs
        )

    def unpack_batch(self, keys, values):
        values = list_as_tensor(values)
        return super().unpack_batch(keys, values)

    def batch_step(self, batch_x, batch_y, logs):
        assert self.optimizer is not None

        def closure():
            self.optimizer.zero_grad()
            batch_pred = self.model.call_module(batch_x)

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
    def unpack_batch(self, keys, values):
        values = list_as_tensor(values)
        return super().unpack_batch(keys, values)

    def pack_prediction(self, x):
        keys, values = super().pack_prediction(x)
        values = list_torch_to_numpy(values)
        return keys, values

    def predict_batch(self, x):
        return self.model.call_module(x)


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

    def call_module(self, x):
        if isinstance(x, (tuple, list)):
            return self.module(*x)

        elif isinstance(x, dict):
            return self.module(**x)

        else:
            return self.module(x)


class LearningRateScheduler(Callback):
    def __init__(self, cls, **kwargs):
        super().__init__()
        self.cls = cls
        self.kwargs = kwargs

    def on_train_begin(self, logs=None):
        self.scheduler = self.cls(self.model.optimizer, **self.kwargs)

    def on_epoch_begin(self, epoch, logs=None):
        self.scheduler.step()


def list_as_tensor(values):
    return [torch.as_tensor(val) if val is not None else None for val in values]


def list_torch_to_numpy(values):
    return [t.detach().numpy() for t in values]
