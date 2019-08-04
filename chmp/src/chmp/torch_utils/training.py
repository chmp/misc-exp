"""Helpers to work with ignite"""
import io
import logging
import pathlib
from typing import Callable, TypeVar

import torch
from ignite.engine import Engine, Events
from ignite.handlers import TerminateOnNan, ModelCheckpoint

from chmp.ds import Debouncer, Loop, sapply, setdefaultattr, singledispatch_on
from . import default_call_model, n2t, t2n

_logger = logging.getLogger(__name__)

T = TypeVar("T")


def disable_ignite_logging():
    logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)


def restore(engine, map_location=None, optional=True):
    checkpointers = list(SimpleCheckpointer.find(engine))

    if len(checkpointers) == 0:
        raise RuntimeError("Need a SimpleCheckpointer instance for restore")

    elif len(checkpointers) > 1:
        raise RuntimeError("Cannot handle multiple SimpleCheckpointer instances")

    checkpointer, = checkpointers
    checkpointer.restore(map_location=map_location, optional=optional)


def attach_all(engine, *handlers):
    for handler in handlers:
        attach(engine, handler)

    return engine


def continue_run(engine, data_loader, max_epochs=1):
    """Helper to continue running an ignite engine."""
    if engine.has_event_handler(restore_engine_state, Events.STARTED):
        engine.remove_event_handler(restore_engine_state, Events.STARTED)

    state_vars = vars(engine.state) if engine.state is not None else {}
    engine.add_event_handler(Events.STARTED, restore_engine_state, state_vars)

    engine.run(data_loader, max_epochs=max_epochs)


def restore_engine_state(engine, state_vars):
    ignored_attrs = {"max_epochs", "dataloader"}

    for k, v in state_vars.items():
        if k.startswith("_") or k in ignored_attrs:
            continue

        setattr(engine.state, k, v)


def make_train_func(
    model,
    loss_func,
    optimizer,
    dtype=None,
    device=None,
    call_model=None,
    get_train_loss=None,
):
    """Create a train func for ``ignite``.

    This function assumes that each batch is of the form ``(x, y)`` with no assumptions placed on ``x`` or ``y``.
    Each batch will be transformed into a ``torch.Tensor``.

    :param model:
        the model to optimize, it will be called with the features
    :param loss_func:
        the loss function to optimize, it will be called with the
        model output and the targets. The return value of the loss
        function must be compatible with ``get_train_loss``.
    :param optimizer:
        the optimizer to use
    :param dtype:
        the dtype of the batch, can be a structured object, e.g., a tuple of dtypes
    :param device:
        the device to assign to the batch, can be a structured object, e.g.,
        a tuple of devices
    :param call_model:
        instead of calling the model directly, ``call_model(model, x)`` it will be used.
        If not given a default implementation is used, that passes ``x`` as varargs if it is a tuple,
        as keyword args if it is a dict and directly otherwise.
    :param get_train_loss:
        The output of ``loss_func`` will be passed through ``get_train_loss`` before calling backward.
        If not given a default implementation is used that takes the first item of the loss if it is a tuple
        and the loss directly otherwise.
    """

    def default_get_train_loss(loss):
        return loss[0] if isinstance(loss, tuple) else loss

    if call_model is None:
        call_model = default_call_model

    if get_train_loss is None:
        get_train_loss = default_get_train_loss

    def train_func(engine, batch):
        x, y = n2t(batch, dtype=dtype, device=device)

        optimizer.zero_grad()

        pred = call_model(model, x)
        loss = loss_func(pred, y)

        train_loss = get_train_loss(loss)
        train_loss.backward()

        optimizer.step()

        return t2n(loss)

    return train_func


@singledispatch_on(1)
def attach(engine, handler):
    return handler.attach(engine)


@attach.register(TerminateOnNan)
def register_terminate_on_nan(engine, handler):
    engine.add_event_handler(Events.ITERATION_COMPLETED, handler)


class SubclassHandler:
    @classmethod
    def find(cls, engine):
        seen = set()

        for l in engine._event_handlers.values():
            for handler, *_ in l:
                # dealt with bound methods
                handler = getattr(handler, "__self__", handler)

                if id(handler) in seen:
                    continue

                seen.add(id(handler))

                if isinstance(handler, cls):
                    yield handler

    def attach(self: T, engine: Engine) -> T:
        for event in [
            Events.EPOCH_STARTED,
            Events.EPOCH_COMPLETED,
            Events.STARTED,
            Events.COMPLETED,
            Events.ITERATION_STARTED,
            Events.ITERATION_COMPLETED,
            Events.EXCEPTION_RAISED,
        ]:
            handler_name = "on_{}".format(event.value)
            default_handler = getattr(SubclassHandler, handler_name)
            handler = getattr(self, handler_name, None)
            unbound_handler = getattr(handler, "__func__", None)

            if handler is None or unbound_handler is default_handler:
                continue

            engine.add_event_handler(event, handler)

        return self

    def on_epoch_started(self, engine: Engine) -> None:
        pass

    def on_epoch_completed(self, engine: Engine) -> None:
        pass

    def on_started(self, engine: Engine) -> None:
        pass

    def on_completed(self, engine: Engine) -> None:
        pass

    def on_iteration_started(self, engine: Engine) -> None:
        pass

    def on_iteration_completed(self, engine: Engine) -> None:
        pass

    def on_exception_raised(self, engine: Engine, exception: Exception) -> None:
        pass


class ProgressBar(SubclassHandler):
    """A progress bar for an ignite ``Engine``.

    Usage::

        ProgressBar().attach(engine)
    """

    def __init__(self, formatter=None, lab=False):
        if formatter is None:
            formatter = ProgressBar.default_formatter

        self.formatter = formatter
        self.lab = lab

        self.loop = None
        self.outer_frame = None
        self.inner_frame = None

    def on_started(self, engine):
        self.loop = None
        self.outer_frame = None
        self.inner_frame = None

    def on_epoch_started(self, engine):
        if self.loop is None:
            self.loop = Loop()
            self.outer_frame = self.loop.push(
                engine.state.max_epochs - engine.state.epoch + 1
            )

        if self.inner_frame is not None:
            self.loop.pop(self.inner_frame)

        self.inner_frame = self.loop.push(len(engine.state.dataloader))

    def on_iteration_completed(self, engine):
        self.inner_frame.finish_item()
        self.print_status(engine)

    def on_epoch_completed(self, engine):
        if self.loop is not None:
            self.loop.pop(self.inner_frame)

        self.inner_frame = None

        self.outer_frame.finish_item()
        self.print_status(engine)

    def on_completed(self, engine: Engine):
        if self.loop is not None:
            self.loop.pop(self.outer_frame)

        self.outer_frame = None
        self.inner_frame = None
        self.loop = None

    def print_status(self, engine):
        if not self.loop.debouncer.should_run():
            return

        self.loop.debouncer.invoked()
        self.update(self.formatter(self.loop, engine.state))

    def update(self, value):
        if self.loop is None:
            return

        self.loop._static_print(value, lab=self.lab)

    @staticmethod
    def default_formatter(loop, state):
        return f'{loop} [{state.epoch} / {state.max_epochs}] {nested_format(state.output, ".3g")}'


def nested_to_float(obj):
    return sapply(float, obj)


def nested_format(obj, fmt):
    if isinstance(obj, tuple):
        return "(" + ", ".join(nested_format(child, fmt) for child in obj) + ")"

    elif isinstance(obj, dict):
        return (
            "{"
            + ", ".join(
                "{!r}: {!s}".format(k, nested_format(v, fmt)) for k, v in obj.items()
            )
            + "}"
        )

    else:
        return format(obj, fmt)


class TrainHistory(SubclassHandler):
    def __init__(self, process_output=nested_to_float):
        self.process_output = process_output

    def on_started(self, engine: Engine):
        setdefaultattr(engine.state, "history_output", [])
        setdefaultattr(engine.state, "history_iteration", [])
        setdefaultattr(engine.state, "history_epoch", [])

    def on_iteration_completed(self, engine):
        engine.state.history_output.append(self.process_output(engine.state.output))
        engine.state.history_iteration.append(engine.state.iteration)
        engine.state.history_epoch.append(engine.state.epoch)


class OutputPlot(SubclassHandler):
    def __init__(self, plotting_func: Callable[[], None], interval=10):
        self.plotting_func = plotting_func
        self.handle = None
        self.debouncer = Debouncer(interval)

    def on_started(self, engine):
        self.update()

    def on_iteration_completed(self, engine):
        self.update()

    def update(self):
        if not self.debouncer.should_run():
            return

        self.debouncer.invoked()

        image = self.get_plot_as_image()

        if self.handle is None:
            from IPython.core.display import display

            self.handle = display(image, display_id=True)

        else:
            self.handle.update(image)

    def get_plot_as_image(self):
        import matplotlib.pyplot as plt
        from IPython.core.display import Image

        with io.BytesIO() as fobj:
            self.call_plotting_func()
            plt.savefig(fobj, format="png")
            plt.close()

            return Image(data=fobj.getvalue(), format="png")

    def call_plotting_func(self):
        import matplotlib.pyplot as plt

        if callable(self.plotting_func):
            self.plotting_func()

        elif isinstance(self.plotting_func, TrainHistory):
            plt.plot(self.plotting_func.output)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")

        else:
            raise RuntimeError("Unknown plotting func type")


class Evaluator(SubclassHandler):
    def __init__(self, valid_data, metrics, process_func):
        self.valid_data = valid_data
        self.process_func = process_func
        self.evaluator = Engine(process_func)

        for name, metric in metrics.items():
            metric.attach(self.evaluator, name)

    def on_epoch_completed(self, engine):
        self.evaluator.run(self.valid_data)

        setdefaultattr(engine.state, "metrics_history", []).append(
            dict(self.evaluator.state.metrics)
        )
        setdefaultattr(engine.state, "metrics_history_epoch", []).append(
            engine.state.epoch
        )


class PersistentState(SubclassHandler):
    """A handler to add support for resuming training in an engine.

    For ease of usage, ``PersistentState`` supports the singleton pattern:

        attach_all(
            engine,
            PersistentState.get(),
        )

    """

    non_persistent_keys = {"dataloader", "max_epochs", "output"}
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()

        return cls._instance

    @classmethod
    def clear(cls):
        cls._instance = None

    def __init__(self):
        self.state = {}

    def on_started(self, engine):
        # copy the previous persisted state
        for k, v in self.state.items():
            setattr(engine.state, k, v)

    def on_epoch_completed(self, engine):
        for k, v in vars(engine.state).items():
            if k.startswith("_") or k in self.non_persistent_keys:
                continue

            self.state[k] = v

    # for compat with checkpointer
    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.state = state


class SimpleCheckpointer(SubclassHandler):
    """Helper to simplify usage ``ignite.handlers.ModelCheckpoint``."""

    def __init__(
        self,
        model,
        optimizer,
        state=None,
        *,
        dirname=".",
        filename_prefix="checkpoint",
        save_interval=None,
        n_saved=1,
        require_empty=False,
        **kwargs,
    ):
        if state is True:
            state = PersistentState.get()

        self.dirname = dirname
        self.filename_prefix = filename_prefix

        self.checkpointer = ModelCheckpoint(
            dirname=dirname,
            filename_prefix=filename_prefix,
            save_interval=save_interval,
            n_saved=n_saved,
            require_empty=require_empty,
            save_as_state_dict=True,
            **kwargs,
        )
        self.to_save = {"model": model, "optimizer": optimizer}

        if state is not None:
            self.to_save["state"] = state

    def on_epoch_completed(self, engine: Engine):
        self.checkpointer(engine, self.to_save)

    def restore(self, map_location=None, optional=True):
        checkpoint = find_latest_checkpoints(
            self.dirname, self.filename_prefix, self.to_save.keys()
        )

        if checkpoint is None:
            if optional:
                _logger.warning("No checkpoint to restore")
                return

            else:
                raise RuntimeError("No checkpoint to restore")

        for k, p in checkpoint.items():
            state = torch.load(p, map_location=map_location)
            self.to_save[k].load_state_dict(state)


def find_latest_checkpoints(dirname, filename_prefix, keys):
    keys = list(keys)
    p = pathlib.Path(dirname).resolve()

    if not p.exists():
        return None

    candidates = {}
    for k in keys:
        for child in p.glob(f"{filename_prefix}_{k}_*.pth"):
            *_, iteration = child.name.rpartition("_")
            iteration, *_ = iteration.partition(".")
            iteration = int(iteration)

            candidates.setdefault(iteration, {})[k] = child

    if not candidates:
        return None

    res = candidates[max(candidates)]

    missing_keys = {*keys} - {*res}
    if missing_keys:
        raise RuntimeError(f"Missing keys: {missing_keys}")

    return res


class ReraiseErrors(SubclassHandler):
    def on_exception_raised(self, engine, exc):
        raise exc
