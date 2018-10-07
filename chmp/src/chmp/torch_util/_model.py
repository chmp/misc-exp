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
        the loss function to use, with signature ``(pred, target) -> loss``
    :param regularization:
        if given a callable, with signature ``(module) -> loss``, that should
        return a regularization loss
    """

    def __init__(self, module, optimizer, loss, regularization=None):
        self.module = module
        self.optimizer = optimizer
        self.loss = loss
        self.regularization = regularization

    def fit(
        self, x=None, y=None, *, batch_size=None, epochs=1, shuffle=True, verbose=False
    ):
        keys, values = pack(x, y)
        n_samples = get_number_of_samples(*values)

        optimizer = self.optimizer(self.module.parameters())

        for epoch in optional_loop(range(epochs), verbose=verbose):
            indices = None if not shuffle else shuffled_indices(n_samples)

            for batch_values in optional_loop_nest(
                iter_batched(values, batch_size=batch_size, indices=indices),
                verbose=verbose,
                label=f"fit (epoch {epoch:,d} / {epochs:,d})",
            ):
                batch_values = list_as_tensor(batch_values)
                batch_x, batch_y = unpack(keys, batch_values)

                self._batch_step(optimizer, batch_x, batch_y)

        if verbose:
            print("\n")

        return self

    def fit_generator(self, generator, *, steps_per_epoch=1, epochs=None, verbose=True):
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
        generator = iter(generator)
        optimizer = self.optimizer(self.module.parameters())
        epoch_sequence = it.count() if epochs is None else range(epochs)

        for epoch in optional_loop(epoch_sequence, verbose=verbose):
            for _ in optional_loop_nest(
                range(steps_per_epoch),
                verbose=verbose,
                label=(
                    f"fit (epoch {epoch:,d} / {epochs:,d})"
                    if epochs is not None
                    else f"fit (epoch {epoch:,d} / ?)"
                ),
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

                batch_x, batch_y = generic_as_tensor(batch_x, batch_y)

                self._batch_step(optimizer, batch_x, batch_y)

            # if the epoch was not aborted
            else:
                continue

            # else break
            break

        if verbose:
            print("\n")

    def _batch_step(self, optimizer, batch_x, batch_y):
        optimizer.zero_grad()
        batch_pred = self.module(batch_x)
        loss = self.loss(batch_pred, batch_y)
        loss = self._add_regularization(loss)

        loss.backward()
        optimizer.step()

    def _add_regularization(self, loss):
        if not self.regularization:
            return loss

        return loss + self.regularization(self.module)

    def predict(self, x=None, batch_size=None, verbose=False):
        keys, values = pack(x)
        batch_keys_values_pairs = []

        for batch_values in optional_loop(
            iter_batched(values, batch_size=batch_size, only_complete=False),
            verbose=verbose,
            label="predict",
        ):
            batch_values = list_as_tensor(batch_values)
            batch_x, = unpack(keys, batch_values)

            pred = self.module(batch_x)

            pred_keys, pred_values = pack(pred)
            pred_values = list_torch_to_numpy(pred_values)

            batch_keys_values_pairs.append((pred_keys, pred_values))

        if verbose:
            print("\n")

        result, = unpack_batches(batch_keys_values_pairs)
        return result

    def predict_generator(self, generator, *, steps=None, verbose=True):
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

            batch_x, = generic_as_tensor(batch_x)

            pred = self.module(batch_x)

            pred_keys, pred_values = pack(pred)
            pred_values = list_torch_to_numpy(pred_values)

            batch_keys_values_pairs.append((pred_keys, pred_values))

        if verbose:
            print("\n")

        result, = unpack_batches(batch_keys_values_pairs)
        return result


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
        if isinstance(arg, (list, tuple)):
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
