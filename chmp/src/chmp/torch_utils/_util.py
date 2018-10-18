import collections
import enum

import numpy as np
import torch

default_batch_size = 32


def register_unknown_kl(type_p, type_q):
    def decorator(func):
        if has_kl(type_p, type_q):
            func.registered = False
            return func

        torch.distributions.kl.register_kl(type_p, type_q)(func)
        func.registered = True
        return func

    return decorator


def has_kl(type_p, type_q):
    import torch

    return (type_p, type_q) in torch.distributions.kl._KL_REGISTRY


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
