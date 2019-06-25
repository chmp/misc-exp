"""Helper to construct models with pytorch."""
import collections
import enum
import itertools as it
import operator as op

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from chmp.ds import smap, szip, flatten_with_index, unflatten


default_batch_size = 32


class fixed:
    """decorator to mark a parameter as not-optimized."""

    def __init__(self, value):
        self.value = value


class optimized:
    """Decorator to mark a parameter as optimized."""

    def __init__(self, value):
        self.value = value


def optional_parameter(arg, *, default=optimized):
    """Make sure arg is a tensor and optionally a parameter.

    Values wrapped with ``fixed`` are returned as a tensor, ``values`` wrapped
    with ``optimized``are returned as parameters. When arg is not one of
    ``fixed`` or ``optimized`` it is wrapped with ``default``.

    Usage::

        class MyModule(torch.nn.Module):
            def __init__(self, a, b):
                super().__init__()

                # per default a will be optimized during training
                self.a = optional_parameter(a, default=optimized)

                # per default B will not be optimized during training
                self.b = optional_parameter(b, default=fixed)

    """
    if isinstance(arg, fixed):
        return torch.as_tensor(arg.value)

    elif isinstance(arg, optimized):
        return torch.nn.Parameter(torch.as_tensor(arg.value))

    elif default is optimized:
        return torch.nn.Parameter(torch.as_tensor(arg))

    elif default is fixed:
        return torch.as_tensor(arg)

    else:
        raise RuntimeError()


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


def t2n(obj, dtype=None):
    """Torch to numpy."""
    return smap(lambda obj: np.asarray(obj.detach().cpu(), dtype=dtype), obj)


def n2t(obj, dtype=None, device=None):
    """Numpy to torch."""
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    if isinstance(device, str):
        device = torch.device(device)

    return smap(lambda obj: torch.as_tensor(obj, dtype=dtype, device=device), obj)


def call_torch(func, arg, *args, dtype=None, device=None, batch_size=64):
    """Call a torch function with numpy arguments and numpy results."""
    args = (arg, *args)
    index, values = flatten_with_index(args)
    result_batches = []

    for start in it.count(0, batch_size):
        end = start + batch_size

        if start >= len(values[0]):
            break

        batch = unflatten(index, (val[start:end] for val in values))
        batch = n2t(batch, dtype=dtype, device=device)
        result = func(*batch)
        result = t2n(result)

        result_batches.append(result)

    result, schema = szip(result_batches, return_schema=True)
    result = smap(lambda _, r: np.concatenate(r, axis=0), schema, result)
    return result


def optimizer_step(optimizer, func):
    optimizer.zero_grad()
    loss = func()
    loss.backward()
    optimizer.step()

    return smap(float, loss)


def identity(x):
    return x


def linear(x, weights):
    """A linear interaction.

    :param x:
        shape ``(batch_size, in_features)``
    :param weights:
        shape ``(n_factors, in_features, out_features)``
    """
    return x @ weights


def factorized_quadratic(x, weights):
    """A factorized quadratic interaction.

    :param x:
        shape ``(batch_size, in_features)``
    :param weights:
        shape ``(n_factors, in_features, out_features)``
    """
    x = x[None, ...]
    res = (x @ weights) ** 2.0 - (x ** 2.0) @ (weights ** 2.0)
    res = res.sum(dim=0)
    return 0.5 * res


def masked_softmax(logits, mask, eps=1e-6, dim=-1):
    """Compute a softmax with certain elements masked out."""
    mask = mask.type(logits.type())
    logits = mask * logits - (1 - mask) / eps

    # ensure stability by normalizing with the maximum
    max_logits, _ = logits.max(dim, True)
    logits = logits - max_logits

    p = mask * torch.exp(logits)
    norm = p.sum(dim, True)
    valid = (norm > eps).type(logits.type())

    p = p / (valid * norm + (1 - valid))

    return p


def find_module(root, predicate):
    """Find a (sub) module using a predicate.

    :param predicate:
        a callable with arguments ``(name, module)``.
    :returns:
        the first module for which the predicate is true or raises
        a ``RuntimeError``.
    """
    for k, v in root.named_modules():
        if predicate(k, v):
            return v

    else:
        raise RuntimeError("could not find module")


class DiagonalScaleShift(torch.nn.Module):
    """Scale and shift the inputs along each dimension independently."""

    @classmethod
    def from_data(cls, data):
        return cls(shift=data.mean(), scale=1.0 / (1e-5 + data.std()))

    def __init__(self, shift=None, scale=None):
        super().__init__()
        assert (shift is not None) or (scale is not None)

        if shift is not None:
            shift = torch.as_tensor(shift).clone()

        if scale is not None:
            scale = torch.as_tensor(scale).clone()

        if shift is None:
            shift = torch.zeros_like(scale)

        if scale is None:
            scale = torch.ones_like(shift)

        self.shift = torch.nn.Parameter(shift)
        self.scale = torch.nn.Parameter(scale)

    def forward(self, x):
        return self.scale * (x - self.shift)


class BinaryOperatorConstant(torch.nn.Module):
    def __init__(self, op, value):
        super().__init__()
        self.op = op
        self.value = optional_parameter(value)

    def forward(self, x):
        return self.op(x, self.value)


class MultiplyConstant(BinaryOperatorConstant):
    def __init__(self, value):
        super().__init__(op=op.mul, value=value)


class DivideConstant(BinaryOperatorConstant):
    def __init__(self, value):
        super().__init__(op=op.truediv, value=value)


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.reshape(-1)


class Add(torch.nn.ModuleList):
    """Apply all modules in parallel and add their outputs."""

    def __init__(self, *children):
        super().__init__(children)

    def forward(self, x):
        return sum(child(x) for child in self)


def format_extra_repr(*kv_pairs):
    return ", ".join("{}={}".format(k, v) for k, v in kv_pairs)


class CallableWrapper(torch.nn.Module):
    def __init__(self, func, **kwargs):
        super().__init__()
        self.func = func
        self.kwargs = kwargs

    def extra_repr(self):
        return format_extra_repr(("func", self.func), *self.kwargs.items())


class Do(CallableWrapper):
    """Call a function as a pure side-effect."""

    def forward(self, x, **kwargs):
        self.func(x, **kwargs, **self.kwargs)
        return x


class Lambda(CallableWrapper):
    def forward(self, *x, **kwargs):
        return self.func(*x, **kwargs, **self.kwargs)


class CallModule(torch.nn.Module):
    def __init__(self, module, caller):
        super().__init__()
        self.module = module
        self.caller = caller

    def forward(self, *args, **kwargs):
        return self.caller(self.module, *args, **kwargs)

    def extra_repr(self):
        return f"caller={self.caller},"


class LocationScale(torch.nn.Module):
    def __init__(self, activation=None, eps=1e-6):
        super().__init__()

        if activation is None:
            activation = Identity()

        self.eps = eps
        self.activation = activation

    def forward(self, x):
        *_, n = x.shape
        assert (n % 2) == 0, "can only handle even number of features"

        loc = x[..., : (n // 2)]
        scale = x[..., (n // 2) :]

        loc = self.activation(loc)
        scale = self.eps + F.softplus(scale)

        return loc, scale

    def extra_repr(self):
        return f"eps={self.eps},"


# TODO: figure out how to properly place the nodes
# TODO: use linear interpolation
class LookupFunction(torch.nn.Module):
    """Helper to define a lookup function incl. its gradient.

    Usage::

        import scipy.special

        x = np.linspace(0, 10, 100).astype('float32')
        iv0 = scipy.special.iv(0, x).astype('float32')
        iv1 = scipy.special.iv(1, x).astype('float32')

        iv = LookupFunction(x.min(), x.max(), iv0, iv1)

        a = torch.linspace(0, 20, 200, requires_grad=True)
        g, = torch.autograd.grad(iv(a), a, torch.ones_like(a))

    """

    def __init__(self, input_min, input_max, forward_values, backward_values):
        super().__init__()
        self.input_min = torch.as_tensor(input_min)
        self.input_max = torch.as_tensor(input_max)
        self.forward_values = torch.as_tensor(forward_values)
        self.backward_values = torch.as_tensor(backward_values)

    def forward(self, x):
        return _LookupFunction.apply(
            x, self.input_min, self.input_max, self.forward_values, self.backward_values
        )


class _LookupFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, input_min, input_max, forward_values, backward_values):
        idx_max = len(forward_values) - 1
        idx_scale = idx_max / (input_max - input_min)
        idx = (idx_scale * (x - input_min)).type(torch.long)
        idx = torch.clamp(idx, 0, idx_max)

        if backward_values is not None:
            ctx.save_for_backward(backward_values[idx])

        else:
            ctx.save_for_backward(None)

        return forward_values[idx]

    @staticmethod
    def backward(ctx, grad_output):
        backward_values, = ctx.saved_tensors
        return grad_output * backward_values, None, None, None, None


def build_mlp(
    in_features,
    out_features,
    *,
    hidden=(),
    hidden_activation=torch.nn.ReLU,
    activation=Identity,
    container=torch.nn.Sequential,
):
    features = [in_features, *hidden, out_features]
    activations = len(hidden) * [hidden_activation] + [activation]

    parts = []
    for a, b, activation in zip(features[:-1], features[1:], activations):
        parts += [torch.nn.Linear(a, b), activation()]

    return container(parts)


def make_data_loader(dataset, mode="fit", **kwargs):
    if mode == "fit":
        default_kwargs = dict(shuffle=True, drop_last=True)

    elif mode == "predict":
        default_kwargs = dict(shuffle=False, drop_last=False)

    else:
        raise ValueError()

    kwargs = {**default_kwargs, **kwargs}
    return torch.utils.data.DataLoader(dataset, **kwargs)


class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, data, dtype=None):
        index, data = flatten_with_index(data)
        data = [np.asarray(v) for v in data]

        self.index = index
        self.data = data
        self.dtype = dtype
        self.length = self._guess_length()

    def _guess_length(self):
        candidates = set()

        for item in self.data:
            if item is None:
                continue

            candidates.add(len(item))

        if len(candidates) != 1:
            raise ValueError(f"Arrays with different lengths: {candidates}")

        length, = candidates
        return length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        res = [(item[idx] if item is not None else None) for item in self.data]
        if self.dtype is not None:
            res = [
                (np.asarray(item, dtype=self.dtype) if item is not None else None)
                for item in res
            ]

        return unflatten(self.index, res)


class Transformer(torch.nn.Module):
    """A attention / transformer Module.

    Masks be two-dimensional and compatible with ``n_query, n_search``. This
    model also supports soft-masks. They must never be ``0``. The hard masks
    must be binary ``{0, 1}``.
    """

    def __init__(
        self,
        key_module,
        query_module=None,
        value_module=None,
        flatten=False,
        search_x=None,
        search_y=None,
    ):
        super().__init__()

        if query_module is None:
            query_module = key_module

        if value_module is None:
            value_module = noop_value_module

        if search_x is not None:
            search_x = torch.as_tensor(search_x)

        if search_y is not None:
            search_y = torch.as_tensor(search_y)

        self.flatten = flatten
        self.key_module = key_module
        self.query_module = query_module
        self.value_module = value_module
        self.search_x = search_x
        self.search_y = search_y

    def forward(self, query_x, mask=None, soft_mask=None, search_x=None, search_y=None):
        if search_x is None:
            search_x = self.search_x

        if search_y is None:
            search_y = self.search_y

        # shape: batch_size, n_values
        values = self.value_module(search_x, search_y)
        value_ndim = values.ndimension()

        values = self._ensure_value_shape(values)

        p = self.compute_weights(
            search_x=search_x, query_x=query_x, mask=mask, soft_mask=soft_mask
        )

        # sum over samples
        # shape: batch_size, n_keys, n_values,
        res = (p[:, :, :, None] * values[None, :, None, :]).sum(dim=1)

        if self.flatten:
            return res.reshape(query_x.size(0), -1)

        # NOTE: for a 1d tensor, we added a new dimension in
        # _ensure_value_shape, remove this dimension
        elif value_ndim == 1:
            return res.reshape(query_x.size(0))

        else:
            return res

    def compute_weights(self, search_x, query_x, mask, soft_mask=None):
        """Compute weights with shape ``(batch_size, n_samples, n_keys)``.
        """
        # shape: batch_size, n_keys, key_size,
        keys = self.key_module(search_x)
        keys = self._ensure_key_shape(keys)

        # shape: n_samples, n_keys, key_size,
        query_keys = self.query_module(query_x)
        query_keys = self._ensure_key_shape(query_keys)

        # shape: batch_size, n_samples, n_keys,
        logits = (query_keys[:, None, :, :] * keys[None, :, :, :]).sum(-1) / (
            keys.size(-1) ** 0.5
        )

        if soft_mask is not None:
            logits = logits + torch.log(soft_mask[:, :, None])

        # shape: batch_size, n_samples, n_keys,
        if mask is not None:
            p = masked_softmax(logits, mask[:, :, None], dim=1)

        else:
            p = F.softmax(logits, dim=1)

        return p

    @staticmethod
    def _ensure_key_shape(keys):
        if keys.ndimension() == 2:
            return keys[:, None, :]

        else:
            return keys

    @staticmethod
    def _ensure_value_shape(values):
        if values.ndimension() == 1:
            return values[:, None]

        else:
            return values


def noop_value_module(_, y):
    return y


@register_unknown_kl(torch.distributions.LogNormal, torch.distributions.Gamma)
def kl_divergence__gamma__log_normal(p, q):
    """Compute the kl divergence with a Gamma prior and LogNormal approximation.

    Taken from C. Louizos, K. Ullrich, M. Welling "Bayesian Compression for Deep Learning"
    https://arxiv.org/abs/1705.08665
    """
    return (
        q.concentration * torch.log(q.rate)
        + torch.lgamma(q.concentration)
        - q.concentration * p.loc
        + torch.exp(p.loc + 0.5 * p.scale ** 2) / q.rate
        - 0.5 * (torch.log(p.scale ** 2.0) + 1 + np.log(2 * np.pi))
    )
