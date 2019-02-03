import operator as op

import numpy as np
import torch

from chmp.ds import sapply

from ._util import fixed, optimized, optional_parameter

__all__ = [
    "identity",
    "linear",
    "factorized_quadratic",
    "masked_softmax",
    "find_module",
    "DiagonalScaleShift",
    "Flatten",
    "Add",
    "Lambda",
    "LookupFunction",
    "fixed",
    "optimized",
    "optional_parameter",
    "t2n",
    "n2t",
]


def t2n(obj, dtype=None):
    """Torch to numpy."""
    return sapply(
        lambda obj, *args, **kwargs: np.asarray(obj.detach().cpu(), *args, **kwargs),
        obj,
        dtype=dtype,
    )


def n2t(obj, dtype=None, device=None):
    """Numpy to torch."""
    return sapply(torch.as_tensor, obj, dtype=dtype, device=device)


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
            shift = torch.tensor(shift)

        if scale is not None:
            scale = torch.tensor(scale)

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


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.reshape(-1)


class Add(torch.nn.ModuleList):
    """Apply all modules in parallel and add their outputs."""

    def __init__(self, *children):
        super().__init__(children)

    def forward(self, x):
        return sum(child(x) for child in self)


class Lambda(torch.nn.Module):
    def __init__(self, func, **kwargs):
        super().__init__()
        self.func = func
        self.kwargs = kwargs

    def forward(self, *x):
        return self.func(*x, **self.kwargs)


# TODO: figure out how to properly place the nodes
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
