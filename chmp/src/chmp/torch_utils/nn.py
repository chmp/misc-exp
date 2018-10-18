import torch

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
]


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


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.reshape(-1)


class Add(torch.nn.ModuleList):
    """Appply all modules in parallel and add their outputs."""

    def __init__(self, *children):
        super().__init__(children)

    def forward(self, x):
        return sum(child(x) for child in self)


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *x, **kwargs):
        return self.func(*x, **kwargs)
