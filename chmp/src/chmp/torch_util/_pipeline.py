import torch


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

    def forward(self, *x):
        return self.func(*x)
