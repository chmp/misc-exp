import numpy as np
import torch
import pytest

# NOTE: also registers the KL divergence
from . import DiagonalScaleShift


def test_diagonal_scale_shift():
    m = DiagonalScaleShift(shift=torch.ones(10), scale=2. * torch.ones(10))
    assert m(torch.zeros(20, 10)).shape == (20, 10)
