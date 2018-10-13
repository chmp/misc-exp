"""Helper to construct models with pytorch."""
from ._attention import Transformer
from ._functional import factorized_quadratic, identity, linear, masked_softmax
from ._model import (
    LearningRateScheduler,
    LossHistory,
    TerminateOnNaN,
    TorchModel,
    iter_batch_indices,
    iter_batched,
)
from ._pipeline import Add, DiagonalScaleShift, Flatten, Lambda, find_module
from ._probabilistic import (
    ExponentialModule,
    LogNormalModule,
    GammaModule,
    KLDivergence,
    NllLoss,
    NormalModelConstantScale,
    NormalModule,
    SimpleBayesTorchModel,
    WeightsHS,
    fixed,
)


__all__ = [
    "Transformer",
    "factorized_quadratic",
    "identity",
    "linear",
    "masked_softmax",
    "LearningRateScheduler",
    "LossHistory",
    "TerminateOnNaN",
    "TorchModel",
    "iter_batch_indices",
    "iter_batched",
    "Add",
    "DiagonalScaleShift",
    "Flatten",
    "Lambda",
    "find_module",
    "ExponentialModule",
    "LogNormalModule",
    "GammaModule",
    "KLDivergence",
    "NllLoss",
    "NormalModelConstantScale",
    "NormalModule",
    "SimpleBayesTorchModel",
    "WeightsHS",
    "fixed",
]
