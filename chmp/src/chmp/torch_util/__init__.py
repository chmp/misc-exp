"""Helper to construct models with pytorch."""
from ._attention import Transformer
from ._batched import (
    Callback,
    History,
    LossHistory,
    TerminateOnNaN,
    iter_batch_indices,
    iter_batched,
)
from ._functional import factorized_quadratic, identity, linear, masked_softmax
from ._model import LearningRateScheduler, TorchModel
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
    optimized,
    optional_parameter,
)


__all__ = [
    "TorchModel",
    "Callback",
    "LearningRateScheduler",
    "History",
    "LossHistory",
    "TerminateOnNaN",
    "Transformer",
    "iter_batch_indices",
    "iter_batched",
    "factorized_quadratic",
    "identity",
    "linear",
    "masked_softmax",
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
    "optional_parameter",
    "fixed",
    "optimized",
]
