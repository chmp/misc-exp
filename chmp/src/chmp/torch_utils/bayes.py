import numpy as np
import torch

from ._util import register_unknown_kl, fixed, optimized, optional_parameter

__all__ = [
    "KLDivergence",
    "SimpleBayesModel",
    "VariationalNormal",
    "VariationalHalfCauchy",
    "NormalModule",
    "GammaModule",
    "LogNormalModule",
    "ExponentialModule",
    "NormalModelConstantScale",
    "WeightsHS",
    "optimized",
    "fixed",
    "optional_parameter",
]


class TorchDistributionModule(torch.nn.Module):
    """Base class to turn a torch distribution into an optimizable module."""

    _forward_attributes_ = {"rsample", "sample", "sample_n", "log_prob", "cdf", "icdf"}

    def __init__(self, *args, **kwargs):
        super().__init__()

        duplicate_parameters = {*self._distribution_parameters_[: len(args)]} & {
            *kwargs
        }
        if duplicate_parameters:
            raise ValueError("duplicate parameters: {}".format(duplicate_parameters))

        kwargs.update(zip(self._distribution_parameters_, args))

        for k, v in kwargs.items():
            setattr(self, k, optional_parameter(v))

    def forward(self):
        """Construct the distribution object"""
        # NOTE: add a zero to ensure the type(p) == 'Tensor' working around a bug in
        #       torch.distributions. With torch==1.0 this bug is fixed.
        return self._distribution_class_(
            *(getattr(self, k).add(0) for k in self._distribution_parameters_)
        )

    def __getattr__(self, key):
        if key in self._forward_attributes_:
            return getattr(self(), key)

        return super().__getattr__(key)


class GammaModule(TorchDistributionModule):
    _distribution_class_ = torch.distributions.Gamma
    _distribution_parameters_ = "concentration", "rate"


class NormalModule(TorchDistributionModule):
    _distribution_class_ = torch.distributions.Normal
    _distribution_parameters_ = "loc", "scale"


class LogNormalModule(TorchDistributionModule):
    _distribution_class_ = torch.distributions.LogNormal
    _distribution_parameters_ = "loc", "scale"


class ExponentialModule(TorchDistributionModule):
    _distribution_class_ = torch.distributions.Exponential
    _distribution_parameters_ = ("rate",)


class NormalModelConstantScale(torch.nn.Module):
    _distribution = torch.distributions.Normal

    def __init__(self, transform=None, scale=1.0):
        super().__init__()
        if transform is not None:
            self.transform = transform

        self.scale = optional_parameter(scale)

    def forward(self, batch_x):
        loc = self.transform(batch_x)
        loc = loc.reshape(-1)

        return loc, torch.ones_like(loc) * self.scale

    def kl_divergence(self):
        if hasattr(self.transform, "kl_divergence"):
            return self.transform.kl_divergence()

        else:
            return 0


class NllLoss:
    """Negative log likelihood loss for pytorch distributions.

    Usage::

        loss = NllLoss(torch.distribtuions.Normal)
        loc, scale = parameter_module(x)
        loss((loc, scale), y)

    """

    def __init__(self, distribution):
        self.distribution = distribution

    def __call__(self, pred, y):
        dist = self._build_dist(pred)
        return -torch.mean(dist.log_prob(y))

    def _build_dist(self, pred):
        if isinstance(pred, tuple):
            return self.distribution(*pred)

        elif isinstance(pred, dict):
            return self.distribution(**pred)

        else:
            return self.distribution(pred)


class KLDivergence:
    """A regularizer using the KL divergence of the model."""

    def __init__(self, n_observations):
        self.n_observations = n_observations

    def __call__(self, model):
        return model.kl_divergence() / self.n_observations


class VariationalHalfCauchy(torch.nn.Module):
    """Variational approximation to Half-Cauchy distributed sample."""

    def __init__(self, shape, tau):
        super().__init__()

        self.shape = shape

        # See Bayesian Compression for Deep Learning for variable meaning
        self.p_inv_beta = torch.distributions.Gamma(0.5, 1.0)
        self.p_alpha = torch.distributions.Gamma(0.5, float(tau) ** 2.0)

        self.q_inv_beta = LogNormalModule(
            0.5 * torch.ones(*shape), 1.0 * torch.ones(*shape)
        )
        self.q_alpha = LogNormalModule(
            0.5 * torch.ones(*shape), 1.0 * torch.ones(*shape)
        )

    def forward(self):
        return torch.sqrt(self.q_alpha.rsample() / self.q_inv_beta.rsample())

    def kl_divergence(self):
        return sum(
            torch.sum(torch.distributions.kl_divergence(q(), p))
            for q, p in [
                (self.q_inv_beta, self.p_inv_beta),
                (self.q_alpha, self.p_alpha),
            ]
        )


class VariationalNormal(torch.nn.Module):
    """Variational approximation to a Normal distributed sample."""

    def __init__(self, shape, loc, scale):
        super().__init__()

        self.p = torch.distributions.Normal(loc, scale)

        # USE a modified Glorot initialization
        stddev = np.sqrt(1.0 / np.mean(shape))
        self.q = NormalModule(
            torch.normal(torch.zeros(*shape), stddev * torch.ones(*shape)),
            stddev * torch.ones(*shape),
        )

    def forward(self):
        return self.q.rsample()

    def kl_divergence(self):
        return torch.sum(torch.distributions.kl_divergence(self.q(), self.p))


class WeightsHS(torch.nn.Module):
    """A module that generates weights with a Horeshoe Prior.

    :param shape:
        the shape of sample to generate
    :param tau_0:
        the scale of the the global scale prior. Per default, this parameter
        is not optimized. Pass as ``optimized(inital_tau_0)`` to fit the
        parameter with maximum likelihood.
    :param regularization:
        if given, the regularization strength.

    To implement a linear regression model with Horseshoe prior, use::

        class LinearHS(NormalModelConstantScale):
            def __init__(self, in_features, out_features, tau_0, bias=True):
                super().__init__()

                self.weights = WeightsHS((in_features, out_features), tau_0=tau_0)
                self.bias = torch.nn.Parameter(torch.zeros(1)) if bias else 0

            def transform(self, x):
                return self.bias + linear(x, self.weights())

            def kl_divergence(self):
                return self.weights.kl_divergence()

    Sources:

    * The basic implementation (incl. the posterior approximation) is taken
        from C. Louizos, K. Ullrich, and M. Welling " Bayesian Compression for
        Deep Learning" (2017).
    * The regularization concept is taken from J. Piironen and A. Vehtari
        "Sparsity information and regularization in the horseshoe and other
        shrinkage priors" (2107).

    """

    def __init__(self, shape, tau_0, regularization=None):
        super().__init__()

        self.shape = shape
        self.regularization = regularization
        self.tau_0 = optional_parameter(tau_0, default=fixed)

        # See Bayesian Compression for Deep Learning for variable meaning
        self.global_scale = VariationalHalfCauchy([1], tau_0)
        self.local_scale = VariationalHalfCauchy(shape, 1.0)
        self.unit_weights = VariationalNormal(shape, 0.0, 1.0)

    def forward(self):
        scale = self.global_scale() * self.local_scale()

        if self.regularization is not None:
            scale = torch.sqrt(
                (self.regularization ** 2.0 * scale ** 2.0)
                / (self.regularization ** 2.0 + scale * 2.0)
            )

        return scale * self.unit_weights()

    def kl_divergence(self):
        return (
            self.global_scale.kl_divergence()
            + self.local_scale.kl_divergence()
            + self.unit_weights.kl_divergence()
        )
