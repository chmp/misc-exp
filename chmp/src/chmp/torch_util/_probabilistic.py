import numpy as np
import torch

from ._util import register_unknown_kl
from ._model import TorchModel


class fixed:
    """decorator to mark a parameter as not-optimized"""

    def __init__(self, value):
        self.value = value


class optimized:
    """decorator to mark a parameter as optimized"""

    def __init__(self, value):
        self.value = value


def optional_parameter(arg):
    if isinstance(arg, fixed):
        return as_tensor(arg.value)

    elif isinstance(arg, optimized):
        return torch.nn.Parameter(as_tensor(arg.value))

    else:
        return torch.nn.Parameter(as_tensor(arg))


def as_tensor(arg):
    return torch.tensor(arg) if not torch.is_tensor(arg) else arg


class TorchDistributionModule(torch.nn.Module):
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

    def __call__(self):
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


class SimpleBayesTorchModel(TorchModel):
    def __init__(self, module, n_observations, **kwargs):
        kwargs.setdefault("loss", NllLoss(module._distribution))
        kwargs.setdefault("regularization", KLDivergence(n_observations=n_observations))
        super().__init__(module=module, **kwargs)


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
    """Negative log likelihood loss for pytorch distributions."""

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


class WeightsHS(torch.nn.Module):
    """A module that generates weights with a Horeshoe Prior."""

    def __init__(self, shape, tau_0):
        super().__init__()

        self.shape = shape

        # See Bayesian Compression for Deep Learning for variable meaning
        self.p_inv_sb = torch.distributions.Gamma(0.5, 1.0)
        self.p_sa = torch.distributions.Gamma(0.5, float(tau_0) ** 2.0)
        self.p_inv_beta = torch.distributions.Gamma(0.5, 1.0)
        self.p_alpha = torch.distributions.Gamma(0.5, 1.0)
        self.p_w = torch.distributions.Normal(0.0, 1.0)

        self.q_inv_sb = LogNormalModule(torch.tensor(0.5), torch.tensor(1.0))
        self.q_sa = LogNormalModule(torch.tensor(0.5), torch.tensor(1.0))
        self.q_inv_beta = LogNormalModule(
            0.5 * torch.ones(*shape), 1.0 * torch.ones(*shape)
        )
        self.q_alpha = LogNormalModule(
            0.5 * torch.ones(*shape), 1.0 * torch.ones(*shape)
        )

        # USE a modified Glorot initialization
        stddev = np.sqrt(1.0 / np.mean(shape))
        self.q_w = NormalModule(
            torch.normal(torch.zeros(*shape), stddev * torch.ones(*shape)),
            stddev * torch.ones(*shape),
        )

    def forward(self):
        scale = torch.sqrt(
            (self.q_sa.rsample() / self.q_inv_sb.rsample())
            * (self.q_alpha.rsample() / self.q_inv_beta.rsample())
        )
        return self.q_w.rsample() * scale

    def kl_divergence(self):
        return sum(
            torch.sum(torch.distributions.kl_divergence(q(), p))
            for q, p in [
                (self.q_inv_sb, self.p_inv_sb),
                (self.q_sa, self.p_sa),
                (self.q_inv_beta, self.p_inv_beta),
                (self.q_alpha, self.p_alpha),
                (self.q_w, self.p_w),
            ]
        )


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
