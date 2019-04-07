import torch
import pytest

# NOTE: also registers the KL divergence
from chmp.torch_utils import NormalModule, WeightsHS, fixed


def test_kl_divergence__gamma__log_normal():
    p = torch.distributions.LogNormal(torch.zeros(2), torch.ones(2))
    q = torch.distributions.Gamma(torch.ones(2), torch.ones(2))

    torch.distributions.kl_divergence(p, q)


def test__module_parameters():
    module = NormalModule(loc=torch.zeros(1), scale=fixed(torch.ones(1)))

    assert {k for k, _ in module.named_parameters()} == {"loc"}

    module = NormalModule(loc=torch.zeros(1), scale=torch.ones(1))

    assert {k for k, _ in module.named_parameters()} == {"loc", "scale"}

    module = NormalModule(torch.zeros(1), scale=fixed(torch.ones(1)))
    assert {k for k, _ in module.named_parameters()} == {"loc"}


def test__module_fixed_parameters_optimize():
    module = NormalModule(torch.zeros(1), fixed(torch.ones(1)))
    optimizer = torch.optim.Adam(module.parameters(), lr=0.1)

    for _ in range(100):
        optimizer.zero_grad()

        x = module.rsample((20,))
        loss = torch.mean((x - 2.0) ** 2.0)
        loss.backward()

        optimizer.step()

    assert float(module.loc) != pytest.approx(0.0)
    assert float(module.scale) == pytest.approx(1.0)


def test_weight_hs_api():
    w = WeightsHS([10, 20, 30], tau_0=1e-5)
    assert w().shape == (10, 20, 30)
    assert w.kl_divergence().shape == ()
