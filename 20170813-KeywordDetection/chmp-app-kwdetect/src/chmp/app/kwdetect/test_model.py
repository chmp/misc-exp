import pytest
import torch

from chmp.app.kwdetect.model import KeywordModel


def test_keyword_model_batch_shape():
    inputs = torch.normal(torch.zeros(10, 100, 15), torch.ones(10, 100, 15))
    lengths = torch.tensor([80, 100, 90, 70, 60, 40, 50, 10, 20, 30])

    model = KeywordModel(n_features=15, n_classes=5)
    model_params = {name for name, _ in model.named_parameters()}
    logits = model(inputs, lengths)

    assert logits.shape == (10, 5)

    # backprop through the model and check that all parameters a have a gradient
    assert get_params_with_grad(model) == set()
    logits.sum().backward()
    assert get_params_with_grad(model) == model_params


def test_keyword_model_sorting_works():
    inputs = torch.normal(torch.zeros(5, 100, 15), torch.ones(5, 100, 15))
    lengths = torch.tensor([80, 100, 90, 70, 60])

    # manually constructed sort / unsort indices
    sort_indices = torch.tensor([1, 2, 0, 3, 4])
    unsort_indices = torch.tensor([2, 0, 1, 3, 4])

    # apply model as-is
    model = KeywordModel(n_features=15, n_classes=3)
    unsorted_logits = model(inputs, lengths)

    # perform manual sorting
    sorted_logits = model(inputs[sort_indices], lengths[sort_indices])
    sorted_logits = sorted_logits[unsort_indices]

    assert sorted_logits.detach().numpy() == pytest.approx(unsorted_logits.detach().numpy())


def get_params_with_grad(module):
    return {
        name
        for name, param in module.named_parameters()
        if param.grad is not None
    }
