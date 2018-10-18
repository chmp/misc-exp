import numpy as np
import pytest
import torch

from chmp.torch_utils.model import Model, parallel_concat
from .nn import Flatten


def build_example_model():
    return Model(
        module=torch.nn.Sequential(
            torch.nn.Linear(in_features=10, out_features=1), Flatten()
        ),
        loss=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam,
    )


def test_torch_model__example_linear_regression():
    model = build_example_model()
    model.fit(
        np.random.normal(size=(100, 10)),
        np.random.normal(size=100),
        metrics=[torch.nn.MSELoss()],
        validation_data=(np.random.normal(size=(5, 10)), np.random.normal(size=5)),
    )
    y_pred = model.predict(np.random.normal(size=(100, 10)))
    assert len(y_pred) == 100


def test_torch_model__example_linear_regression__generators():
    model = build_example_model()

    def fit_data(indices):
        x = np.random.normal(size=10)
        y = np.random.normal()
        return x, y

    def pred_data(indices):
        return np.random.normal(size=10)

    model.fit_transformed(fit_data, range(32), epochs=10)
    y_pred = model.predict_transformed(pred_data, range(100))
    assert len(y_pred) == 100


def test_parallel_concat():
    assert parallel_concat([[[10, 20], [5]], [[30, 40], [6]]]) == (
        pytest.approx(np.asarray([10, 20, 30, 40])),
        pytest.approx(np.asarray([5, 6])),
    )


def test_parallel_concat_no_items():
    with pytest.raises(ValueError):
        parallel_concat([])


def test_parallel_concat_different_subitems():
    with pytest.raises(ValueError):
        # NOTE: the first entry has 2 items, the second only one
        parallel_concat([[[10, 20], [5]], [[30, 40]]])
