import numpy as np
import torch

from chmp.torch_util import TorchModel, Flatten


def build_example_model():
    return TorchModel(
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
        x = np.random.normal(size=(len(indices), 10))
        y = np.random.normal(size=len(indices))
        return x, y

    def pred_data(indices):
        return np.random.normal(size=(len(indices), 10))

    model.fit_transformed(range(32), fit_data, epochs=10)
    y_pred = model.predict_transformed(range(100), pred_data)
    assert len(y_pred) == 100
