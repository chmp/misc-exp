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
    model.fit(np.random.normal(size=(100, 10)), np.random.normal(size=100))
    y_pred = model.predict(np.random.normal(size=(100, 10)))
    assert len(y_pred) == 100


def test_torch_model__example_linear_regression__generators():
    model = build_example_model()

    def fit_data():
        while True:
            for _ in range(32):
                x = np.random.normal(size=(32, 10))
                y = np.random.normal(size=32)
                yield x, y

    def pred_data():
        while True:
            yield np.random.normal(size=(10, 10))

    model.fit_generator(fit_data(), steps_per_epoch=3, epochs=10)
    y_pred = model.predict_generator(pred_data(), steps=10)
    assert len(y_pred) == 100


def test_torch_model__example_linear_regression__finite_generators():
    model = build_example_model()

    def fit_data():
        for _ in range(10):
            for _ in range(32):
                x = np.random.normal(size=(32, 10))
                y = np.random.normal(size=32)
                yield x, y

    def pred_data():
        for _ in range(10):
            yield np.random.normal(size=(10, 10))

    model.fit_generator(fit_data(), steps_per_epoch=32)
    y_pred = model.predict_generator(pred_data())
    assert len(y_pred) == 100
