import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU, MSELoss

from . import Transformer, TorchModel


def test_transformer_fit():
    n_features = 5

    def transformer_fit_data():
        for _ in range(5):
            for _ in range(10):
                batch_x = {
                    "search_x": np.random.normal(size=(50, n_features)).astype(
                        "float32"
                    ),
                    "search_y": np.random.normal(size=50).astype("float32"),
                    "query_x": np.random.normal(size=(10, n_features)).astype(
                        "float32"
                    ),
                }
                batch_y = np.random.normal(size=10).astype("float32")

                yield batch_x, batch_y

    module = Transformer(key_module=Sequential(Linear(5, 20), ReLU()))
    model = TorchModel(
        module=module,
        loss=MSELoss(),
        optimizer=lambda parameters: torch.optim.Adam(parameters, lr=1e-2),
    )

    model.fit_generator(transformer_fit_data(), steps_per_epoch=10, epochs=5)
