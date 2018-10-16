import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU, MSELoss

from . import Transformer, TorchModel


def test_transformer_fit():
    n_features = 5

    def transform_x(indices):
        return dict(
            search_x=np.random.normal(size=(50, n_features)),
            search_y=np.random.normal(size=50),
            query_x=np.random.normal(size=(len(indices), n_features)),
        )

    def transform_xy(indices):
        batch_x = transform_x(indices)
        batch_y = np.random.normal(size=len(indices))

        return batch_x, batch_y

    module = Transformer(key_module=Sequential(Linear(5, 20), ReLU()))
    model = TorchModel(
        module=module,
        loss=MSELoss(),
        optimizer=lambda parameters: torch.optim.Adam(parameters, lr=1e-2),
    )

    model.fit_transformed(range(100), transform_xy, epochs=5)
    pred = model.predict_transformed(range(100), transform_x)

    assert len(pred) == 100
