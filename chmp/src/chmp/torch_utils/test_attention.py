import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU, MSELoss

from .attention import Transformer
from .model import Model


def test_transformer_fit():
    n_features = 5

    def transform_x(idx):
        return dict(query_x=np.random.normal(size=n_features))

    def transform_xy(idx):
        batch_x = transform_x(idx)
        batch_y = np.random.normal()

        return batch_x, batch_y

    module = Transformer(
        key_module=Sequential(Linear(5, 20), ReLU()),
        search_x=np.random.normal(size=(50, n_features)).astype("float32"),
        search_y=np.random.normal(size=50).astype("float32"),
    )
    model = Model(
        module=module,
        loss=MSELoss(),
        optimizer=lambda parameters: torch.optim.Adam(parameters, lr=1e-2),
    )

    model.fit_transformed(transform_xy, range(100), epochs=5)
    pred = model.predict_transformed(transform_x, range(100))

    assert len(pred) == 100
