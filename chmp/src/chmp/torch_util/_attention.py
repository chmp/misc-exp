import torch

from ._functional import masked_softmax


def noop_value_module(_, y):
    return y


class Transformer(torch.nn.Module):
    def __init__(
        self,
        key_module=None,
        query_module=None,
        value_module=noop_value_module,
        flatten=False,
    ):
        super().__init__()

        if query_module is None:
            query_module = key_module

        self.flatten = flatten
        self.key_module = key_module
        self.query_module = query_module
        self.value_module = value_module

    def forward(self, x):
        return self.predict(
            search_x=x["search_x"],
            search_y=x["search_y"],
            query_x=x["query_x"],
            mask=x.get("mask"),
        )

    def predict(self, search_x, search_y, query_x, mask=None):
        if mask is None:
            mask = torch.ones(query_x.shape[0], search_x.shape[0], dtype=torch.int64)

        # shape: batch_size, n_values
        values = self.value_module(search_x, search_y)
        value_ndim = values.ndimension()

        values = self._ensure_value_shape(values)

        p = self.compute_weights(search_x=search_x, query_x=query_x, mask=mask)

        # sum over samples
        # shape: batch_size, n_keys, n_values,
        res = (p[:, :, :, None] * values[None, :, None, :]).sum(dim=1)

        if self.flatten:
            return res.reshape(query_x.size(0), -1)

        # NOTE: for a 1d tensor, we added a new dimension in
        # _ensure_value_shape, remove this dimension
        elif value_ndim == 1:
            return res.reshape(query_x.size(0))

        else:
            return res

    def compute_weights(self, search_x, query_x, mask):
        """Compute weights with shape ``(batch_size, n_samples, n_keys)``.
        """
        # shape: batch_size, n_keys, key_size,
        keys = self.key_module(search_x)
        keys = self._ensure_key_shape(keys)

        # shape: n_samples, n_keys, key_size,
        query_keys = self.query_module(query_x)
        query_keys = self._ensure_key_shape(query_keys)

        # shape: batch_size, n_samples, n_keys,
        logits = (query_keys[:, None, :, :] * keys[None, :, :, :]).sum(-1) / (
            keys.size(-1) ** 0.5
        )

        # shape: batch_size, n_samples, n_keys,
        p = masked_softmax(logits, mask[:, :, None], dim=1)

        return p

    @staticmethod
    def _ensure_key_shape(keys):
        if keys.ndimension() == 2:
            return keys[:, None, :]

        else:
            return keys

    @staticmethod
    def _ensure_value_shape(values):
        if values.ndimension() == 1:
            return values[:, None]

        else:
            return values
