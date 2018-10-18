import torch
import torch.nn.functional as F

from .nn import masked_softmax

__all__ = ["Transformer"]


class Transformer(torch.nn.Module):
    """A attention / transformer model.

    Masks be two-dimensional and compatible with ``n_query, n_search``. This
    model also supports soft-masks. They must never be ``0``. The hard masks
    must be binary ``{0, 1}``.
    """

    def __init__(
        self,
        key_module,
        query_module=None,
        value_module=None,
        flatten=False,
        search_x=None,
        search_y=None,
    ):
        super().__init__()

        if query_module is None:
            query_module = key_module

        if value_module is None:
            value_module = noop_value_module

        if search_x is not None:
            search_x = torch.as_tensor(search_x)

        if search_y is not None:
            search_y = torch.as_tensor(search_y)

        self.flatten = flatten
        self.key_module = key_module
        self.query_module = query_module
        self.value_module = value_module
        self.search_x = search_x
        self.search_y = search_y

    def forward(self, query_x, mask=None, soft_mask=None, search_x=None, search_y=None):
        if search_x is None:
            search_x = self.search_x

        if search_y is None:
            search_y = self.search_y

        # shape: batch_size, n_values
        values = self.value_module(search_x, search_y)
        value_ndim = values.ndimension()

        values = self._ensure_value_shape(values)

        p = self.compute_weights(
            search_x=search_x, query_x=query_x, mask=mask, soft_mask=soft_mask
        )

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

    def compute_weights(self, search_x, query_x, mask, soft_mask=None):
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

        if soft_mask is not None:
            logits = logits + torch.log(soft_mask[:, :, None])

        # shape: batch_size, n_samples, n_keys,
        if mask is not None:
            p = masked_softmax(logits, mask[:, :, None], dim=1)

        else:
            p = F.softmax(logits, dim=1)

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


def noop_value_module(_, y):
    return y
