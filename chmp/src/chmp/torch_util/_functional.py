import torch


def identity(x):
    return x


def linear(x, weights):
    """A linear interaction.

    :param x:
        shape ``(batch_size, in_features)``
    :param weights:
        shape ``(n_factors, in_features, out_features)``
    """
    return x @ weights


def factorized_quadratic(x, weights):
    """A factorized quadratic interaction.

    :param x:
        shape ``(batch_size, in_features)``
    :param weights:
        shape ``(n_factors, in_features, out_features)``
    """
    x = x[None, ...]
    res = (x @ weights) ** 2.0 - (x ** 2.0) @ (weights ** 2.0)
    res = res.sum(dim=0)
    return 0.5 * res


def masked_softmax(logits, mask, eps=1e-6, dim=-1):
    """Compute a softmax with certain elements masked out."""
    mask = mask.type(logits.type())
    logits = mask * logits - (1 - mask) / eps

    # ensure stability by normalizing with the maximum
    max_logits, _ = logits.max(dim, True)
    logits = logits - max_logits

    p = mask * torch.exp(logits)
    norm = p.sum(dim, True)
    valid = (norm > eps).type(logits.type())

    p = p / (valid * norm + (1 - valid))

    return p
