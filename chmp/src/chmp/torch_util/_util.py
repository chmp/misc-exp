import torch


def register_unknown_kl(type_p, type_q):
    def decorator(func):
        if has_kl(type_p, type_q):
            func.registered = False
            return func

        torch.distributions.kl.register_kl(type_p, type_q)(func)
        func.registered = True
        return func

    return decorator


def has_kl(type_p, type_q):
    import torch

    return (type_p, type_q) in torch.distributions.kl._KL_REGISTRY
