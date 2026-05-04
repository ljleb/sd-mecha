import torch


def reconstruct_test_key(config, key, device, dtype):
    if key != "key":
        return None

    meta = config.keys()[key]
    if dtype is None:
        dtype = meta.dtype
    return torch.full(meta.shape, 42, device=device, dtype=dtype)
