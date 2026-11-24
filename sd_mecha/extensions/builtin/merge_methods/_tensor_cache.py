import torch


DEFAULT_CACHE_DEVICE = torch.device("cpu")
DEFAULT_CACHE_DTYPE = torch.float16


def create_tensor_cache(*, device: str | torch.device = "cpu", dtype: torch.dtype = DEFAULT_CACHE_DTYPE):
    return {"device": device, "dtype": dtype}


def tensor_cache_for_key(cache, key):
    if cache is None:
        return None, DEFAULT_CACHE_DEVICE, DEFAULT_CACHE_DTYPE

    device = cache.get("device", DEFAULT_CACHE_DEVICE)
    dtype = cache.get("dtype", DEFAULT_CACHE_DTYPE)
    # Plain caches from older callers can already contain entries with these key names.
    if isinstance(device, dict):
        device = DEFAULT_CACHE_DEVICE
    if isinstance(dtype, dict):
        dtype = DEFAULT_CACHE_DTYPE
    if key in ("device", "dtype") and key in cache and not isinstance(cache[key], dict):
        key = (key,)

    return cache.setdefault(key, {}), torch.device(device), dtype


def matches_cache_storage(tensor, device, dtype):
    actual = tensor.device
    return tensor.dtype == dtype and actual.type == device.type and (
        actual.index == device.index or actual.index is None or device.index is None
    )
