import torch


def clip_position_ids(config, key, device, dtype):
    if key.endswith(".text_model.embeddings.position_ids"):
        meta = config.keys().get(key)
        if meta is not None:
            return torch.arange(meta.shape[1], device=device, dtype=meta.dtype).reshape(meta.shape)
    return None
