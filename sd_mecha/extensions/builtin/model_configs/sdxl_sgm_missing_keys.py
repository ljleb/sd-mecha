import torch


def sdxl_sgm_missing_key_handler(config, key, device, dtype):
    meta = config.keys().get(key)
    if meta is None:
        return None

    if key == "conditioner.embedders.0.transformer.text_model.embeddings.position_ids":
        return torch.arange(meta.shape[1], device=device, dtype=meta.dtype).reshape(meta.shape)

    if (
        key.startswith("conditioner.embedders.0.transformer.text_model.encoder.layers.11.")
        or key.startswith("conditioner.embedders.1.model.transformer.resblocks.31.")
    ):
        return torch.zeros(meta.shape, device=device, dtype=dtype or meta.dtype)

    return None
