import torch


def sdxl_sgm_missing_key_handler(config, key, device, dtype):
    if key == "conditioner.embedders.0.transformer.text_model.embeddings.position_ids":
        meta = config.keys().get(key)
        if meta is not None:
            return torch.arange(meta.shape[1], device=device, dtype=meta.dtype).reshape(meta.shape)

    return _zero_missing_last_layer(config, key, device, dtype, (
        "conditioner.embedders.0.transformer.text_model.encoder.layers.11.",
        "conditioner.embedders.1.model.transformer.resblocks.31.",
    ))


def sdxl_kohya_missing_key_handler(config, key, device, dtype):
    return _zero_missing_last_layer(config, key, device, dtype, (
        "te1.text_model.encoder.layers.11.",
        "te2.text_model.encoder.layers.31.",
    ))


def _zero_missing_last_layer(config, key, device, dtype, prefixes):
    if key.startswith(prefixes):
        meta = config.keys().get(key)
        if meta is not None:
            return torch.zeros(meta.shape, device=device, dtype=dtype or meta.dtype)
    return None
