import torch

from .clip_missing_keys import clip_position_ids


def sdxl_sgm_missing_key_handler(config, key, device, dtype):
    return _sdxl_missing_key_handler(
        config, key, device, dtype, "conditioner.embedders.0.transformer.text_model",
    )


def sdxl_kohya_missing_key_handler(config, key, device, dtype):
    return _sdxl_missing_key_handler(config, key, device, dtype, "te1.text_model")


def _sdxl_missing_key_handler(config, key, device, dtype, clip_l_prefix):
    if key == f"{clip_l_prefix}.embeddings.position_ids":
        return clip_position_ids(config, key, device, dtype)
    if key.startswith(f"{clip_l_prefix}.encoder.layers.11."):
        meta = config.keys().get(key)
        if meta is not None:
            return torch.zeros(meta.shape, device=device, dtype=dtype or meta.dtype)
    return None
