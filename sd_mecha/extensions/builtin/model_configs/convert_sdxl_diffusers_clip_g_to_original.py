from typing import Iterable, Tuple
import torch


def convert_clip_g_key(sgm_key: str) -> Tuple[Iterable[str], bool]:
    transpose = False
    if sgm_key.endswith("text_projection"):
        kohya_key = "te2.text_projection.weight"
    else:
        kohya_key = sgm_key.replace("conditioner.embedders.1.model.", "te2.text_model.")
        kohya_key = kohya_key.replace(".token_embedding.", ".embeddings.token_embedding.")
        kohya_key = kohya_key.replace(".positional_embedding", ".embeddings.position_embedding.weight")
        kohya_key = kohya_key.replace(".transformer.resblocks.", ".encoder.layers.")
        kohya_key = kohya_key.replace(".attn.", ".self_attn.")
        kohya_key = kohya_key.replace(".mlp.c_fc.", ".mlp.fc1.")
        kohya_key = kohya_key.replace(".mlp.c_proj.", ".mlp.fc2.")
        kohya_key = kohya_key.replace(".ln_final.", ".final_layer_norm.")
        kohya_key = kohya_key.replace(".ln_", ".layer_norm")

    if kohya_key.endswith((".in_proj_weight", ".in_proj_bias")):
        is_bias = kohya_key.endswith("bias")
        partial_key = kohya_key.replace(".in_proj_weight", "").replace(".in_proj_bias", "")
        res = tuple((
            f"{partial_key}.{k}_proj.{'bias' if is_bias else 'weight'}"
            for k in ("q", "k", "v")
        ))
    else:
        res = (kohya_key,)

    if sgm_key.endswith("text_projection"):
        transpose = True

    return res, transpose
