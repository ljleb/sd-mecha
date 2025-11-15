import torch
from sd_mecha.extensions.merge_methods import StateDict


def convert_vae(huggingface_sd: StateDict[torch.Tensor], ldm_key: str) -> torch.Tensor:
    huggingface_key = ".".join(ldm_key.split(".")[1:])

    needs_reshape = False
    for sd_weight_name, fake_weight_name in vae_extra_conversion_map.items():
        if f"mid.attn_1.{sd_weight_name}.weight" in huggingface_key or f"mid.attn_1.{sd_weight_name}.bias" in huggingface_key:
            needs_reshape = True
            huggingface_key = huggingface_key.replace(sd_weight_name, fake_weight_name)

    for weight_name in vae_extra_conversion_map.values():
        if f"mid.attn_1.{weight_name}.weight" in huggingface_key:
            needs_reshape = True

    if "attentions" in huggingface_key:
        for sd_part, hf_part in vae_conversion_map_attn.items():
            huggingface_key = huggingface_key.replace(sd_part, hf_part)

    for sd_part, hf_part in vae_conversion_map.items():
        huggingface_key = huggingface_key.replace(sd_part, hf_part)

    huggingface_key = f"vae.{huggingface_key}"
    res = huggingface_sd[huggingface_key]
    if needs_reshape:
        res = reshape_weight_for_sd(res)
    return res


def reshape_weight_for_sd(w):
    # convert HF linear weights to SD conv2d weights
    if w.ndim != 1:
        return w.reshape(*w.shape, 1, 1)
    else:
        return w


vae_conversion_map = {
    # (original, huggingface)
    "nin_shortcut": "conv_shortcut",
    "norm_out": "conv_norm_out",
    "mid.attn_1.": "mid_block.attentions.0.",
}
for i in range(4):
    # down_blocks have two resnets
    for j in range(2):
        hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
        sd_down_prefix = f"encoder.down.{i}.block.{j}."
        vae_conversion_map[sd_down_prefix] = hf_down_prefix

    if i < 3:
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
        sd_downsample_prefix = f"down.{i}.downsample."
        vae_conversion_map[sd_downsample_prefix] = hf_downsample_prefix

        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"up.{3-i}.upsample."
        vae_conversion_map[sd_upsample_prefix] = hf_upsample_prefix

    # up_blocks have three resnets
    # also, up blocks in hf are numbered in reverse from sd
    for j in range(3):
        hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
        sd_up_prefix = f"decoder.up.{3-i}.block.{j}."
        vae_conversion_map[sd_up_prefix] = hf_up_prefix

# this part accounts for mid blocks in both the encoder and the decoder
for i in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{i}."
    sd_mid_res_prefix = f"mid.block_{i+1}."
    vae_conversion_map[sd_mid_res_prefix] = hf_mid_res_prefix


vae_conversion_map_attn = {
    # (original, huggingface)
    "norm.": "group_norm.",
    "q.": "query.",
    "k.": "key.",
    "v.": "value.",
    "proj_out.": "proj_attn.",
}


# This is probably not the most ideal solution, but it does work.
vae_extra_conversion_map = {
    # (original, huggingface)
    "q": "to_q",
    "k": "to_k",
    "v": "to_v",
    "proj_out": "to_out.0",
}
