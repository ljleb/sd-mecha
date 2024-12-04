from torch import Tensor
from sd_mecha.extensions.model_config import ModelConfig
from sd_mecha.extensions.merge_method import convert_to_recipe, StateDict, config_conversion
from sd_mecha.merge_methods import SameMergeSpace


@config_conversion
@convert_to_recipe(identifier="convert_'sd1-diffusers'_to_'sd1-ldm'")
def convert_sd1_diffusers_to_original(
    diffusers_sd: StateDict | ModelConfig["sd1-diffusers"] | SameMergeSpace,
    **kwargs,
) -> Tensor | ModelConfig["sd1-ldm"] | SameMergeSpace:
    ldm_key = kwargs["key"]
    if ldm_key.startswith("model.diffusion_model."):
        return convert_unet(diffusers_sd, ldm_key)
    elif ldm_key.startswith("cond_stage_model."):
        return convert_clip_l(diffusers_sd, ldm_key)
    elif ldm_key.startswith("first_stage_model."):
        return convert_vae(diffusers_sd, ldm_key)
    else:
        return diffusers_sd[ldm_key]


def convert_unet(diffusers_sd: StateDict, ldm_key: str) -> Tensor:
    diffusers_key = '.'.join(ldm_key.split(".")[2:])  # model.diffusion_model.

    for sd_part, hf_part in unet_conversion_map_layer.items():
        diffusers_key = diffusers_key.replace(sd_part, hf_part)

    if "resnets" in diffusers_key:
        for sd_part, hf_part in unet_conversion_map_resnet.items():
            diffusers_key = diffusers_key.replace(sd_part, hf_part)

    diffusers_key = unet_conversion_map.get(diffusers_key, diffusers_key)

    diffusers_key = f"unet.{diffusers_key}"
    return diffusers_sd[diffusers_key]


# conversion dicts src:
# https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_stable_diffusion.py


unet_conversion_map = {
    # (stable-diffusion: HF Diffusers)
    "time_embed.0.weight": "time_embedding.linear_1.weight",
    "time_embed.0.bias": "time_embedding.linear_1.bias",
    "time_embed.2.weight": "time_embedding.linear_2.weight",
    "time_embed.2.bias": "time_embedding.linear_2.bias",
    "input_blocks.0.0.weight": "conv_in.weight",
    "input_blocks.0.0.bias": "conv_in.bias",
    "out.0.weight": "conv_norm_out.weight",
    "out.0.bias": "conv_norm_out.bias",
    "out.2.weight": "conv_out.weight",
    "out.2.bias": "conv_out.bias",
}


unet_conversion_map_resnet = {
    # (stable-diffusion, HF Diffusers)
    "in_layers.0": "norm1",
    "in_layers.2": "conv1",
    "out_layers.0": "norm2",
    "out_layers.3": "conv2",
    "emb_layers.1": "time_emb_proj",
    "skip_connection": "conv_shortcut",
}


unet_conversion_map_layer = {}
# hardcoded number of downblocks and resnets/attentions...
# would need smarter logic for other networks.
for i in range(4):
    # loop over downblocks/upblocks

    for j in range(2):
        # loop over resnets/attentions for downblocks
        hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
        sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
        unet_conversion_map_layer[sd_down_res_prefix] = hf_down_res_prefix

        if i < 3:
            # no attention layers in down_blocks.3
            hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
            sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
            unet_conversion_map_layer[sd_down_atn_prefix] = hf_down_atn_prefix

    for j in range(3):
        # loop over resnets/attentions for upblocks
        hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
        sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
        unet_conversion_map_layer[sd_up_res_prefix] = hf_up_res_prefix

        if i > 0:
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
            unet_conversion_map_layer[sd_up_atn_prefix] = hf_up_atn_prefix

    if i < 3:
        # no downsample in down_blocks.3
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
        sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
        unet_conversion_map_layer[sd_downsample_prefix] = hf_downsample_prefix

        # no upsample in up_blocks.3
        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"output_blocks.{3*i + 2}.{1 if i == 0 else 2}."
        unet_conversion_map_layer[sd_upsample_prefix] = hf_upsample_prefix

hf_mid_atn_prefix = "mid_block.attentions.0."
sd_mid_atn_prefix = "middle_block.1."
unet_conversion_map_layer[sd_mid_atn_prefix] = hf_mid_atn_prefix

for j in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{j}."
    sd_mid_res_prefix = f"middle_block.{2*j}."
    unet_conversion_map_layer[sd_mid_res_prefix] = hf_mid_res_prefix


def convert_clip_l(diffusers_sd: StateDict, ldm_key: str) -> Tensor:
    diffusers_key = '.'.join(ldm_key.split(".")[2:])  # cond_stage_model.transformer.
    diffusers_key = f"text_encoder.{diffusers_key}"
    return diffusers_sd[diffusers_key]


def convert_vae(diffusers_sd: StateDict, ldm_key: str) -> Tensor:
    diffusers_key = '.'.join(ldm_key.split(".")[1:])

    needs_reshape = False
    for sd_weight_name, fake_weight_name in vae_extra_conversion_map.items():
        if f"mid.attn_1.{sd_weight_name}.weight" in diffusers_key or f"mid.attn_1.{sd_weight_name}.bias" in diffusers_key:
            needs_reshape = True
            diffusers_key = diffusers_key.replace(sd_weight_name, fake_weight_name)

    for weight_name in vae_extra_conversion_map.values():
        if f"mid.attn_1.{weight_name}.weight" in diffusers_key:
            needs_reshape = True

    if "attentions" in diffusers_key:
        for sd_part, hf_part in vae_conversion_map_attn.items():
            diffusers_key = diffusers_key.replace(sd_part, hf_part)

    for sd_part, hf_part in vae_conversion_map.items():
        diffusers_key = diffusers_key.replace(sd_part, hf_part)

    diffusers_key = f"vae.{diffusers_key}"
    res = diffusers_sd[diffusers_key]
    if needs_reshape:
        res = reshape_weight_for_sd(res)
    return res


def reshape_weight_for_sd(w):
    # convert HF linear weights to SD conv2d weights
    if not w.ndim == 1:
        return w.reshape(*w.shape, 1, 1)
    else:
        return w


vae_conversion_map = {
    # (stable-diffusion, HF Diffusers)
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
    # (stable-diffusion, HF Diffusers)
    "norm.": "group_norm.",
    "q.": "query.",
    "k.": "key.",
    "v.": "value.",
    "proj_out.": "proj_attn.",
}


# This is probably not the most ideal solution, but it does work.
vae_extra_conversion_map = {
    # (stable-diffusion, HF Diffusers)
    "q": "to_q",
    "k": "to_k",
    "v": "to_v",
    "proj_out": "to_out.0",
}
