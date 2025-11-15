import torch
from sd_mecha.extensions.merge_methods import merge_method, StateDict, Parameter, Return
from .convert_huggingface_sd_vae_to_original import convert_vae
from ... import model_configs

# hf to sd conversion src:
# https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_stable_diffusion.py


sd1_kohya = model_configs.resolve("sd1-kohya")
sd1_ldm = model_configs.resolve("sd1-ldm")


@merge_method(
    identifier=f"convert_'{sd1_kohya.identifier}'_to_'{sd1_ldm.identifier}'",
    is_conversion=True,
)
def convert_sd1_kohya_to_original(
    kohya_sd: Parameter(StateDict[torch.Tensor], model_config=sd1_kohya),
    **kwargs,
) -> Return(torch.Tensor, model_config=sd1_ldm):
    ldm_key = kwargs["key"]
    if ldm_key.startswith("model.diffusion_model."):
        return convert_unet(kohya_sd, ldm_key)
    elif ldm_key.startswith("cond_stage_model."):
        return convert_clip_l(kohya_sd, ldm_key)
    elif ldm_key.startswith("first_stage_model."):
        return convert_vae(kohya_sd, ldm_key)
    else:
        return kohya_sd[ldm_key]


def convert_unet(kohya_sd: StateDict, ldm_key: str) -> torch.Tensor:
    kohya_key = '.'.join(ldm_key.split(".")[2:])  # model.diffusion_model.

    for sd_part, hf_part in unet_conversion_map_layer.items():
        kohya_key = kohya_key.replace(sd_part, hf_part)

    if "resnets" in kohya_key:
        for sd_part, hf_part in unet_conversion_map_resnet.items():
            kohya_key = kohya_key.replace(sd_part, hf_part)

    kohya_key = unet_conversion_map.get(kohya_key, kohya_key)

    kohya_key = f"unet.{kohya_key}"
    return kohya_sd[kohya_key]


unet_conversion_map = {
    # (original: huggingface)
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
    # (original: huggingface)
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


def convert_clip_l(kohya_sd: StateDict, ldm_key: str) -> torch.Tensor:
    kohya_key = '.'.join(ldm_key.split(".")[2:])  # cond_stage_model.transformer.
    kohya_key = f"te.{kohya_key}"
    return kohya_sd[kohya_key]
