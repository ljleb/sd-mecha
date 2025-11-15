from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return, StateDict
from sd_mecha.streaming import StateDictKeyError
from sd_mecha.extensions import model_configs
from torch import Tensor


# hf to sd conversion src:
# https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_stable_diffusion.py


sdxl_diffusers_unet_config = model_configs.resolve('sdxl-diffusers_unet_only')
sdxl_sgm_config = model_configs.resolve('sdxl-sgm')


@merge_method(
    identifier=f"convert_'{sdxl_diffusers_unet_config.identifier}'_to_'{sdxl_sgm_config.identifier}'",
    is_conversion=True,
)
def convert_sdxl_diffusers_unet_to_original(
    diffusers_sd: Parameter(StateDict[Tensor], model_config=sdxl_diffusers_unet_config),
    **kwargs,
) -> Return(Tensor, model_config=sdxl_sgm_config):
    sgm_key = kwargs["key"]
    if not sgm_key.startswith("model.diffusion_model"):
        raise StateDictKeyError(sgm_key)

    kohya_key = ".".join(sgm_key.split(".")[2:])  # model.diffusion_model.

    for sd_part, hf_part in unet_conversion_map_layer.items():
        kohya_key = kohya_key.replace(sd_part, hf_part)

    if "resnets" in kohya_key:
        for sd_part, hf_part in unet_conversion_map_resnet.items():
            kohya_key = kohya_key.replace(sd_part, hf_part)

    kohya_key = unet_conversion_map.get(kohya_key, kohya_key)
    return diffusers_sd[kohya_key]


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
    # the following are for sdxl
    "label_emb.0.0.weight": "add_embedding.linear_1.weight",
    "label_emb.0.0.bias": "add_embedding.linear_1.bias",
    "label_emb.0.2.weight": "add_embedding.linear_2.weight",
    "label_emb.0.2.bias": "add_embedding.linear_2.bias",
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
for i in range(3):
    # loop over downblocks/upblocks

    for j in range(2):
        # loop over resnets/attentions for downblocks
        hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
        sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
        unet_conversion_map_layer[sd_down_res_prefix] = hf_down_res_prefix

        if i > 0:
            hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
            sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
            unet_conversion_map_layer[sd_down_atn_prefix] = hf_down_atn_prefix

    for j in range(4):
        # loop over resnets/attentions for upblocks
        hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
        sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
        unet_conversion_map_layer[sd_up_res_prefix] = hf_up_res_prefix

        if i < 2:
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3 * i + j}.1."
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
unet_conversion_map_layer["output_blocks.2.2.conv."] = "output_blocks.2.1.conv."

hf_mid_atn_prefix = "mid_block.attentions.0."
sd_mid_atn_prefix = "middle_block.1."
unet_conversion_map_layer[sd_mid_atn_prefix] = hf_mid_atn_prefix
for j in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{j}."
    sd_mid_res_prefix = f"middle_block.{2*j}."
    unet_conversion_map_layer[sd_mid_res_prefix] = hf_mid_res_prefix
