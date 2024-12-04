import torch
from torch import Tensor
from sd_mecha.extensions.model_config import ModelConfig
from sd_mecha.extensions.merge_method import convert_to_recipe, StateDict, config_conversion
from sd_mecha.merge_methods import SameMergeSpace


@config_conversion
@convert_to_recipe(identifier="convert_'sdxl-diffusers'_to_'sdxl-sgm'")
def convert_sdxl_diffusers_to_original(
    diffusers_sd: StateDict | ModelConfig["sdxl-diffusers"] | SameMergeSpace,
    **kwargs,
) -> Tensor | ModelConfig["sdxl-sgm"] | SameMergeSpace:
    sgm_key = kwargs["key"]

    if sgm_key.startswith("model.diffusion_model."):
        diffusers_key = sgm_key.replace("model.diffusion_model.", "unet.")
        return diffusers_sd[diffusers_key]
    elif sgm_key.startswith("conditioner.embedders.0.transformer."):
        diffusers_key = sgm_key.replace("conditioner.embedders.0.transformer.", "text_model.")
        return diffusers_sd[diffusers_key]
    elif sgm_key.startswith("conditioner.embedders.1.model.transformer.resblocks."):
        if sgm_key.endswith("text_projection"):
            diffusers_key = "text_encoder_2.text_projection"
        else:
            diffusers_key = sgm_key.replace("conditioner.embedders.1.model.", "text_encoder_2.text_model.")
            diffusers_key = diffusers_key.replace(".transformer.resblocks.", ".encoder.layers.")
            diffusers_key = diffusers_key.replace(".attn.", ".self_attn.")
            diffusers_key = diffusers_key.replace(".mlp.c_fc.", ".mlp_fc1.")
            diffusers_key = diffusers_key.replace(".mlp.c_proj.", ".mlp_fc2.")

        if diffusers_key.endswith((".in_proj_weight", ".in_proj_bias")):
            is_bias = diffusers_key.endswith("bias")
            partial_key = diffusers_key.replace(".in_proj_weight", "").replace(".in_proj_bias", "")
            res = torch.vstack([
                diffusers_sd[f"{partial_key}.{k}_proj.{'bias' if is_bias else 'weight'}"]
                for k in ("q", "k", "v")
            ])
        else:
            res = diffusers_sd[diffusers_key]

        if sgm_key.endswith("text_projection"):
            res = res.T.contiguous()

        return res
