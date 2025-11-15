import torch
from torch import Tensor
from sd_mecha.extensions.merge_methods import merge_method, StateDict, Return, Parameter
from sd_mecha.extensions import model_configs
from .convert_huggingface_sd_vae_to_original import convert_vae


sdxl_kohya_config = model_configs.resolve('sdxl-kohya')
sdxl_sgm_config = model_configs.resolve('sdxl-sgm')


@merge_method(
    identifier=f"convert_'{sdxl_kohya_config.identifier}'_to_'{sdxl_sgm_config.identifier}'",
    is_conversion=True,
)
def convert_sdxl_kohya_to_original(
    kohya_sd: Parameter(StateDict[Tensor], model_config=sdxl_kohya_config),
    **kwargs,
) -> Return(Tensor, model_config=sdxl_sgm_config):
    sgm_key = kwargs["key"]

    if sgm_key.startswith("model.diffusion_model."):
        kohya_key = sgm_key.replace("model.diffusion_model.", "unet.")
        return kohya_sd[kohya_key]
    elif sgm_key.startswith("conditioner.embedders.0."):
        kohya_key = sgm_key.replace("conditioner.embedders.0.transformer.", "te1.")
        return kohya_sd[kohya_key]
    elif sgm_key.startswith("conditioner.embedders.1."):
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
            res = torch.vstack([
                kohya_sd[f"{partial_key}.{k}_proj.{'bias' if is_bias else 'weight'}"]
                for k in ("q", "k", "v")
            ])
        else:
            res = kohya_sd[kohya_key]

        if sgm_key.endswith("text_projection"):
            res = res.T

        return res
    elif sgm_key.startswith("first_stage_model."):
        return convert_vae(kohya_sd, sgm_key)
    else:
        return kohya_sd[sgm_key]
