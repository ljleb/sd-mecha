import torch
import yaml
from model_configs.nn_module_config import create_config_from_module, Component
from model_configs.paths import configs_dir
from model_configs.stable_diffusion_components import list_blocks, create_clip_l_component, create_t5xxl_component, create_vae_component
from sd_mecha.extensions.model_config import ModelConfig
from typing import Iterable


def get_venv() -> str:
    return "comfyui"


def create_configs() -> Iterable[ModelConfig]:
    with open(configs_dir / "sd3-comfyui-base.yaml") as f:
        config = yaml.safe_load(f.read())
    model = SD3Model(config)

    return [
        create_config_from_module(
            identifier="sd3-comfyui",
            merge_space="weight",
            model=model,
            components=(
                create_unet_component(model.model.diffusion_model),
                create_clip_l_component(model.text_encoders.clip_l),
                create_clip_g_component(model.text_encoders.clip_g),
                create_t5xxl_component(model.text_encoders.t5xxl),
                create_vae_component(model.first_stage_model),
            ),
        ),
    ]


def create_unet_component(unet: torch.nn.Module):
    component = Component("unet", unet, [
        *list_blocks("in", unet.joint_blocks.children()),
    ])
    component.blocks[0].modules_to_merge += [unet.x_embedder, unet.pos_embed, unet.context_embedder]
    component.blocks[-1].modules_to_merge += [unet.final_layer]
    for i, block in enumerate(component.blocks):
        block.modules_to_merge += [unet.t_embedder, unet.y_embedder]

    return component


def create_clip_g_component(clip_g: torch.nn.Module) -> Component:
    component = Component("clip_g", clip_g, [
        *list_blocks("in", clip_g.transformer.text_model.encoder.layers.children()),
    ])
    component.blocks[0].modules_to_merge += [
        clip_g.transformer.text_model.embeddings.token_embedding,
        clip_g.transformer.text_model.embeddings.position_embedding,
    ]
    component.blocks[-1].modules_to_merge += [
        clip_g.transformer.text_model.final_layer_norm,
        clip_g.transformer.text_projection,
        clip_g.logit_scale,
    ]

    return component


class SD3Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        from comfy.supported_models import SD3
        from comfy.ldm.models.autoencoder import AutoencodingEngine

        self.model = SD3(config["diffusion_model"]).get_model({})
        self.first_stage_model = AutoencodingEngine(**config["first_stage_model"])
        self.text_encoders = SD3TextEncoders(config)


class SD3TextEncoders(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        from comfy import sd1_clip, sdxl_clip
        from comfy.text_encoders import sd3_clip

        self.clip_l = sd1_clip.SDClipModel(**config["clip_l_model"])
        self.clip_g = sdxl_clip.SDXLClipG()
        self.t5xxl = sd3_clip.T5XXLModel()
