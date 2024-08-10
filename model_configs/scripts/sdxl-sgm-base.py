import logging

import torch

from model_configs.lycoris_config import create_lycoris_configs, create_kohya_config
from model_configs.nn_module_config import create_config_from_module, Block, Component
from model_configs.paths import configs_dir
from model_configs.stable_diffusion_components import create_clip_l_component, create_vae_component, list_blocks
from sd_mecha.extensions.model_config import ModelConfig
from typing import Iterable


def get_venv() -> str:
    return "sgm"


def create_configs() -> Iterable[ModelConfig]:
    from omegaconf import OmegaConf
    from sgm.util import instantiate_from_config
    from sgm.modules.attention import logpy as attention_logging

    attention_logging.setLevel(logging.ERROR)

    config = str(configs_dir / "sdxl-sgm-base.yaml")
    config = OmegaConf.load(config).model
    model = instantiate_from_config(config)

    components = (
        create_unet_component(model.model.diffusion_model),
        create_clip_l_component(model.conditioner.embedders[0]),
        create_clip_g_component(model.conditioner.embedders[1].model),
        create_vae_component(model.first_stage_model),
    )
    lycoris_components = components[:-1]

    return [
        create_config_from_module(
            identifier="sdxl-sgm-base",
            merge_space="weight",
            model=model,
            components=components,
        ),
        *create_lycoris_configs(
            identifier="sdxl",
            model=model,
            components=components,
        ),
        create_kohya_config(
            identifier="sdxl",
            model=model,
            text_encoders=list(model.conditioner.embedders),
            components=lycoris_components,
        )
    ]


def create_unet_component(unet: torch.nn.Module):
    component = Component("unet", unet, [
        *list_blocks("in", unet.input_blocks.children()),
        Block("mid", [unet.middle_block]),
        *list_blocks("out", unet.output_blocks.children()),
    ])
    component.blocks[-1].modules_to_merge += [unet.out]
    for i, block in enumerate(component.blocks):
        if not block.identifier.startswith("in") or i % 3 != 0:
            block.modules_to_merge += [unet.time_embed, unet.label_emb]

    return component


def create_clip_g_component(clip_g: torch.nn.Module) -> Component:
    component = Component("clip_g", clip_g, [
        *list_blocks("in", clip_g.transformer.resblocks.children()),
    ])
    component.blocks[0].modules_to_merge += [clip_g.token_embedding, clip_g.positional_embedding]
    component.blocks[-1].modules_to_merge += [clip_g.ln_final, clip_g.text_projection, clip_g.logit_scale]

    return component
