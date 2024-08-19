import torch.nn
import warnings
from model_configs.nn_module_config import create_config_from_module, Block, Component
from model_configs.paths import configs_dir
from model_configs.stable_diffusion_components import create_clip_l_component, create_vae_component, list_blocks
from sd_mecha.extensions.model_config import ModelConfig
from typing import Iterable


def get_venv() -> str:
    return "ldm"


def create_configs() -> Iterable[ModelConfig]:
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config

    config = str(configs_dir / "sd1-ldm-base.yaml")
    config = OmegaConf.load(config).model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = instantiate_from_config(config)

    components = (
        create_clip_l_component(model.cond_stage_model),
        create_vae_component(model.first_stage_model),
        create_unet_component(model.model.diffusion_model),
    )

    return [
        create_config_from_module(
            identifier="sd1-ldm",
            merge_space="weight",
            model=model,
            components=components,
        ),
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
            block.modules_to_merge += [unet.time_embed]

    return component
