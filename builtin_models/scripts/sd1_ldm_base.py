import torch.nn
from builtin_models.nn_module_config import create_config_from_module, Block, Component
from builtin_models.paths import configs_dir
from builtin_models.stable_diffusion_components import create_txt_component, create_vae_component, list_blocks
from sd_mecha.extensions.model_config import ModelConfig
from typing import Iterable


def get_venv() -> str:
    return "ldm"


def create_configs() -> Iterable[ModelConfig]:
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config

    config = str(configs_dir / "v1-inference.yaml")
    config = OmegaConf.load(config).model
    model = instantiate_from_config(config)

    return [
        create_config_from_module(
            identifier="sd1-ldm-base",
            merge_space="weight",
            model=model,
            components=(
                create_unet_component(model.model.diffusion_model),
                create_txt_component(model.cond_stage_model.transformer),
                create_vae_component(model.first_stage_model),
            ),
        ),
    ]


def create_unet_component(unet: torch.nn.Module):
    component = Component("unet", unet, [
        *list_blocks("in", unet.input_blocks.children()),
        Block("mid", [unet.middle_block]),
        *list_blocks("out", unet.output_blocks.children()),
    ])
    component.blocks[-1].includes += [unet.out]
    for i, block in enumerate(component.blocks):
        if not block.identifier.startswith("in") or i % 3 != 0:
            block.includes += [unet.time_embed]

    return component
