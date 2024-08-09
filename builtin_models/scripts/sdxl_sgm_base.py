import torch
from builtin_models.nn_module_config import create_config_from_module, Block, Component
from builtin_models.paths import repositories_dir
from builtin_models.stable_diffusion_components import create_txt_component, create_vae_component, list_blocks
from sd_mecha.extensions.model_config import ModelConfig
from typing import Iterable


def get_venv() -> str:
    return "sgm"


def create_configs() -> Iterable[ModelConfig]:
    from omegaconf import OmegaConf
    from sgm.util import instantiate_from_config

    config = str(repositories_dir / "stability-ai-generative-models" / "configs" / "inference" / "sd_xl_base.yaml")
    config = OmegaConf.load(config).model
    model = instantiate_from_config(config)

    return [
        create_config_from_module(
            identifier="sdxl-sgm-base",
            merge_space="weight",
            model=model,
            components=(
                create_unet_component(model.model.diffusion_model),
                create_txt_component(model.conditioner.embedders[0].transformer),
                create_txt2_component(model.conditioner.embedders[1].model),
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
            block.includes += [unet.time_embed, unet.label_emb]

    return component


def create_txt2_component(txt2: torch.nn.Module) -> Component:
    component = Component("txt2", txt2, [
        *list_blocks("in", txt2.transformer.resblocks.children()),
    ])
    component.blocks[0].includes += [txt2.token_embedding, txt2.positional_embedding]
    component.blocks[-1].includes += [txt2.ln_final, txt2.text_projection, txt2.logit_scale]

    return component
