import torch.nn
from builtin_models.nn_module_config import create_config_from_module, Component, Block
from builtin_models.paths import repositories_dir
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
                create_txt1_component(model),
                create_txt2_component(model),
                create_unet_component(model),
            ),
        ),
    ]


def create_txt1_component(model: torch.nn.Module) -> Component:
    txt = model.conditioner.embedders[0].transformer.text_model
    component = Component("txt1", txt, [
        *list_blocks("in", txt.encoder.layers.children()),
    ])
    component.blocks[0].includes += [txt.embeddings.token_embedding, txt.embeddings.position_embedding]
    component.blocks[-1].includes += [txt.final_layer_norm]

    return component


def create_txt2_component(model: torch.nn.Module) -> Component:
    txt = model.conditioner.embedders[1].model
    component = Component("txt2", txt, [
        *list_blocks("in", txt.transformer.resblocks.children()),
    ])
    component.blocks[0].includes += [txt.token_embedding, txt.positional_embedding]
    component.blocks[-1].includes += [txt.ln_final, txt.text_projection, txt.logit_scale]

    return component


def create_unet_component(model: torch.nn.Module):
    unet = model.model.diffusion_model
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


def list_blocks(block_id_prefix: str, modules: Iterable[torch.nn.Module]):
    return [
        Block(
            identifier=f"{block_id_prefix}{i}",
            includes=[module]
        )
        for i, module in enumerate(modules)
    ]
