import pathlib
import torch.nn
from builtin_models.disable_init import DisableInitialization, MetaTensorMode
from builtin_models.nn_module_config import create_config_from_module, Component, Block
from builtin_models.paths import extra_path, configs_dir, repositories_dir
from sd_mecha.extensions.model_config import ModelConfig
from typing import List, Iterable


ldm_path = repositories_dir / "stability-ai-stable-diffusion"


def get_identifier() -> str:
    return "sd1-ldm-base"


def get_requirements() -> List[str | pathlib.Path]:
    return [
        "-r", ldm_path / "requirements.txt",
    ]


def create_config() -> ModelConfig:
    with (
        extra_path(ldm_path),
        DisableInitialization(),
        MetaTensorMode()
    ):
        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config

        config = str(configs_dir / "v1-inference.yaml")
        config = OmegaConf.load(config).model
        model = instantiate_from_config(config)

        return create_config_from_module(
            identifier=get_identifier(),
            merge_space="weight",
            model=model,
            components=(
                create_txt_component(model),
                create_unet_component(model),
            ),
        )


def create_txt_component(model: torch.nn.Module) -> Component:
    txt = model.cond_stage_model.transformer.text_model
    blocks = []
    component = Component("txt", txt, [
        *list_blocks("in", txt.encoder.layers.children()),
    ])
    for block in component.blocks:
        if block.identifier == "in0":
            block.includes += [txt.embeddings.token_embedding, txt.embeddings.position_embedding]
        if block.identifier == "in11":
            block.includes += [txt.final_layer_norm]

    return component


def create_unet_component(model: torch.nn.Module):
    unet = model.model.diffusion_model
    component = Component("unet", unet, [
        *list_blocks("in", unet.input_blocks.children()),
        Block("mid", [unet.middle_block]),
        *list_blocks("out", unet.output_blocks.children()),
    ])
    for i, block in enumerate(component.blocks):
        if not block.identifier.startswith("in") or i % 3 != 0:
            block.includes += [unet.time_embed]
        if block.identifier == "out11":
            block.includes += [unet.out]

    return component


def list_blocks(block_id_prefix: str, modules: Iterable[torch.nn.Module]):
    return [
        Block(
            identifier=f"{block_id_prefix}{i}",
            includes=[module]
        )
        for i, module in enumerate(modules)
    ]
