import torch.nn
from builtin_models.nn_module_config import create_config_from_module, Component, Block
from builtin_models.paths import configs_dir
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
                create_unet_component(model),
                create_txt_component(model),
                create_vae_component(model),
            ),
        ),
    ]


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
            block.includes += [unet.time_embed]

    return component


def create_txt_component(model: torch.nn.Module) -> Component:
    txt = model.cond_stage_model.transformer
    component = Component("txt", txt, [
        *list_blocks("in", txt.text_model.encoder.layers.children()),
    ])
    component.blocks[0].includes += [txt.text_model.embeddings.token_embedding, txt.text_model.embeddings.position_embedding]
    component.blocks[-1].includes += [txt.text_model.final_layer_norm, txt.text_projection]

    return component


def create_vae_component(model: torch.nn.Module) -> Component:
    vae = model.first_stage_model
    component = Component("vae", vae, [
        *list_blocks("in", [*vae.encoder.down.children()] + [vae.encoder.mid]),
        *list_blocks("out", [vae.decoder.mid] + [*vae.decoder.up.children()]),
    ], copy_only=True)
    component.blocks[0].includes += [vae.encoder.conv_in]
    component.blocks[4].includes += [vae.encoder.norm_out, vae.encoder.conv_out, vae.quant_conv]
    component.blocks[5].includes += [vae.post_quant_conv, vae.decoder.conv_in]
    component.blocks[-1].includes += [vae.decoder.norm_out, vae.decoder.conv_out]

    return component


def list_blocks(block_id_prefix: str, modules: Iterable[torch.nn.Module]):
    return [
        Block(
            identifier=f"{block_id_prefix}{i}",
            includes=[module]
        )
        for i, module in enumerate(modules)
    ]
