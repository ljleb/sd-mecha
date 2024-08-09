import torch.nn
from model_configs.nn_module_config import Block, Component
from typing import Iterable


def create_clip_l_component(clip_l: torch.nn.Module) -> Component:
    component = Component("clip_l", clip_l, [
        *list_blocks("in", clip_l.transformer.text_model.encoder.layers.children()),
    ])
    component.blocks[0].modules_to_merge += [
        clip_l.transformer.text_model.embeddings.token_embedding,
        clip_l.transformer.text_model.embeddings.position_embedding,
    ]
    if hasattr(clip_l.transformer.text_model, "final_layer_norm"):
        component.blocks[-1].modules_to_merge.append(clip_l.transformer.text_model.final_layer_norm)
    if hasattr(clip_l.transformer, "text_projection"):
        component.blocks[-1].modules_to_merge.append(clip_l.transformer.text_projection)
    if hasattr(clip_l, "logit_scale"):
        component.blocks[-1].modules_to_merge.append(clip_l.logit_scale)

    return component


def create_vae_component(vae: torch.nn.Module) -> Component:
    component = Component("vae", vae, [
        *list_blocks("in", [*vae.encoder.down.children()] + [vae.encoder.mid], copy=True),
        *list_blocks("out", [vae.decoder.mid] + [*vae.decoder.up.children()], copy=True),
    ], copy_only=True)
    component.blocks[0].modules_to_copy += [vae.encoder.conv_in]
    component.blocks[4].modules_to_copy += [vae.encoder.norm_out, vae.encoder.conv_out, vae.quant_conv]
    component.blocks[5].modules_to_copy += [vae.post_quant_conv, vae.decoder.conv_in]
    component.blocks[-1].modules_to_copy += [vae.decoder.norm_out, vae.decoder.conv_out]

    return component


def list_blocks(block_id_prefix: str, modules: Iterable[torch.nn.Module], copy: bool = False):
    return [
        Block(
            identifier=f"{block_id_prefix}{i}",
            **{"modules_to_copy" if copy else "modules_to_merge": [module]},
        )
        for i, module in enumerate(modules)
    ]
