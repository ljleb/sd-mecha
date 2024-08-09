import torch.nn
from builtin_models.nn_module_config import Block, Component
from typing import Iterable


def create_txt_component(txt: torch.nn.Module) -> Component:
    component = Component("txt", txt, [
        *list_blocks("in", txt.text_model.encoder.layers.children()),
    ])
    component.blocks[0].includes += [txt.text_model.embeddings.token_embedding, txt.text_model.embeddings.position_embedding]
    component.blocks[-1].includes += [txt.text_model.final_layer_norm, txt.text_projection]

    return component


def create_vae_component(vae: torch.nn.Module) -> Component:
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
