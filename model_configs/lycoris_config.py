import functools

import lycoris
import torch.nn
from model_configs.nn_module_config import Component, create_config_from_module, Block
from typing import Iterable, List

from sd_mecha.extensions.model_config import ModelConfig


def create_lycoris_configs(
    identifier: str,
    model: torch.nn.Module,
    components: Component | Iterable[Component],
):
    configs = []
    for algo in ["lora"]:  # replace with loop over algos
        lycoris_wrapper = lycoris.create_lycoris(
            model,
            1.0,
            linear_dim=16,
            linear_alpha=1.0,
            algo=algo,
        )
        lycoris_wrapper.apply_to()
        configs.append(create_config_from_lycoris_module(
            f"{identifier}-lycoris-{algo}",
            merge_space="delta",
            lycoris_model=lycoris_wrapper,
            components=components,
        ))
        lycoris_wrapper.restore()

    return configs


def create_kohya_config(
    identifier: str,
    model: torch.nn.Module,
    text_encoders: torch.nn.Module | List[torch.nn.Module],
    components: Component | Iterable[Component],
):
    kohya_wrapper = lycoris.kohya.create_network(
        1.0,
        16,
        1.0,
        model.first_stage_model,
        text_encoders,
        model.model.diffusion_model,
    )
    kohya_wrapper.apply_to(text_encoders, model.model.diffusion_model, apply_text_encoder=True, apply_unet=True)
    try:
        return create_config_from_lycoris_module(
            f"{identifier}-kohya-lora",
            merge_space="delta",
            lycoris_model=kohya_wrapper,
            components=components,
        )
    finally:
        kohya_wrapper.restore()


def create_config_from_lycoris_module(
    identifier: str,
    merge_space: str,
    lycoris_model: torch.nn.Module,
    components: Component | Iterable[Component] = (),
) -> ModelConfig:
    if not isinstance(components, Iterable):
        components = [components]
    else:
        components = list(components)

    lycoris_components = []
    for component in components:
        lycoris_component_blocks = []
        lycoris_component = Component(
            component.identifier,
            None,
            lycoris_component_blocks,
        )

        for block in component.blocks:
            lycoris_block_modules_to_merge = []
            lycoris_block_modules_to_copy = []
            lycoris_block = Block(block.identifier, lycoris_block_modules_to_merge, lycoris_block_modules_to_copy)

            for lycoris_module_name, lycoris_module in lycoris_model.named_children():
                module = lycoris_module.org_module[0]

                for module_to_merge in block.modules_to_merge:
                    if module is module_to_merge or hasattr(module_to_merge, "modules") and module in module_to_merge.modules():
                        lycoris_block_modules_to_merge.append(lycoris_module)

                for module_to_copy in block.modules_to_copy:
                    if module is module_to_copy or hasattr(module_to_copy, "modules") or module in module_to_copy.modules():
                        lycoris_block_modules_to_copy.append(lycoris_module)

            lycoris_component_blocks.append(lycoris_block)

        lycoris_components.append(lycoris_component)

    return create_config_from_module(
        identifier,
        merge_space,
        lycoris_model,
        lycoris_components,
    )
