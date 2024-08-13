import functools

import lycoris
import torch.nn
from lycoris.wrapper import network_module_dict
from model_configs.nn_module_config import Component, create_config_from_module, Block
from typing import Iterable
from sd_mecha.extensions.model_config import ModelConfig


def create_lycoris_configs(
    arch_impl_identifier: str,
    model: torch.nn.Module,
    components: Component | Iterable[Component],
    add_kohya: bool = False,
):
    assert arch_impl_identifier.count("-") == 1, "both the architecture and implementation separated by '-' need to be specified"
    configs = []

    def add_to_configs(lycoris_impl_identifier: str, lycoris_wrapper, algo):
        config = create_config_from_lycoris_module(
            f"{arch_impl_identifier}_{lycoris_impl_identifier}-{algo}",
            merge_space="delta",
            lycoris_model=lycoris_wrapper,
            components=components,
        )
        configs.append(config)

    for algo, create_lycoris in lycoris_map.items():
        lycoris_wrapper = create_lycoris(model)
        lycoris_wrapper.apply_to()
        add_to_configs("lycoris", lycoris_wrapper, algo)
        lycoris_wrapper.restore()
        if add_kohya and algo == "lora":
            lycoris_wrapper.LORA_PREFIX = "lora"
            lycoris_wrapper.apply_to()
            add_to_configs("kohya", lycoris_wrapper, algo)
            lycoris_wrapper.restore()

    return configs


lycoris_map = {
    "lora": functools.partial(
        lycoris.create_lycoris,
        multiplier=1.0,
        linear_dim=4,
        linear_alpha=1.0,
        algo="lora",
        train_norm=True,
    ),
    "loha": functools.partial(
        lycoris.create_lycoris,
        multiplier=1.0,
        linear_dim=4,
        linear_alpha=1.0,
        algo="loha",
        train_norm=True,
    ),
    "lokr": functools.partial(
        lycoris.create_lycoris,
        multiplier=1.0,
        linear_dim=4,
        linear_alpha=1.0,
        algo="lokr",
        train_norm=True,
    ),
    "dylora": functools.partial(
        lycoris.create_lycoris,
        multiplier=1.0,
        linear_dim=4,
        linear_alpha=1.0,
        algo="dylora",
        train_norm=True,
    ),
    "glora": functools.partial(
        lycoris.create_lycoris,
        multiplier=1.0,
        linear_dim=4,
        linear_alpha=1.0,
        algo="glora",
        train_norm=True,
    ),
    "full": functools.partial(
        lycoris.create_lycoris,
        multiplier=1.0,
        linear_dim=4,
        linear_alpha=1.0,
        algo="full",
        train_norm=True,
    ),
    "doft": functools.partial(
        lycoris.create_lycoris,
        multiplier=1.0,
        linear_dim=4,
        linear_alpha=1.0,
        algo="diag-oft",
        train_norm=True,
    ),
    # "boft": functools.partial(
    #     lycoris.create_lycoris,
    #     multiplier=1.0,
    #     linear_dim=8,
    #     conv_dim=8,
    #     linear_alpha=1.0,
    #     algo="boft",
    #     train_norm=True,
    # ),
}


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
