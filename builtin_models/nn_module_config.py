import dataclasses
import re

import torch
from sd_mecha.extensions.merge_space import MergeSpace, get_identifiers
from sd_mecha.extensions.model_config import ModelConfigBlock, ModelConfigComponent, ModelConfig, StateDictKey
from sd_mecha.streaming import TensorMetadata
from typing import Iterable, Dict, List


@dataclasses.dataclass
class Block:
    identifier: str
    includes: List[torch.nn.Module] = dataclasses.field(default_factory=list)
    excludes: List[torch.nn.Module] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Component:
    identifier: str
    module: torch.nn.Module
    blocks: List[Block] = dataclasses.field(default_factory=list)


def create_config_from_module(
    identifier: str,
    merge_space: str,
    model: torch.nn.Module,
    components: Component | Iterable[Component] = (),
) -> ModelConfig:
    if not isinstance(components, Iterable):
        components = [components]

    header, state_dict_keys, direct_state_dict_keys = header_from_model(model)

    named_modules = dict(model.named_modules())
    module_name_map = {}
    for module_name, module in named_modules.items():
        try:
            module_name_map[module] = module_name
        except TypeError:
            pass

    config_components = {}
    keys_to_copy = header
    for component in components:
        config_blocks = {}
        for block in component.blocks:
            all_block_keys = {
                k: header[k]
                for module_to_include in block.includes
                for k in state_dict_keys[module_name_map[module_to_include]]
            }
            block_keys_to_copy = {
                k: header[k]
                for module_to_exclude in block.excludes
                for k in direct_state_dict_keys[module_name_map[module_to_exclude]]
            }
            block_keys_to_merge = {
                k: v
                for k, v in all_block_keys.items()
                if k not in block_keys_to_copy
            }
            keys_to_copy = {
                k: v
                for k, v in keys_to_copy.items()
                if k not in block_keys_to_merge
            }
            config_blocks[block.identifier] = ModelConfigBlock(block_keys_to_merge)

        config_components[component.identifier] = ModelConfigComponent(config_blocks)

    # validate merge space
    merge_space = get_identifiers(MergeSpace(merge_space))[0]

    return ModelConfig(
        identifier=identifier,
        merge_space=merge_space,
        keys_to_copy=keys_to_copy,
        components=config_components,
    )


def header_from_model(model: torch.nn.Module):
    state_dict = model.state_dict(keep_vars=True)
    direct_state_dict_keys: Dict[str, Iterable[StateDictKey]] = {}
    state_dict_hooks = {}

    def create_direct_state_dict_hook(module_name):
        def hook(module, partial_state_dict, _prefix, _local_metadata):
            nonlocal direct_state_dict_keys, state_dict, state_dict_hooks
            state_dict_hooks[module_name].remove()

            module_state_dict_values = module.state_dict(keep_vars=True).values()
            direct_keys = set(
                k
                for k, v in state_dict.items()
                if any(v is v2 for v2 in module_state_dict_values)
            )
            if direct_keys:
                direct_state_dict_keys[module_name] = direct_keys
            return partial_state_dict
        return hook

    for name, module in model.named_modules():
        state_dict_hooks[name] = module._register_state_dict_hook(create_direct_state_dict_hook(name))

    header = header_from_state_dict(model.state_dict(keep_vars=True))

    state_dict_keys: Dict[str, Iterable[StateDictKey]] = direct_state_dict_keys.copy()
    for name, module in model.named_modules():
        if not list(module.children()):
            continue

        default = set()
        for module_name, module_keys in list(direct_state_dict_keys.items()):
            if re.match(rf"^{re.escape(name)}(?=\.|$)", module_name):
                state_dict_keys[name] = state_dict_keys.get(name, default) | module_keys

    return header, state_dict_keys, direct_state_dict_keys


def header_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, TensorMetadata]:
    return {k: TensorMetadata(v.shape, v.dtype) for k, v in state_dict.items()}
