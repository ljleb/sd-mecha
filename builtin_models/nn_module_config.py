import dataclasses
import pathlib
import re
import torch
import traceback

import yaml

from sd_mecha.extensions.merge_space import MergeSpace, get_identifiers
from sd_mecha.extensions.model_config import ModelConfigComponent, ModelConfig, StateDictKey
from sd_mecha.streaming import TensorMetadata
from typing import Iterable, Dict, List, Tuple


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
    stack_frame = traceback.extract_stack(None, 2)[0]

    if not isinstance(components, Iterable):
        components = [components]
    components = {c.identifier: c for c in components}

    header, state_dict_keys, _ = header_from_model(model)

    named_modules = dict(model.named_modules())
    module_name_map = {}
    for module_name, module in named_modules.items():
        try:
            module_name_map[module] = module_name
        except TypeError:
            pass

    model_keys_to_merge = set()
    config_components = {}
    for component in components.values():
        component_block_keys = {}
        component_keys_to_merge = set()
        for block in component.blocks:
            block_keys = set(
                k
                for module_to_include in block.includes
                for k in state_dict_keys[module_name_map[module_to_include]]
            )
            component_block_keys[block.identifier] = block_keys
            component_keys_to_merge.update(block_keys.difference(
                k
                for module_to_exclude in block.excludes
                for k in state_dict_keys[module_name_map[module_to_exclude]]
            ))

        model_keys_to_merge.update(component_keys_to_merge)

        config_component = ModelConfigComponent(
            component.identifier,
            component_block_keys,
        )
        config_components[component.identifier] = config_component

    # validate merge space
    merge_space = get_identifiers(MergeSpace(merge_space))[0]

    return ModelConfig(
        identifier=identifier,
        header=header,
        components=config_components,
        keys_to_merge=model_keys_to_merge,
        merge_space=merge_space,
        _stack_frame=stack_frame,
    )


def header_from_model(model: torch.nn.Module):
    leaf_state_dict_keys: Dict[str, Iterable[StateDictKey]] = {}
    seen_keys = set()
    state_dict_hooks = {}

    def create_leaf_state_dict_hook(module_name):
        def hook(module, state_dict, _prefix, _local_metadata):
            nonlocal leaf_state_dict_keys, seen_keys, state_dict_hooks
            state_dict_hooks[module_name].remove()

            module_state_dict_values = module.state_dict(keep_vars=True).values()
            new_keys = set(
                k
                for k, v in state_dict.items()
                if k not in seen_keys and any(v is v2 for v2 in module_state_dict_values)
            )
            if new_keys:
                leaf_state_dict_keys[module_name] = new_keys
                seen_keys |= new_keys
            return state_dict
        return hook

    for name, module in model.named_modules():
        state_dict_hooks[name] = module._register_state_dict_hook(create_leaf_state_dict_hook(name))

    header = header_from_state_dict(model.state_dict(keep_vars=True))

    state_dict_keys: Dict[str, Iterable[StateDictKey]] = leaf_state_dict_keys.copy()
    for name, module in model.named_modules():
        if not list(module.children()):
            continue

        default = set()
        for module_name, module_keys in list(leaf_state_dict_keys.items()):
            if module_name.startswith(name):
                state_dict_keys[name] = state_dict_keys.get(name, default) | module_keys

    return header, state_dict_keys, leaf_state_dict_keys


def header_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, TensorMetadata]:
    return {k: TensorMetadata(v.shape, v.dtype) for k, v in state_dict.items()}


def create_config_from_key_matcher(
    identifier: str,
    merge_space: str,
    model: torch.nn.Module,
    keys_config: str | pathlib.Path,
) -> ModelConfig:
    header, _, _ = header_from_model(model)
    with open(keys_config, "r") as f:
        yaml_keys_config = yaml.safe_load(f.read())

    components, keys_to_merge = discover_blocks(header, yaml_keys_config, identifier)
    return ModelConfig(
        identifier,
        header,
        keys_to_merge,
        components,
        merge_space,
    )


WILDCARD = "*"


def discover_blocks(
    keys: Iterable[StateDictKey],
    keys_config: dict,
    identifier: str,
) -> Tuple[Dict[str, ModelConfigComponent], List[StateDictKey]]:
    discovered_blocks = {}

    for component, component_config in keys_config["merge"].items():
        prefix = component_config["prefix"]
        for shorthand, patterns in component_config["blocks"].items():
            if isinstance(patterns, str):
                patterns = [patterns]

            first_pattern_re = re.escape(patterns[0]).replace(re.escape(WILDCARD), r'(\w+)')
            pattern = re.compile(rf"^{re.escape(prefix)}\.{first_pattern_re}")

            for key in keys:
                match = pattern.match(key)
                if match:
                    block_id = match.groups()[0] if WILDCARD in patterns[0] else ""
                    block_key = (identifier + "_" + component + "_block_" + shorthand.replace(WILDCARD, block_id))
                    if block_key not in discovered_blocks:
                        discovered_blocks[block_key] = {
                            "patterns": [re.compile(p) for p in sorted((
                                re.escape(f"{prefix}.{p.replace(WILDCARD, block_id)}") + r"(?:\.|$)"
                                for p in patterns
                            ), key=lambda s: len(s.split(".")), reverse=True)],
                        }

    return discovered_blocks
