import dataclasses
import torch
import traceback
from sd_mecha.extensions.model_config import ModelConfigComponent, ModelConfig, register_model_config, StateDictKey
from sd_mecha.streaming import TensorData
from typing import Mapping, Iterable, Dict


@dataclasses.dataclass
class Component:
    identifier: str
    module: torch.nn.Module
    blocks: Mapping[str, torch.nn.Module] = dataclasses.field(default_factory=dict)


def list_blocks(block_id_prefix: str, modules: Iterable[torch.nn.Module]):
    return {
        f"{block_id_prefix}{i}": module
        for i, module in enumerate(modules)
    }


def create_config_autodetect(
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

    keys_to_merge = set()
    config_components = {}
    for module_name, module in named_modules.items():
        for component in components.values():
            if component.module is module:
                config_component = ModelConfigComponent(
                    component.identifier,
                    state_dict_keys[module_name],
                    {
                        block_id: state_dict_keys[module_name_map[block_module]]
                        for block_id, block_module in component.blocks.items()
                    }
                )
                config_components[component.identifier] = config_component
                keys_to_merge |= {x for b in config_component.blocks.values() for x in b}

    return ModelConfig(
        identifier=identifier,
        header=header,
        components=config_components,
        keys_to_merge=keys_to_merge,
        merge_space=merge_space,
        _stack_frame=stack_frame,
    )


def header_from_model(model: torch.nn.Module):
    leaf_state_dict_keys: Dict[str, Iterable[StateDictKey]] = {}
    seen_keys = set()
    state_dict_hooks = []

    def create_state_dict_hook(module_name):
        def hook(_module, state_dict, _prefix, _local_metadata):
            nonlocal leaf_state_dict_keys, seen_keys
            new_keys = set(k for k in state_dict.keys() if k not in seen_keys)
            if new_keys:
                leaf_state_dict_keys[module_name] = new_keys
                seen_keys |= new_keys
            return state_dict
        return hook

    for name, module in model.named_modules():
        if list(module.children()):
            continue
        state_dict_hooks.append(module._register_state_dict_hook(create_state_dict_hook(name)))

    header = header_from_state_dict(model.state_dict())
    for state_dict_hook in state_dict_hooks:
        state_dict_hook.remove()

    state_dict_keys: Dict[str, Iterable[StateDictKey]] = leaf_state_dict_keys.copy()
    for name, module in model.named_modules():
        if not list(module.children()):
            continue

        default = set()
        for module_name, module_keys in list(leaf_state_dict_keys.items()):
            if module_name.startswith(name):
                state_dict_keys[name] = state_dict_keys.get(name, default) | module_keys

    return header, state_dict_keys, leaf_state_dict_keys


def header_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, TensorData]:
    return {k: TensorData(v.shape, v.dtype) for k, v in state_dict.items()}
