import dataclasses
import functools
import torch
from sd_mecha.extensions.merge_space import MergeSpace, get_identifiers
from sd_mecha.extensions.model_config import ModelConfigBlock, ModelConfigComponent, ModelConfig
from sd_mecha.streaming import TensorMetadata
from typing import Iterable, Dict, List, Optional


@dataclasses.dataclass
class Block:
    identifier: str
    modules_to_merge: List[torch.nn.Module] = dataclasses.field(default_factory=list)
    modules_to_copy: List[torch.nn.Module] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Component:
    identifier: str
    module: Optional[torch.nn.Module]
    blocks: List[Block] = dataclasses.field(default_factory=list)
    copy_only: bool = False


def create_config_from_module(
    identifier: str,
    merge_space: str,
    model: torch.nn.Module,
    components: Component | Iterable[Component] = (),
) -> ModelConfig:
    if not isinstance(components, Iterable):
        components = [components]

    state_dict_cache = {}

    def cache_state_dict(module: torch.nn.Module, prefix: str = ""):
        for child_name, child in module.named_children():
            cache_state_dict(child, prefix=f"{prefix}{child_name}.")

        state_dict_cache[module] = module.state_dict(prefix=prefix)
        for k, v in state_dict_cache[module].items():
            state_dict_cache[v] = {k: v}
        for k, v in module.named_parameters(prefix=prefix):
            state_dict_cache[v] = {k: v}
            state_dict_cache[v.data] = {k: v}
        for k, v in module.named_buffers(prefix=prefix):
            state_dict_cache[v] = {k: v}

        def state_dict_cached(*args, destination=None, __original_function, **kwargs):
            if destination is None and args:
                destination = args[0]

            if module not in state_dict_cache:
                state_dict_cache[module] = __original_function(*args, destination=destination, **kwargs).copy()

            res = state_dict_cache[module]
            if destination is not None:
                destination.update(res)

            return res

        module.state_dict = functools.partial(state_dict_cached, __original_function=module.state_dict)

    cache_state_dict(model)
    state_dict = model.state_dict()
    header = header_from_state_dict(state_dict)

    def get_state_dict_keys(module: torch.nn.Module | torch.Tensor) -> Iterable[str]:
        return state_dict_cache[module]

    config_components = {}
    orphan_keys = header  # filtered below
    for component in components:
        all_component_keys = {
            k: header[k]
            for k in get_state_dict_keys(component.module)
        } if component.module is not None else {
            k: header[k]
            for block in component.blocks
            for module in block.modules_to_merge + block.modules_to_copy
            for k in get_state_dict_keys(module)
        }
        component_orphan_keys = all_component_keys  # filtered below
        config_blocks = {}
        for block in component.blocks:
            block_keys_to_merge = {
                k: header[k]
                for module_to_merge in block.modules_to_merge
                for k in get_state_dict_keys(module_to_merge)
            }
            block_keys_to_copy = {
                k: header[k]
                for module_to_copy in block.modules_to_copy
                for k in get_state_dict_keys(module_to_copy)
                if k not in block_keys_to_merge
            }
            component_orphan_keys = {
                k: v
                for k, v in component_orphan_keys.items()
                if k not in block_keys_to_merge and k not in block_keys_to_copy
            }
            config_blocks[block.identifier] = ModelConfigBlock(block_keys_to_merge, block_keys_to_copy)

        config_components[component.identifier] = ModelConfigComponent(component_orphan_keys, config_blocks)
        orphan_keys = {
            k: v
            for k, v in orphan_keys.items()
            if k not in all_component_keys
        }

    # validate merge space
    merge_space = get_identifiers(MergeSpace(merge_space))[0]

    return ModelConfig(
        identifier=identifier,
        merge_space=merge_space,
        orphan_keys_to_copy=orphan_keys,
        components=config_components,
    )


def header_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, TensorMetadata]:
    return {k: TensorMetadata(v.shape, v.dtype) for k, v in state_dict.items()}
