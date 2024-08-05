import dataclasses
import fuzzywuzzy.process
import traceback
import torch
from typing import Set, Dict, List, Iterable


WILDCARD = "*"
StateDictKey = str


@dataclasses.dataclass
class MergeConfigComponent:
    identifier: str
    key: StateDictKey
    blocks: Dict[str, Set[StateDictKey]]


def get_dataclass_caller_frame() -> traceback.FrameSummary:
    return traceback.extract_stack(None, 3)[0]


@dataclasses.dataclass
class MergeConfig:
    identifier: str
    keys: Set[StateDictKey]
    passthrough_keys: List[StateDictKey]
    components: Dict[str, MergeConfigComponent]

    _stack_frame: traceback.FrameSummary = dataclasses.field(default_factory=get_dataclass_caller_frame)

    def hyper_keys(self) -> Iterable[str]:
        for component_name, component in self.components.items():
            for block_name in component.blocks.keys():
                yield f"{self.identifier}_{component_name}_{block_name}"


@dataclasses.dataclass
class ModelComponent:
    identifier: str
    module: torch.nn.Module
    blocks: Dict[str, torch.nn.Module] = dataclasses.field(default_factory=dict)


def register_model_autodetect(
    identifier: str,
    model: torch.nn.Module,
    components: ModelComponent | Iterable[ModelComponent] = (),
    modules_to_ignore: torch.nn.Module | Iterable[torch.nn.Module] = (),
):
    stack_frame = traceback.extract_stack(None, 2)[0]

    if not isinstance(components, Iterable):
        components = [components]
    components = {c.identifier: c for c in components}

    if isinstance(modules_to_ignore, torch.nn.Module):
        modules_to_ignore = [modules_to_ignore]

    module_state_dicts = {}
    seen_keys = set()

    def create_state_dict_hook(module_name):
        def hook(_module, state_dict, _prefix, _local_metadata):
            nonlocal module_state_dicts, seen_keys
            new_keys = set(k for k in state_dict.keys() if k not in seen_keys)
            if new_keys:
                module_state_dicts[module_name] = new_keys
                seen_keys |= new_keys
            return state_dict
        return hook

    for name, module in model.named_modules():
        if list(module.children()):
            continue
        module._register_state_dict_hook(create_state_dict_hook(name))

    keys = set(model.state_dict())
    pass

    # config_components: Dict[str, MergeConfigComponent] = {}
    # passthrough_modules: Dict[str, torch.nn.Module] = {}
    # for module_key, module in model.named_modules():
    #     if module in modules_to_ignore:
    #         passthrough_modules[module_key] = module
    #
    #     for component in components:
    #         if module is component.module:
    #             config_components[component.identifier] = MergeConfigComponent(component.identifier, module_key, {
    #                 block_id: {f"{module_key}.{k}": v for k, v in block_module.state_dict()} for block_id, block_module in component.blocks
    #             })

    # config = MergeConfig(
    #     identifier=identifier,
    #     keys=keys,
    #     components=merged_components,
    #     passthrough_keys=list(passthrough_modules.keys()),
    #     _stack_frame=stack_frame,
    # )
    # register_merge_config(config)


def register_merge_config(config: MergeConfig):
    if config.identifier in _model_impls_registry:
        existing_stack_frame = _model_impls_registry[config.identifier]._stack_frame
        existing_location = f"{existing_stack_frame.filename}:{existing_stack_frame.lineno}"
        raise ValueError(f"Extension {config.identifier} is already registered at {existing_location}.")

    _model_impls_registry[config.identifier] = config


_model_impls_registry = {}


def resolve(identifier: str) -> MergeConfig:
    try:
        return _model_impls_registry[identifier]
    except KeyError:
        suggestion = fuzzywuzzy.process.extractOne(identifier, _model_impls_registry.keys())[0]
        raise ValueError(f"unknown model implementation: {identifier}. Nearest match is '{suggestion}'")


def get_all() -> List[str]:
    return list(_model_impls_registry.keys())
