import dataclasses
import pathlib
import re
import fuzzywuzzy.process
import torch
import yaml
from collections import OrderedDict
from sd_mecha.streaming import TensorMetadata
from typing import Dict, List, Iterable, Mapping, Protocol, runtime_checkable


StateDictKey = str


@dataclasses.dataclass
class ModelConfigBlock:
    keys_to_merge: Mapping[StateDictKey, TensorMetadata]
    keys_to_copy: Mapping[StateDictKey, TensorMetadata]

    def __post_init__(self):
        keys_to_merge = OrderedDict(self.keys_to_merge)
        for k, v in self.keys_to_merge.items():
            keys_to_merge[k] = v
            if isinstance(v, dict):
                keys_to_merge[k] = TensorMetadata(**v)
        self.keys_to_merge = keys_to_merge

        keys_to_copy = OrderedDict(self.keys_to_copy)
        for k, v in self.keys_to_copy.items():
            keys_to_copy[k] = v
            if isinstance(v, dict):
                keys_to_copy[k] = TensorMetadata(**v)
        self.keys_to_copy = keys_to_copy

    def compute_keys(self) -> Dict[StateDictKey, TensorMetadata]:
        return OrderedDict(
            **self.keys_to_merge,
            **self.keys_to_copy,
        )


@dataclasses.dataclass
class ModelConfigComponent:
    orphan_keys_to_copy: Mapping[StateDictKey, TensorMetadata]
    blocks: Mapping[str, ModelConfigBlock]

    def __post_init__(self):
        orphan_keys_to_copy = OrderedDict(self.orphan_keys_to_copy)
        for k, v in self.orphan_keys_to_copy.items():
            orphan_keys_to_copy[k] = v
            if isinstance(v, dict):
                orphan_keys_to_copy[k] = TensorMetadata(**v)
        self.orphan_keys_to_copy = orphan_keys_to_copy

        blocks = OrderedDict(self.blocks)
        for k, v in self.blocks.items():
            blocks[k] = v
            if isinstance(v, dict):
                blocks[k] = ModelConfigBlock(**v)
        self.blocks = blocks

    def compute_keys(self) -> Dict[StateDictKey, TensorMetadata]:
        return OrderedDict(
            **self.compute_keys_to_merge(),
            **self.compute_keys_to_copy(),
        )

    def compute_keys_to_merge(self) -> Dict[StateDictKey, TensorMetadata]:
        return OrderedDict(
            (k, v)
            for block in self.blocks.values()
            for k, v in block.keys_to_merge.items()
        )

    def compute_keys_to_copy(self) -> Dict[StateDictKey, TensorMetadata]:
        return self.orphan_keys_to_copy | OrderedDict(
            (k, v)
            for block in self.blocks.values()
            for k, v in block.keys_to_copy.items()
        )


@runtime_checkable
class ModelConfig(Protocol):
    identifier: str
    orphan_keys_to_copy: Mapping[StateDictKey, TensorMetadata]
    components: Mapping[str, ModelConfigComponent]

    def __class_getitem__(cls, item) -> type:
        config = resolve(item)
        return type(f"{config.identifier}ModelConfigTag", (ModelConfigTag,), {
            "config": config,
        })

    def get_architecture_identifier(self) -> str:
        pass

    def get_implementation_identifier(self) -> str:
        pass

    def hyper_keys(self) -> Iterable[str]:
        pass

    def get_hyper_block_key(self, component_identifier, block_identifier) -> str:
        pass

    def get_hyper_default_key(self, component_identifier) -> str:
        pass

    def compute_keys(self) -> Dict[StateDictKey, TensorMetadata]:
        pass

    def compute_keys_to_merge(self) -> Dict[StateDictKey, TensorMetadata]:
        pass

    def compute_keys_to_copy(self) -> Dict[StateDictKey, TensorMetadata]:
        pass


@dataclasses.dataclass
class ModelConfigImpl(ModelConfig):
    identifier: str
    orphan_keys_to_copy: Mapping[StateDictKey, TensorMetadata]
    components: Mapping[str, ModelConfigComponent]

    def __post_init__(self):
        if not re.fullmatch("[a-z0-9_+]+-[a-z0-9_+]+", self.identifier):
            raise ValueError(
                f"Identifier of model {self.identifier} is invalid: "
                "it must only contain lowercase alphanumerical characters or '+' or '_', "
                "and must match the pattern '<architecture>-<implementation>'. "
                "An example of valid identifier is 'flux_dev-flux'"
            )

        orphan_keys_to_copy = OrderedDict(self.orphan_keys_to_copy)
        for k, v in self.orphan_keys_to_copy.items():
            orphan_keys_to_copy[k] = v
            if isinstance(v, dict):
                orphan_keys_to_copy[k] = TensorMetadata(**v)
        self.orphan_keys_to_copy = orphan_keys_to_copy

        components = OrderedDict(self.components)
        for k, v in self.components.items():
            components[k] = v
            if isinstance(v, dict):
                components[k] = ModelConfigComponent(**v)
        self.components = components

    def get_architecture_identifier(self):
        return self.identifier.split("-")[0]

    def get_implementation_identifier(self):
        return self.identifier.split("-")[1]

    def hyper_keys(self) -> Iterable[str]:
        for component_name, component in self.components.items():
            for block_name in component.blocks.keys():
                yield self.get_hyper_block_key(component_name, block_name)

    def get_hyper_block_key(self, component_identifier, block_identifier):
        if component_identifier not in self.components:
            raise ValueError(f"no such component: {component_identifier}")
        if block_identifier not in self.components[component_identifier].blocks:
            raise ValueError(f"no such block in component {component_identifier}: {block_identifier}")

        return f"{self.identifier}-{component_identifier}_block_{block_identifier}"

    def get_hyper_default_key(self, component_identifier):
        if component_identifier not in self.components:
            raise ValueError(f"no such component: {component_identifier}")

        return f"{self.identifier}-{component_identifier}_default"

    def compute_keys(self) -> Dict[StateDictKey, TensorMetadata]:
        return OrderedDict(
            **self.compute_keys_to_merge(),
            **self.compute_keys_to_copy(),
        )

    def compute_keys_to_merge(self) -> Dict[StateDictKey, TensorMetadata]:
        return OrderedDict(
            (k, v)
            for component in self.components.values()
            for k, v in component.compute_keys_to_merge().items()
        )

    def compute_keys_to_copy(self) -> Dict[StateDictKey, TensorMetadata]:
        return self.orphan_keys_to_copy | OrderedDict(
            (k, v)
            for component in self.components.values()
            for k, v in component.compute_keys_to_copy().items()
        )


class LazyModelConfig(ModelConfig):
    def __init__(self, yaml_config_file: pathlib.Path):
        self.yaml_config_file = yaml_config_file
        self.underlying_config = None
        self.identifier = yaml_config_file.stem

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]

        self._ensure_config()
        return getattr(self.underlying_config, item)

    def _ensure_config(self):
        if self.underlying_config is not None:
            return

        with open(self.yaml_config_file, "r") as f:
            yaml_config = f.read()

        self.underlying_config = from_yaml(yaml_config)


_model_configs_registry_base: Dict[str, ModelConfig] = {}
_model_configs_registry_aux: Dict[str, ModelConfig] = {}


class ModelConfigTag:
    config: ModelConfig


def serialize(obj):
    if dataclasses.is_dataclass(obj):
        return {
            field.name: serialize(getattr(obj, field.name))
            for field in dataclasses.fields(obj)
            if not field.metadata.get("exclude", False)
        }
    elif isinstance(obj, (str, int, float, type(None))):
        return obj
    elif isinstance(obj, torch.dtype):
        return str(obj).split(".")[1]
    elif isinstance(obj, torch.Size):
        return list(obj)
    elif isinstance(obj, Mapping):
        return {str(k): serialize(v) for k, v in obj.items()}
    elif isinstance(obj, Iterable) and not isinstance(obj, bytes):
        return [serialize(v) for v in obj]
    else:
        raise ValueError(f"Cannot serialize {obj!r}")


def to_yaml(model_config: ModelConfig) -> str:
    dict_config = serialize(model_config)
    old_representers = yaml.SafeDumper.yaml_representers.copy()

    def flow_style_list_representer(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    def flow_style_dict_representer(dumper, data):
        if "shape" in data and "dtype" in data:
            return dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=True)
        else:
            return dumper.represent_mapping('tag:yaml.org,2002:map', data)

    try:
        yaml.SafeDumper.add_representer(list, flow_style_list_representer)
        yaml.SafeDumper.add_representer(dict, flow_style_dict_representer)
        return yaml.safe_dump(dict_config, width=2**64, sort_keys=False)
    finally:
        yaml.SafeDumper.yaml_representers.clear()
        yaml.SafeDumper.yaml_representers.update(old_representers)


def from_yaml(yaml_config: str) -> ModelConfig:
    dict_config = yaml.safe_load(yaml_config)
    return ModelConfigImpl(**dict_config)


def register(config: ModelConfig):
    if config.identifier in _model_configs_registry_base or config.identifier in _model_configs_registry_aux:
        raise ValueError(f"Model {config.identifier} already exists")

    _model_configs_registry_base[config.identifier] = config


def register_aux(config: ModelConfig):
    if config.identifier in _model_configs_registry_base or config.identifier in _model_configs_registry_aux:
        raise ValueError(f"Model {config.identifier} already exists")

    _model_configs_registry_aux[config.identifier] = config


def resolve(identifier: str) -> ModelConfig:
    try:
        return _model_configs_registry_base[identifier]
    except KeyError:
        pass

    try:
        return _model_configs_registry_aux[identifier]
    except KeyError:
        pass

    suggestions = fuzzywuzzy.process.extractOne(identifier, _model_configs_registry_base.keys())
    postfix = ""
    if suggestions is not None:
        postfix = f". Nearest match is '{suggestions[0]}'"
    raise ValueError(f"unknown model implementation: {identifier}{postfix}")


def get_all() -> List[ModelConfig]:
    return get_all_base() + get_all_aux()


def get_all_base() -> List[ModelConfig]:
    return list(_model_configs_registry_base.values())


def get_all_aux() -> List[ModelConfig]:
    return list(_model_configs_registry_aux.values())
