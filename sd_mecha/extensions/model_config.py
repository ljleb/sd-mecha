import dataclasses
import functools
import pathlib
import fuzzywuzzy.process
import torch
import yaml
from sd_mecha.streaming import TensorMetadata
from typing import Dict, List, Iterable, Mapping


StateDictKey = str


@dataclasses.dataclass
class ModelConfigBlock:
    keys_to_merge: Mapping[StateDictKey, TensorMetadata]

    def __post_init__(self):
        keys_to_merge = dict(self.keys_to_merge)
        for k, v in self.keys_to_merge.items():
            keys_to_merge[k] = v
            if isinstance(v, dict):
                keys_to_merge[k] = TensorMetadata(**v)
        self.keys_to_merge = keys_to_merge


@dataclasses.dataclass
class ModelConfigComponent:
    blocks: Mapping[str, ModelConfigBlock]

    def __post_init__(self):
        blocks = dict(self.blocks)
        for k, v in self.blocks.items():
            blocks[k] = v
            if isinstance(v, dict):
                blocks[k] = ModelConfigBlock(**v)
        self.blocks = blocks

    @property
    def keys_to_merge(self) -> Dict[StateDictKey, TensorMetadata]:
        return {
            k: v
            for block in self.blocks.values()
            for k, v in block.keys_to_merge.items()
        }


@dataclasses.dataclass
class ModelConfig:
    identifier: str
    merge_space: str
    keys_to_copy: Mapping[StateDictKey, TensorMetadata]
    components: Mapping[str, ModelConfigComponent]

    def __post_init__(self):
        components = dict(self.components)
        for k, v in self.components.items():
            components[k] = v
            if isinstance(v, dict):
                components[k] = ModelConfigComponent(**v)
        self.components = components

    def hyper_keys(self) -> Iterable[str]:
        for component_name, component in self.components.items():
            for block_name in component.blocks.keys():
                yield f"{self.identifier}_{component_name}_block_{block_name}"

    @property
    def keys(self) -> Dict[StateDictKey, TensorMetadata]:
        return self.keys_to_merge | self.keys_to_copy

    @property
    def keys_to_merge(self) -> Dict[StateDictKey, TensorMetadata]:
        return {
            k: v
            for component in self.components.values()
            for k, v in component.keys_to_merge.items()
        }


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
    old_representer = yaml.SafeDumper.yaml_representers.get(list)

    def flow_style_list_representer(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    try:
        yaml.SafeDumper.add_representer(list, flow_style_list_representer)
        return yaml.safe_dump(dict_config, sort_keys=False)
    finally:
        if old_representer is not None:
            yaml.SafeDumper.add_representer(list, old_representer)


def from_yaml(yaml_config: str) -> ModelConfig:
    dict_config = yaml.safe_load(yaml_config)
    return ModelConfig(**dict_config)


def register_model_config(config: ModelConfig):
    if config.identifier in _model_configs_registry:
        raise ValueError(f"Model {config.identifier} already exists")

    _model_configs_registry[config.identifier] = config


@functools.cache
def _register_builtin_model_configs():
    yaml_directory = pathlib.Path(__file__).parent.parent / "model_configs"
    for yaml_config_path in yaml_directory.glob("*.yaml"):
        with open(yaml_config_path, "r") as f:
            yaml_config = f.read()
        model_config = from_yaml(yaml_config)
        register_model_config(model_config)


_model_configs_registry: Dict[str, ModelConfig] = {}


def resolve(identifier: str) -> ModelConfig:
    try:
        return _model_configs_registry[identifier]
    except KeyError:
        suggestion = fuzzywuzzy.process.extractOne(identifier, _model_configs_registry.keys())[0]
        raise ValueError(f"unknown model implementation: {identifier}. Nearest match is '{suggestion}'")


def get_all() -> List[ModelConfig]:
    return list(_model_configs_registry.values())


_register_builtin_model_configs()
