import dataclasses
import functools
import pathlib
import re
import fuzzywuzzy.process
import torch
import yaml
from sd_mecha.streaming import TensorMetadata
from typing import Dict, List, Iterable, Mapping, Sized, Optional

StateDictKey = str


@dataclasses.dataclass
class ModelConfigBlock:
    keys_to_merge: Mapping[StateDictKey, TensorMetadata]
    keys_to_copy: Mapping[StateDictKey, TensorMetadata]

    def __post_init__(self):
        keys_to_merge = dict(self.keys_to_merge)
        for k, v in self.keys_to_merge.items():
            keys_to_merge[k] = v
            if isinstance(v, dict):
                keys_to_merge[k] = TensorMetadata(**v)
        self.keys_to_merge = keys_to_merge

        keys_to_copy = dict(self.keys_to_copy)
        for k, v in self.keys_to_copy.items():
            keys_to_copy[k] = v
            if isinstance(v, dict):
                keys_to_copy[k] = TensorMetadata(**v)
        self.keys_to_copy = keys_to_copy

    @property
    def keys(self) -> Dict[StateDictKey, TensorMetadata]:
        return {
            **self.keys_to_merge,
            **self.keys_to_copy,
        }


@dataclasses.dataclass
class ModelConfigComponent:
    orphan_keys_to_copy: Mapping[StateDictKey, TensorMetadata]
    blocks: Mapping[str, ModelConfigBlock]

    def __post_init__(self):
        orphan_keys_to_copy = dict(self.orphan_keys_to_copy)
        for k, v in self.orphan_keys_to_copy.items():
            orphan_keys_to_copy[k] = v
            if isinstance(v, dict):
                orphan_keys_to_copy[k] = TensorMetadata(**v)
        self.orphan_keys_to_copy = orphan_keys_to_copy

        blocks = dict(self.blocks)
        for k, v in self.blocks.items():
            blocks[k] = v
            if isinstance(v, dict):
                blocks[k] = ModelConfigBlock(**v)
        self.blocks = blocks

    @property
    def keys(self) -> Dict[StateDictKey, TensorMetadata]:
        return {
            **self.keys_to_merge,
            **self.keys_to_copy,
        }

    @property
    def keys_to_merge(self) -> Dict[StateDictKey, TensorMetadata]:
        return {
            k: v
            for block in self.blocks.values()
            for k, v in block.keys_to_merge.items()
        }

    @property
    def keys_to_copy(self) -> Dict[StateDictKey, TensorMetadata]:
        return self.orphan_keys_to_copy | {
            k: v
            for block in self.blocks.values()
            for k, v in block.keys_to_copy.items()
        }


@dataclasses.dataclass
class ModelConfig:
    identifier: str
    merge_space: str
    orphan_keys_to_copy: Mapping[StateDictKey, TensorMetadata]
    components: Mapping[str, ModelConfigComponent]

    def __post_init__(self):
        orphan_keys_to_copy = dict(self.orphan_keys_to_copy)
        for k, v in self.orphan_keys_to_copy.items():
            orphan_keys_to_copy[k] = v
            if isinstance(v, dict):
                orphan_keys_to_copy[k] = TensorMetadata(**v)
        self.orphan_keys_to_copy = orphan_keys_to_copy

        components = dict(self.components)
        for k, v in self.components.items():
            components[k] = v
            if isinstance(v, dict):
                components[k] = ModelConfigComponent(**v)
        self.components = components

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

    @property
    def keys(self) -> Dict[StateDictKey, TensorMetadata]:
        return {
            **self.keys_to_merge,
            **self.keys_to_copy,
        }

    @property
    def keys_to_merge(self) -> Dict[StateDictKey, TensorMetadata]:
        return {
            k: v
            for component in self.components.values()
            for k, v in component.keys_to_merge.items()
        }

    @property
    def keys_to_copy(self) -> Dict[StateDictKey, TensorMetadata]:
        return self.orphan_keys_to_copy | {
            k: v
            for component in self.components.values()
            for k, v in component.keys_to_copy.items()
        }


_model_configs_registry: Dict[str, ModelConfig] = {}


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
    return ModelConfig(**dict_config)


def register_model_config(config: ModelConfig):
    if config.identifier in _model_configs_registry:
        raise ValueError(f"Model {config.identifier} already exists")
    if not re.fullmatch("[a-z0-9_+]+-[a-z0-9_+]+", config.identifier):
        raise ValueError(
            f"Identifier of model {config.identifier} is invalid: "
            "it must only contain lowercase alphanumerical characters or '+' or '_', "
            "and must match the pattern '<architecture>-<implementation>'. "
            "An example of valid identifier is 'flux_dev-flux'"
        )

    _model_configs_registry[config.identifier] = config


# run only once
@functools.cache
def _register_builtin_model_configs():
    yaml_directory = pathlib.Path(__file__).parent.parent / "model_configs"
    for yaml_config_path in yaml_directory.glob("*.yaml"):
        register_model_config(LazyModelConfig(yaml_config_path))


class LazyModelConfig(ModelConfig):
    def __init__(self, yaml_config_file: pathlib.Path):
        self.yaml_config_file = yaml_config_file
        self.underlying_config = None

    @property
    def identifier(self) -> str:
        return self.yaml_config_file.stem

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


def resolve(identifier: str) -> ModelConfig:
    try:
        return _model_configs_registry[identifier]
    except KeyError:
        pass

    config_ids = []
    for config in get_all():
        if config.identifier == identifier:
            return config
        else:
            config_ids.append(config.identifier)

    suggestion = fuzzywuzzy.process.extractOne(identifier, config_ids)[0]
    raise ValueError(f"unknown model implementation: {identifier}. Nearest match is '{suggestion}'")


def get_all() -> List[ModelConfig]:
    return get_all_base() + get_all_lycoris()


def get_all_base() -> List[ModelConfig]:
    return list(_model_configs_registry.values())


def get_all_lycoris() -> List[ModelConfig]:
    base_configs = get_all_base()
    lycoris_configs = []
    for base_config in base_configs:
        lycoris_configs.append(to_lycoris_config(base_config, "lora"))

    return lycoris_configs


def to_lycoris_config(base_config: ModelConfig, algorithms: Optional[str | Iterable[str]] = None) -> ModelConfig:
    if algorithms is None:
        algorithms = "lora"  # all algos
    if isinstance(algorithms, str):
        algorithms = [algorithms]
    if not isinstance(algorithms, Sized):
        algorithms = list(algorithms)

    if "lora" not in algorithms or len(algorithms) != 1:
        raise ValueError(f"unknown lycoris algorithms {algorithms}")

    return ModelConfig(
        identifier=f"{base_config.identifier}_{'_'.join(algorithms)}",
        merge_space="delta",
        orphan_keys_to_copy=_to_lycoris_keys(base_config.orphan_keys_to_copy, algorithms),
        components={
            component_id: dataclasses.replace(
                component,
                blocks={
                    block_id: dataclasses.replace(
                        block,
                        keys_to_merge=_to_lycoris_keys(block.keys_to_merge, algorithms),
                        keys_to_copy=_to_lycoris_keys(block.keys_to_copy, algorithms),
                    )
                    for block_id, block in component.blocks.items()
                },
            )
            for component_id, component in base_config.components.items()
        },
    )


def _to_lycoris_keys(
    keys: Mapping[StateDictKey, TensorMetadata],
    algorithms: Iterable[str],
    prefix: str = "lycoris",
) -> Dict[StateDictKey, TensorMetadata]:
    lycoris_keys = {}

    # ignore `algorithms` for now, assume lora only
    for key, meta in keys.items():
        if key.endswith("bias"):
            continue

        key = key.split('.')
        if key[-1] == "weight":
            key = key[:-1]
        key = "_".join(key)

        for suffix in ("lora_up.weight", "lora_down.weight", "alpha"):
            lycoris_key = f"{prefix}_{key}.{suffix}"
            lycoris_keys[lycoris_key] = dataclasses.replace(meta, shape=None)

    return lycoris_keys


_register_builtin_model_configs()
