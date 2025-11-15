import abc
import dataclasses
import inspect
import pathlib
import re
import fuzzywuzzy.process
import torch
import yaml
from collections import OrderedDict
from sd_mecha.streaming import TensorMetadata
from typing import Dict, List, Iterable, Mapping, Protocol, runtime_checkable, Optional
try:
    from yaml import CLoader as YamlLoader
except ImportError:
    from yaml import Loader as YamlLoader


StateDictKey = str


@dataclasses.dataclass
class KeyMetadata:
    shape: Optional[List[int]] | torch.Size
    dtype: Optional[str] | torch.dtype
    aliases: Iterable[str] = dataclasses.field(default_factory=tuple, metadata={"exclude": lambda p: bool(p)})
    optional: bool = dataclasses.field(default=False, metadata={"exclude": lambda p: not p})

    def __post_init__(self):
        if isinstance(self.shape, list):
            self.shape = torch.Size(self.shape)
        if isinstance(self.dtype, str):
            self.dtype = getattr(torch, self.dtype)

    def metadata(self) -> TensorMetadata:
        return TensorMetadata(self.shape, self.dtype)


@dataclasses.dataclass
class ModelComponent:
    _keys: Mapping[StateDictKey, KeyMetadata] = dataclasses.field(metadata={"serial_name": "keys"})

    def __post_init__(self):
        keys = OrderedDict()
        for k, v in self._keys.items():
            if isinstance(v, Mapping):
                keys[k] = KeyMetadata(**v)
            else:
                keys[k] = v
        self._keys = keys

    def keys(self) -> Mapping[StateDictKey, KeyMetadata]:
        return OrderedDict(
            (k, v)
            for k, v in self._keys.items()
        )

    def metadata(self) -> Mapping[StateDictKey, TensorMetadata]:
        return OrderedDict(
            (k, v.metadata())
            for k, v in self._keys.items()
        )

    def aliases(self) -> Mapping[StateDictKey, Iterable[StateDictKey]]:
        return OrderedDict(
            (k, v.aliases)
            for k, v in self._keys.items()
        )


@runtime_checkable
class ModelConfig(Protocol):
    def eq(self, other):
        other_identifier = getattr(other, "identifier", None)
        return self.identifier == other_identifier

    def __repr__(self):
        return f"<model config '{self.identifier}'>"

    @property
    @abc.abstractmethod
    def identifier(self) -> str:
        ...

    @abc.abstractmethod
    def get_architecture_identifier(self) -> str:
        ...

    @abc.abstractmethod
    def get_implementation_identifier(self) -> str:
        ...

    @abc.abstractmethod
    def components(self) -> Mapping[str, ModelComponent]:
        ...

    @abc.abstractmethod
    def keys(self) -> Mapping[StateDictKey, KeyMetadata]:
        ...

    @abc.abstractmethod
    def metadata(self) -> Mapping[StateDictKey, TensorMetadata]:
        ...

    @abc.abstractmethod
    def aliases(self) -> Mapping[StateDictKey, Iterable[StateDictKey]]:
        ...


@dataclasses.dataclass
class ModelConfigImpl(ModelConfig):
    _identifier: str = dataclasses.field(metadata={"serial_name": "identifier"})
    _components: Mapping[str, ModelComponent] = dataclasses.field(metadata={"serial_name": "components"})

    _keys_cache: Mapping[StateDictKey, KeyMetadata] = dataclasses.field(default=None, init=False, hash=False, compare=False, metadata={"exclude": True})
    _metadata_cache: Mapping[StateDictKey, TensorMetadata] = dataclasses.field(default=None, init=False, hash=False, compare=False, metadata={"exclude": True})
    _aliases_cache: Mapping[StateDictKey, Iterable[StateDictKey]] = dataclasses.field(default=None, init=False, hash=False, compare=False, metadata={"exclude": True})

    def __post_init__(self):
        if not re.fullmatch("[a-z0-9._+]+-[a-z0-9._+]+", self.identifier):
            raise ValueError(
                f"Identifier of model {self.identifier} is invalid: "
                "it must only contain lowercase alphanumerical characters, '.', '_' or '+', "
                "and must match the pattern '<architecture>-<implementation>'. "
                "An example of valid identifier is 'flux_dev-flux'"
            )

        components = OrderedDict()
        for k, v in self._components.items():
            if isinstance(v, Mapping):
                components[k] = ModelComponent(v)
            else:
                components[k] = v
        self._components = components

    @property
    def identifier(self) -> str:
        return self._identifier

    def get_architecture_identifier(self):
        return self.identifier.split("-")[0]

    def get_implementation_identifier(self):
        return self.identifier.split("-")[1]

    def components(self) -> Mapping[str, ModelComponent]:
        return self._components

    def keys(self) -> Mapping[StateDictKey, KeyMetadata]:
        if self._keys_cache is None:
            self._keys_cache = OrderedDict(
                (k, v)
                for component in self.components().values()
                for k, v in component.keys().items()
            )
        return self._keys_cache

    def metadata(self) -> Mapping[StateDictKey, TensorMetadata]:
        if self._metadata_cache is None:
            self._metadata_cache = OrderedDict(
                (k, v)
                for component in self.components().values()
                for k, v in component.metadata().items()
            )
        return self._metadata_cache

    def aliases(self) -> Mapping[StateDictKey, Iterable[StateDictKey]]:
        if self._aliases_cache is None:
            self._aliases_cache = OrderedDict(
                (k, v)
                for component in self.components().values()
                for k, v in component.aliases().items()
            )
        return self._aliases_cache


def ModelConfigImpl__init__patch(self, *args, **kwargs):
    for field in dataclasses.fields(ModelConfigImpl):
        if "serial_name" in field.metadata and field.metadata["serial_name"] in kwargs:
            kwargs[field.name] = kwargs.pop(field.metadata["serial_name"])

    ModelConfigImpl__init__(self, *args, **kwargs)


ModelConfigImpl__init__ = ModelConfigImpl.__init__
ModelConfigImpl.__init__ = ModelConfigImpl__init__patch


class LazyModelConfigBase(ModelConfig):
    def __init__(self):
        self.underlying_config = None

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        for name, value in inspect.getmembers(ModelConfig):
            if (
                (inspect.isfunction(value) or isinstance(value, property)) and
                getattr(value, "__isabstractmethod__", False) and
                name not in cls.__dict__
            ):
                # implement all remaining abstract methods as delegating to underlying_config
                setattr(cls, name, property(lambda self, name=name: resolve_lazy_model_config_attribute(self, name=name)))

    def _ensure_config(self) -> None:
        if self.underlying_config is not None:
            return

        self.underlying_config = self.create_config()

    @abc.abstractmethod
    def create_config(self) -> ModelConfig:
        raise NotImplementedError


def resolve_lazy_model_config_attribute(self: LazyModelConfigBase, name: str):
    self._ensure_config()
    attribute = getattr(self.underlying_config, name)
    if inspect.ismethod(attribute):
        method = attribute.__func__.__get__(self.underlying_config, self.underlying_config.__class__)
        return method
    return attribute


class StructuralModelConfig(ModelConfig):
    def __init__(self, keys: Mapping[StateDictKey, TensorMetadata]):
        self._keys_cache = {k: KeyMetadata(v.shape, v.dtype) for k, v in keys.items()}
        self._metadata_cache = None
        self._aliases_cache = None

    @property
    def identifier(self) -> str:
        return "structural"

    def get_architecture_identifier(self) -> str:
        return self.identifier

    def get_implementation_identifier(self) -> str:
        return ""

    def components(self) -> Mapping[str, ModelComponent]:
        return {"keys": ModelComponent(self.keys())}

    def keys(self) -> Mapping[StateDictKey, KeyMetadata]:
        return self._keys_cache

    def metadata(self) -> Mapping[StateDictKey, TensorMetadata]:
        if self._metadata_cache is None:
            self._metadata_cache = OrderedDict(
                (k, v.metadata())
                for k, v in self.keys().items()
            )
        return self._metadata_cache

    def aliases(self) -> Mapping[StateDictKey, Iterable[StateDictKey]]:
        if self._aliases_cache is None:
            self._aliases_cache = OrderedDict(
                (k, v.aliases)
                for k, v in self.keys().items()
            )
        return self._aliases_cache


class InferModelConfig(ModelConfig):
    def eq(self, other):
        raise RuntimeError("the config has not yet been inferred")

    @property
    def identifier(self) -> str:
        return "infer"

    def get_architecture_identifier(self) -> str:
        return self.identifier

    def get_implementation_identifier(self) -> str:
        return ""

    def components(self) -> Mapping[str, ModelComponent]:
        raise RuntimeError("the config has not yet been inferred")

    def keys(self) -> Mapping[StateDictKey, KeyMetadata]:
        raise RuntimeError("the config has not yet been inferred")

    def metadata(self) -> Mapping[StateDictKey, TensorMetadata]:
        raise RuntimeError("the config has not yet been inferred")

    def aliases(self) -> Mapping[StateDictKey, Iterable[StateDictKey]]:
        raise RuntimeError("the config has not yet been inferred")


INFER = InferModelConfig()


class YamlModelConfig(LazyModelConfigBase):
    def __init__(self, yaml_config_file: pathlib.Path):
        super().__init__()
        self.yaml_config_file = yaml_config_file
        self._identifier = yaml_config_file.stem

    @property
    def identifier(self) -> str:
        return self._identifier

    def create_config(self) -> ModelConfig:
        with open(self.yaml_config_file, "r") as f:
            yaml_config = f.read()

        return from_yaml(yaml_config)


_model_configs_registry_base: Dict[str, ModelConfig] = {}
_model_configs_registry_aux: Dict[str, ModelConfig] = {}


def serialize(obj):
    if isinstance(obj, ModelComponent):
        return serialize(obj.keys())
    elif dataclasses.is_dataclass(obj):
        return {
            field.metadata.get("serial_name", field.name): serialize(getattr(obj, field.name))
            for field in dataclasses.fields(obj)
            if not (
                callable(field.metadata.get("exclude")) and field.metadata["exclude"](getattr(obj, field.name))
                or field.metadata.get("exclude", False)
            )
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
    dict_config = yaml.load(yaml_config, Loader=YamlLoader)
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

    if identifier == INFER.identifier:
        return INFER

    if identifier == "structural":
        raise ValueError(
            "the 'structural' model config is not a unique object, "
            "it needs to be manually instantiated in this way: model_configs.StructuralModelConfig(...)"
        )

    suggestions = fuzzywuzzy.process.extractOne(identifier, _model_configs_registry_base.keys())
    postfix = ""
    if suggestions is not None:
        postfix = f". Nearest match is '{suggestions[0]}'"
    raise ValueError(f"unknown model implementation: {identifier}{postfix}")


def get_all() -> List[ModelConfig]:
    res = get_all_base() + get_all_aux()
    res.sort(key=lambda c: c.identifier, reverse=True)
    return res


def get_all_base() -> List[ModelConfig]:
    return list(_model_configs_registry_base.values())


def get_all_aux() -> List[ModelConfig]:
    return list(_model_configs_registry_aux.values())
