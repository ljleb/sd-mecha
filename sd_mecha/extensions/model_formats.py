import abc
import functools
import inspect
import pathlib
import fuzzywuzzy.process
import torch
from typing import Mapping, Protocol, Optional, runtime_checkable, List
from sd_mecha.streaming import InSafetensorsDict, OutSafetensorsDict
from sd_mecha.typing_ import WriteOnlyMapping


@runtime_checkable
class ModelFormat(Protocol):
    identifier: str

    @abc.abstractmethod
    def matches(self, path: pathlib.Path) -> bool:
        ...

    @abc.abstractmethod
    def get_read_dict(self, path: pathlib.Path, buffer_size: int) -> Mapping[str, torch.Tensor]:
        ...

    @abc.abstractmethod
    def get_write_dict(
        self,
        path: pathlib.Path,
        model_config,
        mecha_recipe: str,
        buffer_size: int,
    ) -> WriteOnlyMapping[str, torch.Tensor]:
        ...


def register_model_format(
    model_format: Optional[type(ModelFormat)] = None, *,
    identifier: Optional[str] = None,
):
    if model_format is None:
        return lambda model_format: __register_model_format_impl(model_format, identifier=identifier)
    return __register_model_format_impl(model_format, identifier=identifier)


def __register_model_format_impl(
    model_format: type(ModelFormat), *,
    identifier: Optional[str],
):
    if not inspect.isclass(model_format):
        raise ValueError(f"model_format must be a class, not {type(ModelFormat)}")

    if identifier is None:
        identifier = model_format.__name__

    if identifier in _model_format_registry:
        raise ValueError(f"model format '{identifier}' already exists")

    model_format = model_format()
    model_format.identifier = identifier
    _model_format_registry[identifier] = model_format


def resolve(identifier: str) -> ModelFormat:
    try:
        return _model_format_registry[identifier]
    except KeyError as e:
        suggestion = fuzzywuzzy.process.extractOne(str(e), _model_format_registry.keys())[0]
        raise ValueError(f"unknown model format: {e}. Nearest match is '{suggestion}'")


def get_all() -> List[ModelFormat]:
    return list(_model_format_registry.values())


_model_format_registry = {}


@functools.cache
def _register_builtin_model_formats():
    @register_model_format(identifier="single_file")
    class SingleFileModelFormat(ModelFormat):
        def matches(self, path: pathlib.Path) -> bool:
            path = path.resolve()
            return path.suffix == ".safetensors"

        def get_read_dict(self, path: pathlib.Path, buffer_size: int) -> Mapping[str, torch.Tensor]:
            return InSafetensorsDict(path, buffer_size)

        def get_write_dict(
            self,
            path: pathlib.Path,
            model_config,
            mecha_recipe: str,
            buffer_size: int,
        ) -> WriteOnlyMapping[str, torch.Tensor]:
            return OutSafetensorsDict(path, model_config.metadata, mecha_recipe, buffer_size)


_register_builtin_model_formats()
