import abc
import functools
import inspect
import json
import pathlib
import fuzzywuzzy.process
import torch
from typing import Mapping, Protocol, Optional, Iterable, Tuple, runtime_checkable, List
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
            return OutSafetensorsDict(path, model_config.keys, mecha_recipe, buffer_size)

    @register_model_format(identifier="diffusers")
    class DiffusersDirectoryModelFormat(ModelFormat):
        def matches(self, path: pathlib.Path) -> bool:
            path = path.resolve()
            return path.is_dir()

        def get_read_dict(self, path: pathlib.Path, buffer_size: int) -> Mapping[str, torch.Tensor]:
            return DiffusersInSafetensorsDict(path, buffer_size)

        def get_write_dict(
            self,
            path: pathlib.Path,
            model_config,
            mecha_recipe: str,
            buffer_size: int,
        ) -> WriteOnlyMapping[str, torch.Tensor]:
            return DiffusersOutSafetensorsDict(path, model_config, mecha_recipe, buffer_size)

    class DiffusersInSafetensorsDict(Mapping[str, torch.Tensor]):
        def __init__(self, dir_path: pathlib.Path, minimum_buffer_size: int):
            model_index_path = dir_path / "model_index.json"
            if model_index_path.exists():
                with model_index_path.open("r") as f:
                    components = [
                        k for k, v in json.loads(f.read()).items()
                        if not k.startswith("_") and (dir_path / k).is_dir()
                    ]
            else:
                components = [
                    p.stem for p in dir_path.iterdir()
                    if p.is_dir()
                ]

            split_buffer_size = minimum_buffer_size // len(components)
            dicts = {}
            for component in components:
                component_path = find_best_safetensors_path(dir_path / component)
                if component_path is None:
                    continue

                dicts[component] = InSafetensorsDict(component_path, split_buffer_size)

            self.dicts = dicts
            self.minimum_buffer_size = minimum_buffer_size

        def __del__(self):
            self.close()

        def __getitem__(self, item: str) -> torch.Tensor:
            component, key = item.split(".", maxsplit=1)
            if key in self.dicts[component]:
                return self.dicts[component][key]

            raise KeyError(item)

        def __iter__(self):
            return iter(self.keys())

        def __len__(self) -> int:
            return sum(len(d) for d in self.dicts)

        def close(self):
            for d in self.dicts.values():
                d.close()

        def keys(self) -> Iterable[str]:
            for component, d in self.dicts.items():
                for key in d:
                    if key != "__metadata__":
                        yield f"{component}.{key}"

        def values(self) -> Iterable[torch.Tensor]:
            for key in self.keys():
                yield self[key]

        def items(self) -> Iterable[Tuple[str, torch.Tensor]]:
            for key in self.keys():
                yield key, self[key]

    def find_best_safetensors_path(dir_path: pathlib.Path) -> pathlib.Path:
        best_file = None
        for file in dir_path.iterdir():
            if file.is_file() and "model" in file.name and file.suffix == ".safetensors":
                if best_file is None or "fp16" in str(best_file.name):
                    best_file = file

        return best_file

    class DiffusersOutSafetensorsDict(WriteOnlyMapping[str, torch.Tensor]):
        def __init__(
                self,
                path: pathlib.Path,
                model_config,
                mecha_recipe: str,
                minimum_buffer_size: int,
        ):
            if path.is_file() or path.is_dir() and len(list(path.iterdir())) > 0:
                raise ValueError(f"the output path specified exists and is not empty: {path}")

            path.mkdir(exist_ok=True)
            for component in model_config.components:
                (path / component).mkdir(exist_ok=True)

            data_offset = 0
            template_header = {}
            for k, v in model_config.keys:
                template_header[k] = v.safetensors_header_value(data_offset)
                data_offset += v.get_byte_size()

            split_buffer_size = minimum_buffer_size // len(model_config.components)
            self.dicts = {
                component: OutSafetensorsDict(
                    path / component / "model.safetensors",
                    {
                        k.split(".", maxsplit=1)[1]: v
                        for k, v in template_header.items()
                        if k.startswith(component + ".")
                    },
                    mecha_recipe,
                    split_buffer_size,
                )
                for component in model_config.components
            }
            self.minimum_buffer_size = minimum_buffer_size

        def __del__(self):
            self.close()

        def __setitem__(self, item: str, value: torch.Tensor) -> None:
            component, key = item.split(".", maxsplit=1)
            self.dicts[component][key] = value

        def __len__(self) -> int:
            return sum(len(d) for d in self.dicts)

        def close(self):
            for d in self.dicts.values():
                d.close()


_register_builtin_model_formats()
