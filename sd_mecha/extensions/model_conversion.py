import dataclasses
import functools
import pathlib
import fuzzywuzzy.process
import traceback
import torch
from sd_mecha import extensions
from typing import Mapping, Optional, Iterable, Protocol, Tuple


class ConversionCallback(Protocol):
    def __call__(self, state_dict: Mapping[str, torch.Tensor], **kwargs) -> Iterable[Tuple[str, torch.Tensor]]:
        ...


@dataclasses.dataclass
class Converter:
    __f: ConversionCallback
    identifier: str
    merge_space: type
    needs_header_conversion: bool
    strict_suffixes: bool
    key_suffixes: Optional[Tuple[str]]
    location: str

    def get_tensor(self, state_dict: Mapping[str, torch.Tensor], sd_path: pathlib.Path | str) -> Iterable[Tuple[str, torch.Tensor]]:
        return self.__f(state_dict, sd_path=sd_path)


def register_conversion(
    *,
    merge_space: type,
    identifier: Optional[str] = None,
    model_archs: str | Iterable[str] = ("__default__",),
    needs_header_conversion: bool = True,
    strict_suffixes: bool = False,
    key_suffixes: Optional[Iterable[str]] = None,
):
    stack_frame = traceback.extract_stack(None, 2)[0]
    partial = functools.partial(
        __register_conversion_impl,
        identifier=identifier,
        merge_space=merge_space,
        model_archs=model_archs,
        needs_header_conversion=needs_header_conversion,
        strict_suffixes=strict_suffixes,
        key_suffixes=key_suffixes,
        stack_frame=stack_frame,
    )
    return partial


def __register_conversion_impl(
    f: ConversionCallback,
    *,
    identifier: Optional[str],
    merge_space: type,
    model_archs: str | Iterable[str],
    needs_header_conversion: bool,
    strict_suffixes: bool,
    key_suffixes: Optional[Iterable[str]],
    stack_frame: traceback.FrameSummary,
):
    if identifier is None:
        identifier = f.__name__

    if isinstance(model_archs, str):
        model_archs = [model_archs]

    if not model_archs:
        raise ValueError(f"cannot register model type '{identifier}' without an architecture")

    if key_suffixes is not None:
        key_suffixes = tuple(key_suffixes)

    model_archs = [
        extensions.model_arch.resolve(model_arch).identifier
        if model_arch != "__default__"
        else model_arch
        for model_arch in model_archs
    ]
    if identifier in _converters_registry and (not model_archs or any(model_arch in _converters_registry[identifier] for model_arch in model_archs)):
        existing_location = _converters_registry[identifier].location
        raise ValueError(f"model type extension '{identifier}' is already registered at {existing_location}.")

    location = f"{stack_frame.filename}:{stack_frame.lineno}"
    if identifier not in _converters_registry:
        _converters_registry[identifier] = {}
    for model_arch in model_archs:
        _converters_registry[identifier][model_arch] = Converter(f, identifier, merge_space, needs_header_conversion, strict_suffixes, key_suffixes, location)

    return f


_converters_registry = {}


def resolve(identifier: str, arch_identifier: str) -> Converter:
    try:
        related_model_types = _converters_registry[identifier]
        res = related_model_types.get(arch_identifier)
        if res is None:
            res = related_model_types.get("__default__")
        if res is None:
            raise KeyError(identifier)
        return res
    except KeyError as e:
        suggestion = fuzzywuzzy.process.extractOne(str(e), _converters_registry.keys())[0]
        raise ValueError(f"unknown model type: {e}. Nearest match is '{suggestion}'")
