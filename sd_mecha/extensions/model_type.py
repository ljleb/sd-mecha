import dataclasses
import functools
import fuzzywuzzy.process
import traceback
import torch
from sd_mecha import extensions
from sd_mecha.merge_space import MergeSpace
from sd_mecha.extensions.model_arch import ModelArch
from sd_mecha.streaming import DTYPE_REVERSE_MAPPING, DTYPE_MAPPING, InSafetensorsDict
from torch._subclasses.fake_tensor import FakeTensorMode
from typing import Callable, Mapping, Optional, List, Iterable


ModelTypeCallback = Callable[[Mapping[str, torch.Tensor], str], torch.Tensor]


@dataclasses.dataclass
class ModelType:
    __f: ModelTypeCallback
    identifier: str
    merge_space: MergeSpace
    needs_header_conversion: bool
    location: str

    def get_tensor(self, state_dict: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
        return self.__f(state_dict, key)

    def convert_header(self, state_dict: InSafetensorsDict | Mapping[str, torch.Tensor], model_arch: ModelArch):
        def _create_header(fake_state_dict):
            data_offsets = 0
            return {
                k: {
                    "dtype": DTYPE_REVERSE_MAPPING[t.dtype][0],
                    "shape": list(t.shape),
                    "data_offsets": [data_offsets, (data_offsets := data_offsets + t.numel() * t.element_size())],
                }
                for k, t in fake_state_dict.items()
            }

        if isinstance(state_dict, InSafetensorsDict):
            if not self.needs_header_conversion:
                return {
                    k: v
                    for k, v in state_dict.header.items()
                    if k != "__metadata__"
                }

            def _create_fake_tensor(*shape, dtype):
                return torch.empty(shape, dtype=dtype)

            with FakeTensorMode():
                fake_state_dict = {
                    k: _create_fake_tensor(*h["shape"], dtype=DTYPE_MAPPING[h["dtype"]][0])
                    for k, h in state_dict.header.items()
                    if k != "__metadata__"
                }
                converted_state_dict = {}
                for k in model_arch.keys:
                    try:
                        converted_state_dict[k] = self.get_tensor(fake_state_dict, k)
                    except KeyError:
                        continue

                return _create_header(converted_state_dict)
        else:
            return _create_header(state_dict)


def register_model_type(
    *,
    merge_space: MergeSpace,
    identifier: Optional[str] = None,
    model_archs: str | Iterable[str] = ("__default__",),
    needs_header_conversion: bool = True,
):
    stack_frame = traceback.extract_stack(None, 2)[0]
    partial = functools.partial(
        __register_model_type_impl,
        identifier=identifier,
        merge_space=merge_space,
        model_archs=model_archs,
        needs_header_conversion=needs_header_conversion,
        stack_frame=stack_frame,
    )
    return partial


def __register_model_type_impl(
    f: ModelTypeCallback,
    *,
    identifier: Optional[str],
    merge_space: MergeSpace,
    model_archs: str | Iterable[str],
    needs_header_conversion: bool,
    stack_frame: traceback.FrameSummary,
):
    if identifier is None:
        identifier = f.__name__

    if isinstance(model_archs, str):
        model_archs = [model_archs]

    if not model_archs:
        raise ValueError(f"cannot register model type '{identifier}' without an architecture")

    model_archs = [
        extensions.model_arch.resolve(model_arch).identifier
        if model_arch != "__default__"
        else model_arch
        for model_arch in model_archs
    ]
    if identifier in _model_types_registry and (not model_archs or any(model_arch in _model_types_registry[identifier] for model_arch in model_archs)):
        existing_location = _model_types_registry[identifier].location
        raise ValueError(f"model type extension '{identifier}' is already registered at {existing_location}.")

    location = f"{stack_frame.filename}:{stack_frame.lineno}"
    if identifier not in _model_types_registry:
        _model_types_registry[identifier] = {}
    for model_arch in model_archs:
        _model_types_registry[identifier][model_arch] = ModelType(f, identifier, merge_space, needs_header_conversion, location)

    return f


_model_types_registry = {}


def resolve(identifier: str, arch_identifier: str) -> ModelType:
    try:
        related_model_types = _model_types_registry[identifier]
        res = related_model_types.get(arch_identifier)
        if res is None:
            res = related_model_types.get("__default__")
        if res is None:
            raise KeyError(identifier)
        return res
    except KeyError as e:
        suggestion = fuzzywuzzy.process.extractOne(str(e), _model_types_registry.keys())[0]
        raise ValueError(f"unknown model type: {e}. Nearest match is '{suggestion}'")
