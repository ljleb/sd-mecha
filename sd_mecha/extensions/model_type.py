import dataclasses
import functools
import fuzzywuzzy.process
import traceback
import torch
from sd_mecha.merge_space import MergeSpace
from sd_mecha.extensions.model_version import ModelVersion
from sd_mecha.streaming import DTYPE_REVERSE_MAPPING, DTYPE_MAPPING
from torch._subclasses.fake_tensor import FakeTensorMode
from typing import Callable, Mapping, Optional, List


ModelTypeCallback = Callable[[Mapping[str, torch.Tensor], str], torch.Tensor]


@dataclasses.dataclass
class ModelType:
    __f: ModelTypeCallback
    identifier: str
    merge_space: MergeSpace
    location: str

    def get(self, state_dict: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
        return self.__f(state_dict, key)

    def convert_header(self, header: Mapping[str, Mapping[str, str | List[int]]], model_version: ModelVersion):
        fake_mode = FakeTensorMode()

        def _create_fake_tensor(*shape, dtype):
            return fake_mode.from_tensor(torch.empty(shape, dtype=dtype))

        with fake_mode:
            fake_state_dict = {
                k: _create_fake_tensor(*h["shape"], dtype=DTYPE_MAPPING[h["dtype"]][0])
                for k, h in header.items()
                if k != "__metadata__"
            }
            converted_state_dict = {}
            for k in model_version.keys:
                try:
                    converted_state_dict[k] = self.get(fake_state_dict, k)
                except KeyError:
                    continue

            data_offsets = 0
            converted_header = {
                k: {
                    "dtype": DTYPE_REVERSE_MAPPING[t.dtype][0],
                    "shape": list(t.shape),
                    "data_offsets": [data_offsets, (data_offsets := data_offsets + t.numel() * t.element_size())],
                }
                for k, t in converted_state_dict.items()
            }
        return converted_header


def register_model_type(
    f: Optional[ModelTypeCallback] = None,
    *,
    identifier: Optional[str] = None,
    merge_space: MergeSpace = MergeSpace.DELTA,
):
    stack_frame = traceback.extract_stack(None, 2)[0]
    partial = functools.partial(
        __register_model_type_impl,
        identifier=identifier,
        merge_space=merge_space,
        stack_frame=stack_frame,
    )
    if f is None:
        return partial
    return partial(f)


def __register_model_type_impl(
    f: ModelTypeCallback,
    *,
    identifier: Optional[str],
    merge_space: MergeSpace,
    stack_frame: traceback.FrameSummary,
):
    if identifier is None:
        identifier = f.__name__

    if identifier in _model_types_registry:
        existing_location = _model_types_registry[identifier].location
        raise ValueError(f"Extension '{identifier}' is already registered at {existing_location}.")

    location = f"{stack_frame.filename}:{stack_frame.lineno}"
    _model_types_registry[identifier] = ModelType(f, identifier, merge_space, location)


_model_types_registry = {}


def resolve(identifier: str) -> ModelType:
    try:
        return _model_types_registry[identifier]
    except KeyError as e:
        suggestion = fuzzywuzzy.process.extractOne(str(e), _model_types_registry.keys())[0]
        raise ValueError(f"unknown merge method: {e}. Nearest match is '{suggestion}'")
