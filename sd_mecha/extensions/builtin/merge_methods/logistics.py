import torch
from torch import Tensor
from types import ModuleType
from typing import Literal, Optional, TypeVar
from sd_mecha.extensions.merge_methods import merge_method, StateDict, Parameter, ParameterInfo, Return, param_ref
from sd_mecha.keys_map import KeyMapBuilder
from sd_mecha.streaming import StateDictKeyError, NonFiniteStateDictKeyError


T = TypeVar("T")


@merge_method(reuse_outputs=False)
class fallback:
    @staticmethod
    def map_keys(b: KeyMapBuilder):
        for key in b.keys():
            a_inputs = b.a.keys[key] @ dict.fromkeys(["a"])
            default_inputs = b.default.keys[key] @ dict.fromkeys(["default"])
            b[key] = a_inputs & default_inputs | a_inputs | default_inputs

    def __call__(
        self,
        a: Parameter(StateDict[T]),
        default: Parameter(StateDict[T]),
        **kwargs,
    ) -> Return(T):
        (key,), _ = relation = kwargs["key_relation"]
        params = relation.meta
        for param in params:
            try:
                return locals()[param][key]
            except StateDictKeyError:
                continue

        raise StateDictKeyError(key)


cast_dtype_map = {
    "float64": torch.float64,
    "int64": torch.int64,
    "float32": torch.float32,
    "int32": torch.int32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int16": torch.int16,
    "int8": torch.int8,
    "bool": torch.bool,
}
for dtype_str in (
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
):
    if hasattr(torch, dtype_str):
        cast_dtype_map[dtype_str] = getattr(torch, dtype_str)
cast_dtype_map_reversed = {v: k for k, v in cast_dtype_map.items()}
CastDtype = Literal[tuple(cast_dtype_map)]


def _cast_device_choices():
    devices = ["cpu", "cpu:0"]
    for name, backend in sorted(vars(torch).items()):
        if name == "cpu" or not isinstance(backend, ModuleType):
            continue
        available = getattr(backend, "is_available", None)
        count_devices = getattr(backend, "device_count", None)
        if not callable(available):
            continue
        try:
            torch.device(name)
        except RuntimeError:
            continue
        if available():
            count = count_devices() if callable(count_devices) else None
            if count != 0:
                devices.append(name)
                devices.extend(f"{name}:{index}" for index in range(count or 0))
    return tuple(devices)


CastDevice = Literal[_cast_device_choices()]


@merge_method(reuse_outputs=False)
def cast(
    a: Parameter(Tensor),
    device: Parameter(Optional[CastDevice], allow_dynamic=True) = None,
    dtype: Parameter(Optional[CastDtype], allow_dynamic=True) = None,
) -> Return(Tensor):
    """Move or cast weights to a device or dtype.

    Args:
        a: The input weights.
        device: A device visible to PyTorch when sd-mecha was imported, or a recipe that produces
            its name. None keeps the input device.
        dtype: A name from cast_dtype_map, or a recipe that produces one, such as get_dtype().
            None keeps the input dtype. Available names depend on the installed PyTorch build.
    """
    to_kwargs = {}
    if device is not None:
        to_kwargs["device"] = device
    if dtype is not None:
        to_kwargs["dtype"] = cast_dtype_map[dtype]
    return a.to(**to_kwargs)


@merge_method(reuse_outputs=False)
def get_dtype(
    a: Parameter(Tensor),
) -> Return(str, "param"):
    return cast_dtype_map_reversed[a.dtype]


@merge_method(reuse_outputs=False)
def get_device(
    a: Parameter(Tensor),
) -> Return(str, "param"):
    return str(a.device)


def _component_type(info: ParameterInfo):
    return Literal[tuple(info.model_config.components())]


@merge_method(reuse_outputs=False)
def pick_component(
    a: Parameter(StateDict[T]),
    component: Parameter(param_ref("a").map(_component_type), "param", allow_dynamic=True),
    **kwargs,
) -> Return(T):
    key = kwargs["key"]
    if key in a.model_config.components()[component].keys():
        return a[key]
    else:
        raise StateDictKeyError(key)


@merge_method(reuse_outputs=False)
def omit_component(
    a: Parameter(StateDict[T]),
    component: Parameter(param_ref("a").map(_component_type), "param", allow_dynamic=True),
    **kwargs,
) -> Return(T):
    key = kwargs["key"]
    if key in a.model_config.components()[component].keys():
        raise StateDictKeyError(key)
    else:
        return a[key]


@merge_method(reuse_outputs=False)
def stack(
    *values: Parameter(Tensor),
) -> Return(Tensor):
    return torch.stack(values)


@merge_method(reuse_outputs=False)
def omit_non_finite(
    a: Parameter(Tensor),
    **kwargs,
) -> Return(Tensor):
    key = kwargs["key"]

    if not a.isfinite().all():
        raise NonFiniteStateDictKeyError(key)
    else:
        return a
