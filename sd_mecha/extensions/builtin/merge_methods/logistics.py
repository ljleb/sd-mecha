import torch
from torch import Tensor
from typing import TypeVar
from sd_mecha.extensions.merge_methods import merge_method, StateDict, Parameter, Return
from sd_mecha.streaming import StateDictKeyError


T = TypeVar("T")


@merge_method
def fallback(
    a: Parameter(StateDict[T]),
    default: Parameter(StateDict[T]),
    **kwargs,
) -> Return(T):
    key = kwargs["key"]
    try:
        return a[key]
    except StateDictKeyError:
        return default[key]


@merge_method
def cast(
    a: Parameter(Tensor),
    device: Parameter(str) = None,
    dtype: Parameter(str) = None,
) -> Return(Tensor):
    to_kwargs = {}
    if device is not None:
        to_kwargs["device"] = device

    if dtype is not None:
        if dtype not in cast_dtype_map:
            raise ValueError(f"Unknown dtype {dtype}. Possible values are None, {', '.join(cast_dtype_map.keys())}")
        to_kwargs["dtype"] = cast_dtype_map[dtype]

    return a.to(**to_kwargs)


cast_dtype_map = {
    "float64": torch.float64,
    "int64": torch.int64,
    "float32": torch.float32,
    "int32": torch.int32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int16": torch.int16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "int8": torch.int8,
    "bool": torch.bool,
}
for dtype_str in ("uint8", "uint16", "uint32", "uint64"):
    if hasattr(torch, dtype_str):
        cast_dtype_map[dtype_str] = getattr(torch, dtype_str)
cast_dtype_map_reversed = {v: k for k, v in cast_dtype_map.items()}


@merge_method
def get_dtype(
    a: Parameter(Tensor),
) -> Return(str, "param"):
    return cast_dtype_map_reversed[a.dtype]


@merge_method
def get_device(
    a: Parameter(Tensor),
) -> Return(str, "param"):
    return str(a.device)


@merge_method
def pick_component(
    a: Parameter(StateDict[T]),
    component: Parameter(str, "param"),
    **kwargs,
) -> Return(T):
    if component not in a.model_config.components():
        raise ValueError(
            f'Component "{component}" does not exist in config "{a.model_config.identifier}". '
            f"Valid components: {tuple(a.model_config.components())}"
        )

    key = kwargs["key"]
    if key in a.model_config.components()[component].keys():
        return a[key]
    else:
        raise StateDictKeyError(key)


@merge_method
def omit_component(
    a: Parameter(StateDict[T]),
    component: Parameter(str, "param"),
    **kwargs,
) -> Return(T):
    if component not in a.model_config.components():
        raise ValueError(
            f'Component "{component}" does not exist in config "{a.model_config.identifier}". '
            f"Valid components: {tuple(a.model_config.components())}"
        )

    key = kwargs["key"]
    if key in a.model_config.components()[component].keys():
        raise StateDictKeyError(key)
    else:
        return a[key]


@merge_method
def stack(
    *values: Parameter(Tensor),
) -> Return(Tensor):
    return torch.stack(values)
