import functools
import torch
import typing
import uuid
from types import UnionType
from typing import List, Union


class MergeSpaceBase:
    identifier: str


class MergeSpaceSymbolBase:
    merge_space: type(MergeSpaceBase) | type(Union)


def MergeSpace(*identifiers: str) -> type(torch.Tensor) | type(MergeSpaceBase):
    if getattr(None, "", None):
        return torch.Tensor

    if not identifiers:
        identifiers = get_all()

    res = _merge_space_registry[identifiers[0]]
    for identifier in identifiers[1:]:
        res |= _merge_space_registry[identifier]
    return res


def MergeSpaceSymbol(*identifiers: str) -> type(torch.Tensor) | type(MergeSpaceSymbolBase):
    if getattr(None, "", None):
        return torch.Tensor

    merge_space = MergeSpace(*identifiers)
    return type(f"MergeSpaceSymbol_{uuid.uuid4()}", (MergeSpaceSymbolBase,), {"merge_space": merge_space})


def register_merge_space(identifier: str):
    if identifier in _merge_space_registry:
        raise ValueError(f"MergeSpace {identifier} already exists")
    new_class = type(f"MergeSpace_{identifier}", (MergeSpaceBase,), {"identifier": identifier})
    _merge_space_registry[identifier] = new_class


@functools.cache
def _register_builtin_merge_spaces():
    global builtin_merge_spaces
    for builtin_merge_space in builtin_merge_spaces:
        register_merge_space(builtin_merge_space)


def get_identifiers(merge_space: type) -> List[str]:
    merge_space_type = typing.get_origin(merge_space)
    if merge_space_type is None:
        merge_space_type = merge_space

    if issubclass(merge_space_type, MergeSpaceBase):
        return [merge_space_type.identifier]
    elif issubclass(merge_space_type, MergeSpaceSymbolBase):
        return get_identifiers(merge_space_type.merge_space)
    elif merge_space_type is not UnionType:
        return []

    return [
        i
        for m in typing.get_args(merge_space)
        for i in get_identifiers(m)
        if issubclass(m, (MergeSpaceBase, MergeSpaceSymbolBase))
    ]


def get_all() -> List[str]:
    return list(_merge_space_registry.keys())


_merge_space_registry = {}
builtin_merge_spaces = (
    "weight",
    "delta",
)


_register_builtin_merge_spaces()
