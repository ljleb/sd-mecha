import typing
from typing import List
import uuid


class MergeSpaceBase:
    identifier: str


class MergeSpaceSymbolBase:
    merge_space: type(MergeSpaceBase)


def MergeSpace(*identifiers: str) -> type(MergeSpaceBase):
    if not identifiers:
        identifiers = list(_merge_space_registry.keys())

    res = _merge_space_registry[identifiers[0]]
    for identifier in identifiers[1:]:
        res |= _merge_space_registry[identifier]
    return res


def MergeSpaceSymbol(*identifiers: str) -> type(MergeSpaceSymbolBase):
    merge_space = MergeSpace(*identifiers)
    return type(f"MergeSpaceSymbol_{uuid.uuid4()}", (MergeSpaceSymbolBase,), {"merge_space": merge_space})


def register_merge_space(identifier: str):
    if identifier in _merge_space_registry:
        raise ValueError(f"MergeSpace {identifier} already exists")
    new_class = type(f"MergeSpace_{identifier}", (MergeSpaceBase,), {"identifier": identifier})
    _merge_space_registry[identifier] = new_class


def get_identifiers(merge_space: type) -> List[str]:
    return [
        m.identifier
        if issubclass(merge_space, MergeSpaceBase) else
        m.merge_space.identifier
        for m in typing.get_args(merge_space)
        if issubclass(m, (MergeSpaceBase, MergeSpaceSymbolBase))
    ]


_merge_space_registry = {}


def register_builtin_merge_spaces():
    global builtin_merge_spaces
    for builtin_merge_space in builtin_merge_spaces:
        register_merge_space(builtin_merge_space)


builtin_merge_spaces = (
    "weight",
    "delta",
)
