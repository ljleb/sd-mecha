import abc
import typing
from types import UnionType
from typing import runtime_checkable, Protocol, TypeVar, Any


K = TypeVar("K")
V = TypeVar("V")


@runtime_checkable
class WriteOnlyMapping(Protocol[K, V]):
    @abc.abstractmethod
    def __setitem__(self, key: K, value: V) -> None:
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        ...


def is_subclass(source: type | UnionType, target: type | UnionType):
    source_origin = typing.get_origin(source) or source
    target_origin = typing.get_origin(target) or target
    if isinstance(source_origin, TypeVar):
        return source == target
    if isinstance(target_origin, TypeVar):
        return any(is_subclass(source_origin, constraint) for constraint in target_origin.__constraints__)
    if is_union(source_origin):
        return all(is_subclass(arg, target) for arg in typing.get_args(source))
    return issubclass(source_origin, target)


def is_instance(source: Any, target: type | UnionType):
    target_origin = typing.get_origin(target) or target
    if isinstance(target_origin, TypeVar):
        return any(isinstance(source, constraint) for constraint in target_origin.__constraints__)
    if is_union(target_origin):
        return any(isinstance(source, typing.get_origin(arg) or arg) for arg in typing.get_args(target))
    return isinstance(source, target)


def is_union(typ3) -> bool:
    if typ3 is typing.Union:
        return True

    return issubclass(typ3, UnionType)
