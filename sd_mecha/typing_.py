import abc
import typing
from typing import runtime_checkable, Protocol, TypeVar


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


def is_subclass(source: type, target: type):
    source = typing.get_origin(source) or source
    target = typing.get_origin(target) or target
    if isinstance(source, TypeVar):
        return False
    if isinstance(target, TypeVar):
        return any(is_subclass(source, constraint) for constraint in target.__constraints__)
    return issubclass(source, target)
