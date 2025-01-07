import abc
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
