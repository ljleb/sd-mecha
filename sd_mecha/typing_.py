import abc
import inspect
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


class ClassObject(abc.ABCMeta):
    _type_cache = {}

    def __new__(mcls, name, bases, namespace, **kwargs):
        bad_methods = [
            n
            for n, value in namespace.items()
            if inspect.isfunction(value)
        ]
        if bad_methods:
            names = ", ".join(sorted(bad_methods))
            raise TypeError(
                f"{name} is a class object type and cannot define instance methods: {names}"
            )

        bases = mcls._reduce_bases(bases)
        return super().__new__(mcls, name, bases, namespace, **kwargs)

    @staticmethod
    def _reduce_bases(bases):
        bases = tuple(dict.fromkeys(bases))
        return tuple(
            base
            for base in bases
            if not any(base is not other and issubclass(other, base) for other in bases)
        )

    @staticmethod
    def _find_bound_hook(cls, name: str):
        for base in cls.__mro__:
            hook = base.__dict__.get(name)
            if hook is not None:
                return hook.__get__(None, cls)
        return None

    @classmethod
    def compose_type(mcls, bases):
        bases = mcls._reduce_bases(bases)

        if len(bases) == 1:
            return bases[0]

        cached = mcls._type_cache.get(bases)
        if cached is not None:
            return cached

        name = "__".join(base.__name__ for base in bases)
        composed_cls = mcls(name, bases, {})
        mcls._type_cache[bases] = composed_cls
        return composed_cls

    def __call__(cls, *args, **kwargs):
        hook = ClassObject._find_bound_hook(cls, "__call__")
        if hook is not None:
            return hook(*args, **kwargs)

        raise TypeError(
            f"{cls.__name__} is a class object type and should not be instantiated. "
            f"Use `{cls.__name__}` directly instead as a regular object."
        )

    def __or__(cls, other):
        hook = ClassObject._find_bound_hook(cls, "__or__")
        if hook is not None:
            return hook(other)
        return NotImplemented

    def __ror__(cls, other):
        hook = ClassObject._find_bound_hook(cls, "__ror__")
        if hook is not None:
            return hook(other)
        return NotImplemented


def subclasses(cls):
    seen = set()
    result = ()

    def walk(base):
        nonlocal result
        for sub in base.__subclasses__():
            if sub in seen:
                continue
            seen.add(sub)

            if not inspect.isabstract(sub):
                result += (sub,)

            walk(sub)

    walk(cls)
    return result
