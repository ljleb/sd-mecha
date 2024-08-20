import dataclasses
import functools
import fuzzywuzzy.process
from typing import Optional, Callable


@dataclasses.dataclass
class ModelConversion:
    __f: Callable
    identifier: str


def register_conversion(
    *,
    identifier: Optional[str] = None,
):
    partial = functools.partial(
        __register_conversion_impl,
        identifier=identifier,
    )
    return partial


def __register_conversion_impl(
    f: Callable,
    *,
    identifier: Optional[str],
):
    if identifier is None:
        identifier = f.__name__

    if identifier in _converters_registry:
        raise ValueError(f"model conversion '{identifier}' is already registered.")

    _converters_registry[identifier] = ModelConversion(f, identifier)

    return f


_converters_registry = {}


def resolve(identifier: str) -> ModelConversion:
    try:
        return _converters_registry[identifier]
    except KeyError as e:
        suggestion = fuzzywuzzy.process.extractOne(str(e), _converters_registry.keys())[0]
        raise ValueError(f"unknown model type: {e}. Nearest match is '{suggestion}'")
