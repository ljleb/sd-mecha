from typing import TypeVar
from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return, StateDict


T = TypeVar("T")


@merge_method(is_conversion=True)
class convert_singleton:
    @staticmethod
    def map_keys(b):
        b[...] = b.keys["key"]

    def __call__(
        self,
        singleton: Parameter(StateDict[T], model_config="singleton-mecha"),
    ) -> Return(T):
        return singleton["key"]
