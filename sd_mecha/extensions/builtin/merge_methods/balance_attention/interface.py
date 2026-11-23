from torch import Tensor
from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return, StateDict


@merge_method(is_interface=True)
def balance_attention(
    a: Parameter(Tensor),
    **kwargs,
) -> Return(Tensor):
    ...
