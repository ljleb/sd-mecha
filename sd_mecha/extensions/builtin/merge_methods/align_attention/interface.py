from torch import Tensor
from sd_mecha import merge_method, Parameter, Return, StateDict


@merge_method(is_interface=True)
def balance_attention_energy(
    a: Parameter(StateDict[Tensor]),
    **kwargs,
) -> Return(Tensor):
    ...


@merge_method(is_interface=True)
def align_attention(
    a: Parameter(StateDict[Tensor]),
    ref: Parameter(StateDict[Tensor]),
    permute_heads: Parameter(bool) = False,
) -> Return(StateDict[Tensor]):
    ...
