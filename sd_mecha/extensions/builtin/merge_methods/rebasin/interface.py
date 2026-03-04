from torch import Tensor
from sd_mecha import merge_method, Parameter, Return, StateDict


@merge_method(is_interface=True)
def rebasin(
    a: Parameter(StateDict[Tensor]),
    ref: Parameter(StateDict[Tensor]),
    iters: Parameter(int) = 10,
) -> Return(StateDict[Tensor]):
    ...


@merge_method(is_interface=True)
def randperm(
    a: Parameter(StateDict[Tensor]),
    seed: Parameter(int) = None,
    **kwargs,
) -> Return(Tensor):
    ...
