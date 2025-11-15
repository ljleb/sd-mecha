import functools
import torch
from torch import Tensor
from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return


@merge_method
def clamp(
    a: Parameter(Tensor),
    *bounds: Parameter(Tensor),
    stiffness: Parameter(float) = 0.0,
) -> Return(Tensor):
    maximums = functools.reduce(torch.maximum, bounds)
    minimums = functools.reduce(torch.minimum, bounds)

    if stiffness:
        bounds = torch.stack(bounds)
        average = bounds.mean(dim=0)

        smallest_positive = maximums
        largest_negative = minimums

        for i, bound in enumerate(bounds):
            smallest_positive = torch.where((smallest_positive >= bound) & (bound >= average), bound, smallest_positive)
            largest_negative = torch.where((largest_negative <= bound) & (bound <= average), bound, largest_negative)

        maximums = (1-stiffness)*maximums + stiffness*smallest_positive
        minimums = (1-stiffness)*minimums + stiffness*largest_negative

    return torch.minimum(torch.maximum(a, minimums), maximums)
