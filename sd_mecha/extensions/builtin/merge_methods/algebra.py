from torch import Tensor
from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return


@merge_method
def scale(
    a: Parameter(Tensor),
    factor: Parameter(Tensor) = 1.0,
) -> Return(Tensor):
    """Multiply a tensor by a scalar factor.

    Args:
        a: Input tensor (weight or delta).
        factor: Scaling multiplier. Can be a scalar tensor or Python float.

    Returns:
        Scaled tensor a * factor.
    """
    return a * factor


@merge_method
def add(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
) -> Return(Tensor):
    """Elementwise addition of two tensors.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        Elementwise sum a + b.
    """
    return a + b
