import re
from torch import Tensor
from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return


@merge_method
def scale_by_mask(
    a: Parameter(Tensor),
    pattern: Parameter(str, "param"),
    multiplier: Parameter(Tensor) = 1.0,
    **kwargs,
) -> Return(Tensor):
    """Scale tensor if its state-dict key matches a regex pattern.

    This is a key-aware operation. The decorator supplies the "key" for the
    current tensor in kwargs.

    Args:
        a: Input tensor (weight or delta).
        pattern: Regex pattern to match against the key string.
        multiplier: Scaling factor applied when the key matches. Keys not
            matching the pattern are returned unchanged.

    Returns:
        Either a * multiplier (if matched) or the original tensor.
    """
    key = kwargs.get("key", "")
    try:
        matched = re.search(pattern, key) is not None
    except re.error as exc:
        raise ValueError(f"Invalid regex pattern: {pattern!r}") from exc
    return a * multiplier if matched else a
  
