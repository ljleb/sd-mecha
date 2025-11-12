import math
import torch
from torch import Tensor
from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return

@merge_method
def scale_to_match_rms(
    a: Parameter(Tensor),
    ref: Parameter(Tensor),
    target_ratio: Parameter(Tensor) = 1.0,
    clip_multiple: Parameter(float) = 2.5,
    eps: Parameter(float) = 1e-12
) -> Return(Tensor):
    """Rescale tensor a to match target_ratio * RMS(ref), with clipping.

    For a "delta" tensor a and reference weight ref, this rescales a so its
    per-tensor RMS is close to target_ratio * RMS(ref), while capping the scale
    factor to [1/clip_multiple, clip_multiple]. This curbs extreme changes.

    Args:
        a: Input tensor (usually a delta).
        ref: Reference tensor (usually the base weight for the same key).
        target_ratio: Desired ratio of RMS(a) to RMS(ref) after rescaling.
        clip_multiple: Limits the scale factor s to [1/clip_multiple, clip_multiple].
        eps: Numerical stability epsilon.

    Returns:
        Rescaled tensor.
    """
    # Compute RMS; handle empty/scalar shapes
    if a.numel() == 0:
        return a

    rms_a = torch.sqrt(torch.clamp((a.float() ** 2).mean(), min=0.0) + eps)
    rms_ref = torch.sqrt(torch.clamp((ref.float() ** 2).mean(), min=0.0) + eps)

    # RMS(s*a) ~= target_ratio * RMS(ref)
    s = (rms_ref * target_ratio) / rms_a
    if not math.isfinite(s.item()):
        return a

    s = s.clamp(min=1.0 / max(clip_multiple, 1e-6), max=max(clip_multiple, 1.0))
    s = s.to(dtype=a.dtype, device=a.device)
    return a * s
