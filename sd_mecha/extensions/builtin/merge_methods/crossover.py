import math
import torch
from typing import Tuple
from torch import Tensor
from sd_mecha import merge_method, Parameter, Return


@merge_method
def crossover(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    alpha: Parameter(float) = 0.5,
    tilt: Parameter(float) = 0.0,
) -> Return(Tensor):
    if alpha == 0:
        return a
    if alpha == 1:
        return b
    if tilt == 1:
        return torch.lerp(a, b, alpha)

    if len(a.shape) == 0 or torch.allclose(a.half(), b.half()):
        return torch.lerp(a, b, tilt)

    shape = a.shape

    a_dft = torch.fft.rfftn(a, s=shape)
    b_dft = torch.fft.rfftn(b, s=shape)

    dft_filter = create_filter(a_dft.shape, alpha, tilt, device=a.device)

    x_dft = (1 - dft_filter)*a_dft + dft_filter*b_dft
    return torch.fft.irfftn(x_dft, s=shape)


def create_filter(shape: Tuple[int, ...] | torch.Size, alpha: float, tilt: float, device=None):
    """
    Create a crossover filter. The cut is first tilted around (0, 0), then slid along its normal until it touches the point (alpha, 1 - alpha).
    :param shape: shape of the filter
    :param alpha: the ratio between the low frequencies and high frequencies. must be in [0, 1]
      0 = all 0s, 1 = all 1s, 0s correspond to low frequencies and 1s correspond to high frequencies
    :param tilt: tilt of the filter. 0 = vertical filter, 0.5 = 45 degrees, 1 = degenerates to a weighted sum with alpha=alpha
    :param device: device of the filter
    :return:
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    # normalize tilt to the range [0, 4]
    tilt -= math.floor(tilt // 4 * 4)
    if tilt > 2:
        alpha = 1 - alpha
        alpha_inverted = True
    else:
        alpha_inverted = False

    gradients = list(reversed([
        torch.linspace(0, 1, s, device=device)
        if i == 0 or s == 1 else
        # negative frequencies are in the second half of the dimension
        torch.cat([
            torch.linspace(0, (s - 1) // 2, s - s // 2, device=device),
            torch.linspace(s // 2, 1, s // 2, device=device)
        ]) / (s // 2)
        for i, s in enumerate(reversed(shape))
    ]))

    if len(shape) > 1:
        grids = torch.meshgrid(*(g**2 for g in gradients), indexing='ij')
        mesh = (torch.stack(grids).sum(dim=0) / len(shape)).sqrt()
    else:
        mesh = gradients[0]

    if tilt < 1e-10 or abs(tilt - 4) < 1e-10:
        dft_filter = (mesh > 1 - alpha).float()
    elif abs(tilt - 2) < 1e-10:
        dft_filter = (mesh < 1 - alpha).float()
    else:
        tilt_cot = 1 / math.tan(math.pi * tilt / 2)
        if tilt <= 1 or 2 < tilt <= 3:
            dft_filter = mesh*tilt_cot + alpha*tilt_cot + alpha - tilt_cot
        else:  # 1 < tilt <= 2 or 3 < tilt
            dft_filter = mesh*tilt_cot - alpha*tilt_cot + alpha
        dft_filter = dft_filter.clip(0, 1)

    if alpha_inverted:
        dft_filter = 1 - dft_filter
    return dft_filter
