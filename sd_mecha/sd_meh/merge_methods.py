import functools
import math
import operator
import torch
from torch import Tensor
from typing import Tuple, TypeVar, Dict, Optional
from sd_mecha.sd_meh.extensions import merge_methods, MergeSpace, LiftFlag


EPSILON = 1e-10


SharedSpace = TypeVar("SharedSpace", bound=LiftFlag[MergeSpace.MODEL | MergeSpace.DELTA])


@merge_methods.register()
def weighted_sum(
    a: Tensor | SharedSpace,
    b: Tensor | SharedSpace,
    alpha: float,
) -> Tensor | SharedSpace:
    return (1 - alpha) * a + alpha * b


@merge_methods.register()
def add(
    a: Tensor | SharedSpace,
    b: Tensor | LiftFlag[MergeSpace.DELTA],
    alpha: float,
) -> Tensor | SharedSpace:
    return a + alpha * b


@merge_methods.register()
def subtract(
    a: Tensor | LiftFlag[MergeSpace.MODEL],
    b: Tensor | LiftFlag[MergeSpace.MODEL],
) -> Tensor | LiftFlag[MergeSpace.DELTA]:
    return a - b


@merge_methods.register()
def add_perpendicular(
    a: Tensor | LiftFlag[MergeSpace.DELTA],
    b: Tensor | LiftFlag[MergeSpace.DELTA],
    alpha: float,
) -> Tensor | LiftFlag[MergeSpace.DELTA]:
    iters = 1  # 200
    for i in range(iters):
        if i == 0:
            adjusted_alpha = i / iters
        else:
            adjusted_alpha = 1 - (1 - (1 + i) * alpha / iters) / (1 - i * alpha / iters)

        norm_a = torch.linalg.norm(a)
        cos_sim = (a / norm_a * (b / norm_a)).sum()
        b_perp = b - a * cos_sim
        new_a = a + adjusted_alpha * b_perp
        if torch.isnan(new_a).any():
            return a
        a = new_a

    return a


@merge_methods.register()
def multiply_difference(
    a: Tensor | LiftFlag[MergeSpace.DELTA],
    b: Tensor | LiftFlag[MergeSpace.DELTA],
    alpha: float,
    beta: float,
) -> Tensor | LiftFlag[MergeSpace.DELTA]:
    a_pow = torch.pow(torch.abs(a), (1 - alpha))
    b_pow = torch.pow(torch.abs(b), alpha)
    difference = torch.copysign(a_pow * b_pow, weighted_sum(a, b, beta))
    return difference


@merge_methods.register()
def similarity_add_difference(
    a: Tensor | LiftFlag[MergeSpace.DELTA],
    b: Tensor | LiftFlag[MergeSpace.DELTA],
    alpha: float,
    beta: float,
) -> Tensor | LiftFlag[MergeSpace.DELTA]:
    threshold = torch.maximum(torch.abs(a), torch.abs(b))
    similarity = (a * b / threshold**2 + 1) / 2
    similarity = torch.nan_to_num(similarity * beta, nan=beta)

    ab_diff = add(a, b, alpha)
    ab_sum = weighted_sum(a, b, alpha / 2)
    return (1 - similarity) * ab_diff + similarity * ab_sum


@merge_methods.register()
def ties_add_difference(
    a: Tensor | LiftFlag[MergeSpace.DELTA],
    b: Tensor | LiftFlag[MergeSpace.DELTA],
    alpha: float,
    beta: float,
) -> Tensor | LiftFlag[MergeSpace.DELTA]:
    deltas = []
    signs = []
    for m in [a, b]:
        deltas.append(filter_top_k(m, beta))
        signs.append(torch.sign(deltas[-1]))

    signs = torch.stack(signs, dim=0)
    final_sign = torch.sign(torch.sum(signs, dim=0))
    deltas = torch.stack(deltas, dim=0)
    delta_filters = deltas * (signs == final_sign).float()

    res = torch.zeros_like(a, device=a.device)
    for delta_filter in delta_filters:
        res += delta_filter

    param_count = torch.sum(delta_filters, dim=0)
    return alpha * torch.nan_to_num(res / param_count)


def filter_top_k(a: Tensor, k: float):
    k = max(int((1 - k) * torch.numel(a)), 1)
    k_value, _ = torch.kthvalue(torch.abs(a.flatten()).float(), k)
    top_k_filter = (torch.abs(a) >= k_value).float()
    return a * top_k_filter


@merge_methods.register()
def tensor_sum(
    a: Tensor | SharedSpace,
    b: Tensor | SharedSpace,
    alpha: float,
    beta: float,
) -> Tensor | SharedSpace:
    if alpha + beta <= 1:
        tt = a.clone()
        talphas = int(a.shape[0] * beta)
        talphae = int(a.shape[0] * (alpha + beta))
        tt[talphas:talphae] = b[talphas:talphae].clone()
    else:
        talphas = int(a.shape[0] * (alpha + beta - 1))
        talphae = int(a.shape[0] * beta)
        tt = b.clone()
        tt[talphas:talphae] = a[talphas:talphae].clone()
    return tt


@merge_methods.register()
def top_k_tensor_sum(
    a: Tensor | SharedSpace,
    b: Tensor | SharedSpace,
    alpha: float,
    beta: float,
) -> Tensor | SharedSpace:
    a_flat = torch.flatten(a)
    a_dist = torch.msort(a_flat)
    b_indices = torch.argsort(torch.flatten(b), stable=True)
    redist_indices = torch.argsort(b_indices)

    start_i, end_i, region_is_inverted = ratio_to_region(alpha, beta, torch.numel(a))
    start_top_k = kth_abs_value(a_dist, start_i)
    end_top_k = kth_abs_value(a_dist, end_i)

    indices_mask = (start_top_k < torch.abs(a_dist)) & (torch.abs(a_dist) <= end_top_k)
    if region_is_inverted:
        indices_mask = ~indices_mask
    indices_mask = torch.gather(indices_mask.float(), 0, redist_indices)

    a_redist = torch.gather(a_dist, 0, redist_indices)
    a_redist = (1 - indices_mask) * a_flat + indices_mask * a_redist
    return a_redist.reshape_as(a)


def kth_abs_value(a: Tensor, k: int) -> Tensor:
    if k <= 0:
        return torch.tensor(-1, device=a.device)
    else:
        return torch.kthvalue(torch.abs(a.float()), k)[0]


def ratio_to_region(width: float, offset: float, n: int) -> Tuple[int, int, bool]:
    if width < 0:
        offset += width
        width = -width
    width = min(width, 1)

    if offset < 0:
        offset = 1 + offset - int(offset)
    offset = math.fmod(offset, 1.0)

    if width + offset <= 1:
        inverted = False
        start = offset * n
        end = (width + offset) * n
    else:
        inverted = True
        start = (width + offset - 1) * n
        end = offset * n

    return round(start), round(end), inverted


@merge_methods.register()
def distribution_crossover(
    a: Tensor | LiftFlag[MergeSpace.MODEL],
    b: Tensor | LiftFlag[MergeSpace.MODEL],
    c: Tensor | LiftFlag[MergeSpace.MODEL],
    alpha: float,
    beta: float,
) -> Tensor | LiftFlag[MergeSpace.MODEL]:
    if a.shape == ():
        return weighted_sum(a, b, alpha)

    c_indices = torch.argsort(torch.flatten(c))
    a_dist = torch.gather(torch.flatten(a), 0, c_indices)
    b_dist = torch.gather(torch.flatten(b), 0, c_indices)

    a_dft = torch.fft.rfft(a_dist.float())
    b_dft = torch.fft.rfft(b_dist.float())

    dft_filter = torch.arange(0, torch.numel(a_dft), device=a_dft.device).float()
    dft_filter /= torch.numel(a_dft)
    if beta > EPSILON:
        dft_filter = (dft_filter - alpha) / math.tan(beta * math.pi / 2) + alpha
        dft_filter = torch.clamp(dft_filter, 0.0, 1.0)
    else:
        dft_filter = (dft_filter >= alpha).float()

    x_dft = (1 - dft_filter) * a_dft + dft_filter * b_dft
    x_dist = torch.fft.irfft(x_dft, a_dist.shape[0])
    x_values = torch.gather(x_dist, 0, torch.argsort(c_indices))
    return x_values.reshape_as(a)


@merge_methods.register()
def rotate(
    a: Tensor | LiftFlag[MergeSpace.DELTA],
    b: Tensor | LiftFlag[MergeSpace.DELTA],
    alpha: float,
    beta: float,
    cache: Optional[Dict[str, Tensor]],
) -> Tensor | LiftFlag[MergeSpace.DELTA]:
    if alpha == 0 and beta == 0:
        return a

    is_conv = len(a.shape) == 4 and a.shape[-1] != 1
    if len(a.shape) == 0 or is_conv or torch.allclose(a.half(), b.half()):
        return weighted_sum(a, b, beta)

    if len(a.shape) == 4:
        shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    else:
        shape_2d = (-1, a.shape[-1])

    a_neurons = a.reshape(*shape_2d).double()
    b_neurons = b.reshape(*shape_2d).double()

    a_centroid = a_neurons.mean(0)
    b_centroid = b_neurons.mean(0)
    new_centroid = weighted_sum(a_centroid, b_centroid, alpha)
    if len(a.shape) == 1 or len(a.shape) == 2 and a.shape[0] == 1:
        return new_centroid.reshape_as(a)

    a_neurons -= a_centroid
    b_neurons -= b_centroid

    alpha_is_float = alpha != round(alpha)

    if cache is not None and "rotation" in cache:
        rotation = transform = cache["rotation"].to(a.device)
    else:
        svd_driver = "gesvd" if a.is_cuda else None
        u, _, v_t = torch.linalg.svd(a_neurons.T @ b_neurons, driver=svd_driver)

        if alpha_is_float:
            # cancel reflection. without this, eigenvalues often have a complex component
            #   and then we can't obtain a valid dtype for the merge
            u[:, -1] /= torch.det(u) * torch.det(v_t)

        rotation = transform = u @ v_t
        if not torch.isfinite(u).all():
            raise ValueError(
                f"determinant error: {torch.det(rotation)}. "
                'This can happen when merging on the CPU with the "rotate" method. '
                "Consider merging on a cuda device, "
                "or try setting alpha to 1 for the problematic blocks. "
                "See this related discussion for more info: "
                "https://github.com/s1dlx/meh/pull/50#discussion_r1429469484"
            )

        if cache is not None:
            cache["rotation"] = rotation.cpu()

    if alpha_is_float:
        transform = fractional_matrix_power(transform, alpha, cache)
    elif alpha == 0:
        transform = torch.eye(
            len(transform),
            dtype=transform.dtype,
            device=transform.device,
        )
    elif alpha != 1:
        transform = torch.linalg.matrix_power(transform, round(alpha))

    if beta != 0:
        # interpolate the relationship between the neurons
        a_neurons = weighted_sum(a_neurons, b_neurons @ rotation.T, beta)

    a_neurons @= transform
    a_neurons += new_centroid
    return a_neurons.reshape_as(a).to(a.dtype)


def fractional_matrix_power(matrix: Tensor, power: float, cache: Dict[str, Tensor]):
    if cache is not None and "eigenvalues" in cache:
        eigenvalues = cache["eigenvalues"].to(matrix.device)
        eigenvectors = cache["eigenvectors"].to(matrix.device)
        eigenvectors_inv = cache["eigenvectors_inv"].to(matrix.device)
    else:
        eigenvalues, eigenvectors = torch.linalg.eig(matrix)
        eigenvectors_inv = torch.linalg.inv(eigenvectors)
        if cache is not None:
            cache["eigenvalues"] = eigenvalues.cpu()
            cache["eigenvectors"] = eigenvectors.cpu()
            cache["eigenvectors_inv"] = eigenvectors_inv.cpu()

    eigenvalues.pow_(power)
    result = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors_inv
    return result.real.to(dtype=matrix.dtype)
