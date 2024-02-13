import functools
import math
import operator
import textwrap

import torch
from torch import Tensor
from typing import Tuple

__all__ = [
    "weighted_sum",
    "weighted_subtraction",
    "tensor_sum",
    "add_difference",
    "sum_twice",
    "triple_sum",
    "euclidean_add_difference",
    "multiply_difference",
    "top_k_tensor_sum",
    "similarity_add_difference",
    "distribution_crossover",
    "ties_add_difference",
    "add_perpendicular",
    "rotate",
]


EPSILON = 1e-10  # Define a small constant EPSILON to prevent division by zero


def weighted_sum(a: Tensor, b: Tensor, alpha: float, **kwargs) -> Tensor:
    return (1 - alpha) * a + alpha * b


def weighted_subtraction(
    a: Tensor, b: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    # Adjust beta if both alpha and beta are 1.0 to avoid division by zero
    if alpha == 1.0 and beta == 1.0:
        beta -= EPSILON

    return (a - alpha * beta * b) / (1 - alpha * beta)


def tensor_sum(a: Tensor, b: Tensor, alpha: float, beta: float, **kwargs) -> Tensor:
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


def add_difference(a: Tensor, b: Tensor, c: Tensor, alpha: float, **kwargs) -> Tensor:
    return a + alpha * (b - c)


def sum_twice(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    return (1 - beta) * ((1 - alpha) * a + alpha * b) + beta * c


def triple_sum(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    return (1 - alpha - beta) * a + alpha * b + beta * c


def euclidean_add_difference(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, **kwargs
) -> Tensor:
    a_diff = a.float() - c.float()
    b_diff = b.float() - c.float()
    a_diff = torch.nan_to_num(a_diff / torch.linalg.norm(a_diff))
    b_diff = torch.nan_to_num(b_diff / torch.linalg.norm(b_diff))

    distance = (1 - alpha) * a_diff**2 + alpha * b_diff**2
    distance = torch.sqrt(distance)
    sum_diff = weighted_sum(a.float(), b.float(), alpha) - c.float()
    distance = torch.copysign(distance, sum_diff)

    target_norm = torch.linalg.norm(sum_diff)
    return c + distance / torch.linalg.norm(distance) * target_norm


def multiply_difference(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    diff_a = torch.pow(torch.abs(a.float() - c), (1 - alpha))
    diff_b = torch.pow(torch.abs(b.float() - c), alpha)
    difference = torch.copysign(diff_a * diff_b, weighted_sum(a, b, beta) - c)
    return c + difference.to(c.dtype)


def top_k_tensor_sum(
    a: Tensor, b: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
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


def similarity_add_difference(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    threshold = torch.maximum(torch.abs(a), torch.abs(b))
    similarity = ((a * b / threshold**2) + 1) / 2
    similarity = torch.nan_to_num(similarity * beta, nan=beta)

    ab_diff = a + alpha * (b - c)
    ab_sum = (1 - alpha / 2) * a + (alpha / 2) * b
    return (1 - similarity) * ab_diff + similarity * ab_sum


def distribution_crossover(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
):
    if a.shape == ():
        return alpha * a + (1 - alpha) * b

    c_indices = torch.argsort(torch.flatten(c))
    a_dist = torch.gather(torch.flatten(a), 0, c_indices)
    b_dist = torch.gather(torch.flatten(b), 0, c_indices)

    a_dft = torch.fft.rfft(a_dist.float())
    b_dft = torch.fft.rfft(b_dist.float())

    dft_filter = torch.arange(0, torch.numel(a_dft), device=a_dft.device).float()
    dft_filter /= torch.numel(a_dft)
    if beta > EPSILON:
        dft_filter = (alpha - dft_filter) / math.tan(beta * math.pi / 2) + alpha
        dft_filter = torch.clamp(dft_filter, 0.0, 1.0)
    else:
        dft_filter = (dft_filter >= alpha).float()

    x_dft = (1 - dft_filter) * a_dft + dft_filter * b_dft
    x_dist = torch.fft.irfft(x_dft, a_dist.shape[0])
    x_values = torch.gather(x_dist, 0, torch.argsort(c_indices))
    return x_values.reshape_as(a)


def ties_add_difference(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    deltas = []
    signs = []
    for m in [a, b]:
        deltas.append(filter_top_k(m - c, beta))
        signs.append(torch.sign(deltas[-1]))

    signs = torch.stack(signs, dim=0)
    final_sign = torch.sign(torch.sum(signs, dim=0))
    delta_filters = (signs == final_sign).float()

    res = torch.zeros_like(c, device=c.device)
    for delta_filter, delta in zip(delta_filters, deltas):
        res += delta_filter * delta

    param_count = torch.sum(delta_filters, dim=0)
    return c + alpha * torch.nan_to_num(res / param_count)


def filter_top_k(a: Tensor, k: float):
    k = max(int((1 - k) * torch.numel(a)), 1)
    k_value, _ = torch.kthvalue(torch.abs(a.flatten()).float(), k)
    top_k_filter = (torch.abs(a) >= k_value).float()
    return a * top_k_filter


def add_perpendicular(
    a: Tensor, b: Tensor, alpha: float, c: Tensor = None, **kwargs
) -> Tensor:
    a_diff = a.float() - c.float()
    b_diff = b.float() - c.float()
    a_ortho = a_diff * (a_diff / torch.linalg.norm(a_diff) * (b_diff / torch.linalg.norm(a_diff))).sum()
    b_perp = b_diff - a_ortho
    res = a + alpha * b_perp
    if torch.isnan(res).any():
        return a
    return res.to(a.dtype)


def rotate(a: Tensor, b: Tensor, alpha: float, beta: float, **kwargs):
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

    if kwargs["cache"] is not None and "rotation" in kwargs["cache"]:
        rotation = transform = kwargs["cache"]["rotation"].to(a.device)
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
                textwrap.dedent(
                    f"""determinant error: {torch.det(rotation)}.
                This can happen when merging on the CPU with the "rotate" method.
                Consider merging on a cuda device, or try setting alpha to 1 for the problematic blocks.
                See this related discussion for more info: https://github.com/s1dlx/meh/pull/50#discussion_r1429469484"""
                )
            )

        if kwargs["cache"] is not None:
            kwargs["cache"]["rotation"] = rotation.cpu()

    if alpha_is_float:
        transform = fractional_matrix_power(transform, alpha, kwargs["cache"])
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


def fractional_matrix_power(matrix: Tensor, power: float, cache: dict):
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
