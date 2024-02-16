import functools
import math
import operator
import torch
from torch import Tensor
from typing import Tuple, TypeVar, Dict, Optional
from sd_mecha.sd_meh.extensions import merge_methods, MergeSpace, LiftFlag


EPSILON = 1e-10
SharedMergeSpace = TypeVar("SharedMergeSpace", bound=LiftFlag[MergeSpace.MODEL | MergeSpace.DELTA])


@merge_methods.register
def weighted_sum(
    a: Tensor | SharedMergeSpace,
    b: Tensor | SharedMergeSpace,
    alpha: float,
) -> Tensor | SharedMergeSpace:
    return (1 - alpha) * a + alpha * b


@merge_methods.register
def slerp(
    a: Tensor | SharedMergeSpace,
    b: Tensor | SharedMergeSpace,
    alpha: float,
) -> Tensor | SharedMergeSpace:
    a_normalized = a / a.norm()
    b_normalized = b / b.norm()

    ab_dot = torch.sum(a_normalized * b_normalized)

    if 1 - torch.abs(ab_dot) < EPSILON:
        return weighted_sum(a, b, alpha)

    omega = torch.arccos(ab_dot)
    a_contrib = a * torch.sin((1-alpha)*omega)
    b_contrib = b * torch.sin(alpha*omega)
    res = (a_contrib + b_contrib) / torch.sin(omega)
    return res * weighted_sum(a.norm(), b.norm(), alpha)


@merge_methods.register
def add_difference(
    a: Tensor | SharedMergeSpace,
    b: Tensor | LiftFlag[MergeSpace.DELTA],
    alpha: float,
) -> Tensor | SharedMergeSpace:
    return a + alpha * b


@merge_methods.register
def subtract(
    a: Tensor | LiftFlag[MergeSpace.MODEL],
    b: Tensor | LiftFlag[MergeSpace.MODEL],
) -> Tensor | LiftFlag[MergeSpace.DELTA]:
    return a - b


@merge_methods.register
def perpendicular_component(
    a: Tensor | SharedMergeSpace,
    b: Tensor | SharedMergeSpace,
) -> Tensor | SharedMergeSpace:
    norm_a = torch.linalg.norm(a)
    res = b - a * (a / norm_a * (b / norm_a)).sum()
    if res.isnan().any():
        return torch.zeros_like(a)
    return res


@merge_methods.register
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


@merge_methods.register
def copy_difference(  # aka train_difference
    a: Tensor | SharedMergeSpace,
    b: Tensor | SharedMergeSpace,
    c: Tensor | SharedMergeSpace,
):
    ab_diff = a - b
    bc_dist = torch.abs(b - c)
    ba_dist = torch.abs(b - a)

    sum_distances = bc_dist + ba_dist
    scale = torch.where(
        sum_distances != 0,
        ba_dist / sum_distances,
        torch.tensor(0.0, dtype=a.dtype, device=a.device)
    )
    sign_scale = torch.sign(b - c)
    scale = sign_scale * torch.abs(scale)
    new_diff = scale * torch.abs(ab_diff)
    return new_diff * 1.8


@merge_methods.register
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


@merge_methods.register
def similarity_sum(  # aka add_cosine_a
    a: torch.Tensor | LiftFlag[MergeSpace.MODEL],
    b: torch.Tensor | LiftFlag[MergeSpace.MODEL],
    alpha: float,
) -> torch.Tensor | LiftFlag[MergeSpace.MODEL]:
    a_norm = torch.nn.functional.normalize(a, dim=0)
    b_norm = torch.nn.functional.normalize(b, dim=0)
    similarity = torch.nn.functional.cosine_similarity(a_norm, b_norm, dim=0)
    return add_cosine_generic(a, b, alpha, similarity)


@merge_methods.register
def unrestricted_similarity_sum(  # aka add_cosine_b
    a: torch.Tensor | LiftFlag[MergeSpace.MODEL],
    b: torch.Tensor | LiftFlag[MergeSpace.MODEL],
    alpha: float,
) -> torch.Tensor | LiftFlag[MergeSpace.MODEL]:
    similarity = torch.nn.functional.cosine_similarity(a, b, dim=0)
    dot_product = torch.sum(a * b)
    magnitude_similarity = dot_product / (torch.norm(a) * torch.norm(b))
    combined_similarity = (similarity + magnitude_similarity) / 2.0
    return add_cosine_generic(a, b, alpha, combined_similarity)


def add_cosine_generic(a: torch.Tensor, b: torch.Tensor, alpha: float, similarity: torch.Tensor) -> torch.Tensor:
    k = 1 - torch.clamp(similarity - alpha, 0, 1)
    return weighted_sum(a, b, k)


@merge_methods.register
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


@merge_methods.register
def tensor_sum(
    a: Tensor | SharedMergeSpace,
    b: Tensor | SharedMergeSpace,
    alpha: float,
    beta: float,
) -> Tensor | SharedMergeSpace:
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


@merge_methods.register
def clip(
    a: Tensor | SharedMergeSpace,
    b: Tensor | SharedMergeSpace,
    c: Tensor | SharedMergeSpace,
) -> Tensor | SharedMergeSpace:
    maximums = torch.maximum(b, c)
    minimums = torch.minimum(b, c)
    return torch.minimum(torch.maximum(a, minimums), maximums)


@merge_methods.register
def top_k_tensor_sum(
    a: Tensor | SharedMergeSpace,
    b: Tensor | SharedMergeSpace,
    alpha: float,
    beta: float,
) -> Tensor | SharedMergeSpace:
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


@merge_methods.register
def distribution_crossover(
    a: Tensor | SharedMergeSpace,
    b: Tensor | SharedMergeSpace,
    c: Tensor | SharedMergeSpace,
    alpha: float,
    beta: float,
) -> Tensor | SharedMergeSpace:
    if a.shape == ():
        return weighted_sum(a, b, alpha)

    c_indices = torch.argsort(torch.flatten(c))
    a_dist = torch.gather(torch.flatten(a), 0, c_indices)
    b_dist = torch.gather(torch.flatten(b), 0, c_indices)

    a_dft = torch.fft.rfft(a_dist)
    b_dft = torch.fft.rfft(b_dist)

    dft_filter = create_filter((a_dft.numel(),), alpha, beta, device=a.device)

    x_dft = (1 - dft_filter) * a_dft + dft_filter * b_dft
    x_dist = torch.fft.irfft(x_dft, a_dist.shape[0])
    x_values = torch.gather(x_dist, 0, torch.argsort(c_indices))
    return x_values.reshape_as(a)


@merge_methods.register
def crossover(
    a: Tensor | SharedMergeSpace,
    b: Tensor | SharedMergeSpace,
    alpha: float,
    beta: float,
) -> Tensor | SharedMergeSpace:
    if alpha == 0 and beta == 0:
        return a

    if len(a.shape) == 0 or torch.allclose(a.half(), b.half()):
        return weighted_sum(a, b, beta)

    if a.shape[0] > 40000 or len(a.shape) == 4 and sum(a.shape[2:]) > 2:
        shape = a.shape[1:]
    else:
        shape = a.shape

    a_dft = torch.fft.rfftn(a, s=shape)
    b_dft = torch.fft.rfftn(b, s=shape)

    dft_filter = create_filter(a_dft.shape, alpha, beta, device=a.device)

    x_dft = (1 - dft_filter)*a_dft + dft_filter*b_dft
    return torch.fft.irfftn(x_dft, s=shape)


def create_filter(shape: Tuple[int, ...] | torch.Size, alpha: float, beta: float, steps=100, precision=EPSILON, device=None):
    gradients = [
        torch.linspace(0, 1, s, device=device)**2
        for s in shape
    ]

    if len(shape) > 1:
        grids = torch.meshgrid(*gradients, indexing='ij')
        mesh = torch.sqrt(torch.sum(torch.stack(grids), dim=0)) / math.sqrt(len(shape))
    else:
        mesh = gradients[0]

    # train the cut to pick the right ratio of parameters
    phi_alpha = alpha
    dft_filter = mesh
    for step in range(steps):
        if beta < EPSILON:
            dft_filter = (mesh > 1 - phi_alpha).float()
        else:
            cot_b = 1 / math.tan(math.pi * beta / 2)
            dft_filter = torch.clamp(mesh*cot_b + phi_alpha*cot_b + phi_alpha - cot_b, 0, 1)
        filter_mean = dft_filter.mean()
        loss = alpha - filter_mean
        if abs(loss) < precision:
            break
        phi_alpha += loss

    return dft_filter


@merge_methods.register
def rotate(
    a: Tensor | SharedMergeSpace,
    b: Tensor | SharedMergeSpace,
    alpha: float,
    beta: float,
    cache: Optional[Dict[str, Tensor]],
) -> Tensor | SharedMergeSpace:
    if alpha == 0 and beta == 0:
        return a

    is_conv = len(a.shape) == 4 and a.shape[-1] != 1
    if len(a.shape) == 0 or is_conv or torch.allclose(a.half(), b.half()):
        return weighted_sum(a, b, beta)

    if len(a.shape) == 4:
        shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    else:
        shape_2d = (-1, a.shape[-1])

    a_neurons = a.reshape(*shape_2d)
    b_neurons = b.reshape(*shape_2d)

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
    return a_neurons.reshape_as(a)


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
