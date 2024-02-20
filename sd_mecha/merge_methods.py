import functools
import math
import operator
import torch
from torch import Tensor
from typing import Tuple, TypeVar, Dict, Optional
from sd_mecha.extensions import MergeSpace, LiftFlag, convert_to_recipe


EPSILON = 1e-10
SameMergeSpace = TypeVar("SameMergeSpace", bound=LiftFlag[MergeSpace.MODEL | MergeSpace.DELTA])


@convert_to_recipe
def weighted_sum(
    a: Tensor | SameMergeSpace,
    b: Tensor | SameMergeSpace,
    *,
    alpha: float = 0.5,
    **kwargs,
) -> Tensor | SameMergeSpace:
    return (1 - alpha) * a + alpha * b


@convert_to_recipe
def slerp(
    a: Tensor | SameMergeSpace,
    b: Tensor | SameMergeSpace,
    *,
    alpha: float = 0.5,
    **kwargs,
) -> Tensor | SameMergeSpace:
    a_normalized = a / a.norm()
    b_normalized = b / b.norm()

    ab_dot = torch.sum(a_normalized * b_normalized)

    if 1 - torch.abs(ab_dot) < EPSILON:
        return weighted_sum.__wrapped__(a, b, alpha=alpha)

    omega = torch.arccos(ab_dot)
    a_contrib = a * torch.sin((1-alpha)*omega)
    b_contrib = b * torch.sin(alpha*omega)
    res = (a_contrib + b_contrib) / torch.sin(omega)
    return res * weighted_sum.__wrapped__(a.norm(), b.norm(), alpha=alpha)


@convert_to_recipe
def add_difference(
    a: Tensor | SameMergeSpace,
    b: Tensor | LiftFlag[MergeSpace.DELTA],
    *,
    alpha: float = 0.5,
    **kwargs,
) -> Tensor | SameMergeSpace:
    return a + alpha * b


@convert_to_recipe
def subtract(
    a: Tensor | LiftFlag[MergeSpace.MODEL],
    b: Tensor | LiftFlag[MergeSpace.MODEL],
    **kwargs,
) -> Tensor | LiftFlag[MergeSpace.DELTA]:
    return a - b


@convert_to_recipe
def perpendicular_component(
    a: Tensor | SameMergeSpace,
    b: Tensor | SameMergeSpace,
    **kwargs,
) -> Tensor | SameMergeSpace:
    norm_a = torch.linalg.norm(a)
    res = b - a * (a / norm_a * (b / norm_a)).sum()
    if res.isnan().any():
        return torch.zeros_like(a)
    return res


@convert_to_recipe
def geometric_sum(
    a: Tensor | LiftFlag[MergeSpace.DELTA],
    b: Tensor | LiftFlag[MergeSpace.DELTA],
    *,
    alpha: float,
    **kwargs,
) -> Tensor | LiftFlag[MergeSpace.DELTA]:
    a = torch.complex(a, torch.zeros_like(a))
    b = torch.complex(b, torch.zeros_like(b))
    res = a ** (1 - alpha) * b ** alpha
    return res.abs() * torch.cos(res.angle())


@convert_to_recipe
def similarity_add_difference(
    a: Tensor | LiftFlag[MergeSpace.DELTA],
    b: Tensor | LiftFlag[MergeSpace.DELTA],
    *,
    alpha: float,
    similarity_scale: float = 1.0,
    **kwargs,
) -> Tensor | LiftFlag[MergeSpace.DELTA]:
    threshold = torch.maximum(torch.abs(a), torch.abs(b))
    similarity = (a * b / threshold**2 + 1) / 2
    similarity = torch.nan_to_num(similarity * similarity_scale, nan=similarity_scale)

    ab_diff = add_difference.__wrapped__(a, b, alpha=alpha)
    ab_sum = weighted_sum.__wrapped__(a, b, alpha=alpha / 2)
    return (1 - similarity) * ab_diff + similarity * ab_sum


@convert_to_recipe
def normalized_similarity_sum(  # aka add_cosine_a
    a: Tensor | LiftFlag[MergeSpace.MODEL],
    b: Tensor | LiftFlag[MergeSpace.MODEL],
    *,
    alpha: float,
    **kwargs,
) -> Tensor | LiftFlag[MergeSpace.MODEL]:
    a_norm = torch.nn.functional.normalize(a, dim=0)
    b_norm = torch.nn.functional.normalize(b, dim=0)
    similarity = torch.nn.functional.cosine_similarity(a_norm, b_norm, dim=0)
    return add_cosine_generic(a, b, alpha, similarity)


@convert_to_recipe
def similarity_sum(  # aka add_cosine_b
    a: Tensor | LiftFlag[MergeSpace.MODEL],
    b: Tensor | LiftFlag[MergeSpace.MODEL],
    *,
    alpha: float,
    **kwargs,
) -> Tensor | LiftFlag[MergeSpace.MODEL]:
    similarity = torch.nn.functional.cosine_similarity(a, b, dim=0)
    dot_product = torch.sum(a * b)
    magnitude_similarity = dot_product / (torch.norm(a) * torch.norm(b))
    combined_similarity = (similarity + magnitude_similarity) / 2.0
    return add_cosine_generic(a, b, alpha, combined_similarity)


def add_cosine_generic(a: Tensor, b: Tensor, alpha: float, similarity: Tensor) -> Tensor:
    k = 1 - torch.clamp(similarity - alpha, 0, 1)
    return weighted_sum.__wrapped__(a, b, alpha=k)


@convert_to_recipe
def ties_sum(  # aka add_difference_ties
    *models: Tensor | LiftFlag[MergeSpace.DELTA],
    alpha: float,
    k: float = 0.2,
    **kwargs,
) -> Tensor | LiftFlag[MergeSpace.DELTA]:
    deltas = []
    signs = []
    for m in models:
        deltas.append(filter_top_k(m, k))
        signs.append(torch.sign(deltas[-1]))

    signs = torch.stack(signs, dim=0)
    final_sign = torch.sign(torch.sum(signs, dim=0))
    deltas = torch.stack(deltas, dim=0)
    delta_filters = (signs == final_sign).float()
    filtered_delta = (deltas * delta_filters).sum(dim=0)

    param_counts = torch.sum(delta_filters, dim=0)
    return alpha * torch.nan_to_num(filtered_delta / param_counts)


def filter_top_k(a: Tensor, k: float):
    k = max(int((1 - k) * torch.numel(a)), 1)
    k_value, _ = torch.kthvalue(torch.abs(a.flatten()).float(), k)
    top_k_filter = (torch.abs(a) >= k_value).float()
    return a * top_k_filter


@convert_to_recipe
def copy_region(  # aka tensor_sum
    a: Tensor | SameMergeSpace,
    b: Tensor | SameMergeSpace,
    *,
    width: float,
    offset: float,
    **kwargs,
) -> Tensor | SameMergeSpace:
    start_i, end_i, region_is_inverted = ratio_to_region(width, offset, a.size(0))
    if region_is_inverted:
        b[start_i:end_i] = a[start_i:end_i]
        return b
    else:
        a[start_i:end_i] = b[start_i:end_i]
        return a


@convert_to_recipe
def copy_top_k(  # aka top_k_tensor_sum
    a: Tensor | SameMergeSpace,
    b: Tensor | SameMergeSpace,
    *,
    width: float,
    offset: float,
    **kwargs,
) -> Tensor | SameMergeSpace:
    a_flat = torch.flatten(a)
    a_dist = torch.msort(a_flat)
    b_indices = torch.argsort(torch.flatten(b), stable=True)
    redist_indices = torch.argsort(b_indices)

    start_i, end_i, region_is_inverted = ratio_to_region(width, offset, torch.numel(a))
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


@convert_to_recipe
def copy_difference(  # aka train_difference
    a: Tensor | SameMergeSpace,
    b: Tensor | SameMergeSpace,
    c: Tensor | SameMergeSpace,
    **kwargs,
) -> Tensor | SameMergeSpace:
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


@convert_to_recipe
def distribution_crossover(
    a: Tensor | SameMergeSpace,
    b: Tensor | SameMergeSpace,
    c: Tensor | SameMergeSpace,
    *,
    mean: float,
    tilt: float,
    **kwargs,
) -> Tensor | SameMergeSpace:
    if mean == 0:
        return a
    if tilt == 1 or a.shape == ():
        return weighted_sum.__wrapped__(a, b, alpha=mean)

    c_indices = torch.argsort(torch.flatten(c))
    a_dist = torch.gather(torch.flatten(a), 0, c_indices)
    b_dist = torch.gather(torch.flatten(b), 0, c_indices)

    a_dft = torch.fft.rfft(a_dist)
    b_dft = torch.fft.rfft(b_dist)

    dft_filter = create_filter((a_dft.numel(),), mean, tilt, device=a.device)

    x_dft = (1 - dft_filter) * a_dft + dft_filter * b_dft
    x_dist = torch.fft.irfft(x_dft, a_dist.shape[0])
    x_values = torch.gather(x_dist, 0, torch.argsort(c_indices))
    return x_values.reshape_as(a)


@convert_to_recipe
def crossover(
    a: Tensor | SameMergeSpace,
    b: Tensor | SameMergeSpace,
    *,
    mean: float,
    tilt: float,
    **kwargs,
) -> Tensor | SameMergeSpace:
    if mean == 0:
        return a
    if tilt == 1:
        return weighted_sum.__wrapped__(a, b, alpha=mean)

    if len(a.shape) == 0 or torch.allclose(a.half(), b.half()):
        return weighted_sum.__wrapped__(a, b, alpha=tilt)

    if a.shape[0] > 40000 or len(a.shape) == 4 and sum(a.shape[2:]) > 2:
        shape = a.shape[1:]
    else:
        shape = a.shape

    a_dft = torch.fft.rfftn(a, s=shape)
    b_dft = torch.fft.rfftn(b, s=shape)

    dft_filter = create_filter(a_dft.shape, mean, tilt, device=a.device)

    x_dft = (1 - dft_filter)*a_dft + dft_filter*b_dft
    return torch.fft.irfftn(x_dft, s=shape)


def create_filter(shape: Tuple[int, ...] | torch.Size, mean: float, tilt: float, steps=100, precision=EPSILON, device=None):
    """
    Create a crossover filter. The cut is first tilted, then slid along its normal to match the mean.
    :param shape: shape of the filter
    :param mean: the mean of the filter. must be in [0, 1]
    :param tilt: tilt of the filter. 0 = vertical filter, 0.5 = 45 degrees, 1 = degenerates to a weighted sum with alpha=mean
    :param steps: maximum number of optimization steps to apply over the mean until the filter converges
    :param precision: the accepted loss between the requested mean and the effective mean of the filter
    :param device: device of the filter
    :return:
    """
    if not 0 <= mean <= 1:
        raise ValueError("filter mean must be between 0 and 1")

    gradients = [
        torch.linspace(0, 1, s, device=device)**2
        for s in shape
    ]

    if len(shape) > 1:
        grids = torch.meshgrid(*gradients, indexing='ij')
        mesh = torch.sqrt(torch.sum(torch.stack(grids), dim=0)) / math.sqrt(len(shape))
    else:
        mesh = gradients[0]

    # train the offset of the cut to pick the right ratio of parameters
    trained_offset = mean
    dft_filter = mesh
    for step in range(steps):
        if tilt < EPSILON:
            dft_filter = (mesh > 1 - trained_offset).float()
        else:
            tilt_cot = 1 / math.tan(math.pi * tilt / 2)
            dft_filter = torch.clamp(mesh*tilt_cot + trained_offset*tilt_cot + trained_offset - tilt_cot, 0, 1)
        current_mean = dft_filter.mean()
        loss = mean - current_mean
        if abs(loss) < precision:
            break
        trained_offset += loss

    return dft_filter


@convert_to_recipe(volatile_hypers=["cache"])
def rotate(
    a: Tensor | SameMergeSpace,
    b: Tensor | SameMergeSpace,
    *,
    alpha: float,
    beta: float,
    cache: Optional[Dict[str, Dict[str, Tensor]]] = None,
    **kwargs,
) -> Tensor | SameMergeSpace:
    if alpha == 0 and beta == 0:
        return a

    is_conv = len(a.shape) == 4 and a.shape[-1] != 1
    if len(a.shape) == 0 or is_conv or torch.allclose(a.half(), b.half()):
        return weighted_sum.__wrapped__(a, b, alpha=beta)

    if len(a.shape) == 4:
        shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    else:
        shape_2d = (-1, a.shape[-1])

    a_neurons = a.reshape(*shape_2d)
    b_neurons = b.reshape(*shape_2d)

    a_centroid = a_neurons.mean(0)
    b_centroid = b_neurons.mean(0)
    new_centroid = weighted_sum.__wrapped__(a_centroid, b_centroid, alpha=alpha)
    if len(a.shape) == 1 or len(a.shape) == 2 and a.shape[0] == 1:
        return new_centroid.reshape_as(a)

    a_neurons -= a_centroid
    b_neurons -= b_centroid

    alpha_is_float = alpha != round(alpha)

    if cache is not None:
        key = kwargs["key"]
        if key not in cache:
            cache[key] = {}
        cache = cache[key]

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
        a_neurons = weighted_sum.__wrapped__(a_neurons, b_neurons @ rotation.T, alpha=beta)

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


@convert_to_recipe
def clip(
    a: Tensor | SameMergeSpace,
    *bounds: Tensor | SameMergeSpace,
    stiffness: float = 0.0,
    **kwargs,
) -> Tensor | SameMergeSpace:
    maximums = functools.reduce(torch.maximum, bounds)
    minimums = functools.reduce(torch.minimum, bounds)
    centers = (maximums + minimums) / 2

    if stiffness:
        smallest_positive = maximums
        largest_negative = minimums

        for i, bound in enumerate(bounds):
            smallest_positive = torch.where((smallest_positive >= bound) & (bound >= centers), bound, smallest_positive)
            largest_negative = torch.where((largest_negative <= bound) & (bound <= centers), bound, largest_negative)

        maximums = weighted_sum.__wrapped__(maximums, smallest_positive, alpha=stiffness)
        minimums = weighted_sum.__wrapped__(minimums, largest_negative, alpha=stiffness)

    return torch.minimum(torch.maximum(a, minimums), maximums)
