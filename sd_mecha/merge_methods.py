import functools
import math
import operator
import numpy as np
import torch
from scipy.stats import binom
from torch import Tensor
from typing import Tuple, TypeVar, Dict, Optional
from sd_mecha.hypers import Hyper
from sd_mecha.merge_space import MergeSpace
from sd_mecha.extensions.merge_method import LiftFlag, convert_to_recipe


EPSILON = 1e-10
SameMergeSpace = TypeVar("SameMergeSpace", bound=LiftFlag[MergeSpace.BASE | MergeSpace.DELTA])


@convert_to_recipe
def weighted_sum(
    a: Tensor | SameMergeSpace,
    b: Tensor | SameMergeSpace,
    *,
    alpha: Hyper = 0.5,
    **kwargs,
) -> Tensor | SameMergeSpace:
    return (1 - alpha) * a + alpha * b


@convert_to_recipe
def slerp(
    a: Tensor | SameMergeSpace,
    b: Tensor | SameMergeSpace,
    *,
    alpha: Hyper = 0.5,
    **kwargs,
) -> Tensor | SameMergeSpace:
    a_normalized = a / a.norm()
    b_normalized = b / b.norm()

    ab_dot = (a_normalized * b_normalized).sum().clip(-1, 1)

    omega = torch.arccos(ab_dot)
    a_contrib = a_normalized * torch.sin((1-alpha)*omega)
    b_contrib = b_normalized * torch.sin(alpha*omega)
    res = (a_contrib + b_contrib) / torch.sin(omega)
    res *= weighted_sum.__wrapped__(a.norm(), b.norm(), alpha=alpha)
    if res.isnan().any():
        return weighted_sum.__wrapped__(a, b, alpha=alpha)
    return res


@convert_to_recipe
def add_difference(
    a: Tensor | SameMergeSpace,
    b: Tensor | LiftFlag[MergeSpace.DELTA],
    *,
    alpha: Hyper = 0.5,
    **kwargs,
) -> Tensor | SameMergeSpace:
    return a + alpha * b


@convert_to_recipe
def subtract(
    a: Tensor | LiftFlag[MergeSpace.BASE],
    b: Tensor | LiftFlag[MergeSpace.BASE],
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
    alpha: Hyper = 0.5,
    **kwargs,
) -> Tensor | LiftFlag[MergeSpace.DELTA]:
    a = torch.complex(a, torch.zeros_like(a))
    b = torch.complex(b, torch.zeros_like(b))
    res = a ** (1 - alpha) * b ** alpha
    return res.real


@convert_to_recipe
def add_cosine_a(
    a: Tensor | LiftFlag[MergeSpace.BASE],
    b: Tensor | LiftFlag[MergeSpace.BASE],
    *,
    alpha: Hyper,
    **kwargs,
) -> Tensor | LiftFlag[MergeSpace.BASE]:
    a_norm = torch.nn.functional.normalize(a, dim=0)
    b_norm = torch.nn.functional.normalize(b, dim=0)
    similarity = torch.nn.functional.cosine_similarity(a_norm, b_norm, dim=0)
    return add_cosine_generic(a, b, alpha, similarity)


@convert_to_recipe
def add_cosine_b(
    a: Tensor | LiftFlag[MergeSpace.BASE],
    b: Tensor | LiftFlag[MergeSpace.BASE],
    *,
    alpha: Hyper,
    **kwargs,
) -> Tensor | LiftFlag[MergeSpace.BASE]:
    similarity = torch.nn.functional.cosine_similarity(a, b, dim=0)
    dot_product = torch.sum(a * b)
    magnitude_similarity = dot_product / (torch.norm(a) * torch.norm(b))
    combined_similarity = (similarity + magnitude_similarity) / 2.0
    return add_cosine_generic(a, b, alpha, combined_similarity)


def add_cosine_generic(a: Tensor, b: Tensor, alpha: float, similarity: Tensor) -> Tensor:
    k = 1 - torch.clamp(similarity - alpha, 0, 1)
    return weighted_sum.__wrapped__(a, b, alpha=k)


# latex notes in reference to original implementation: https://arxiv.org/abs/2306.01708
# - `delta`: $$ \hat{\tau}_t $$
# - `signs`: $$ \gamma_t $$
# - `final_sign`: $$ \gamma_m^p = sgn(\sum_{t=1}^n \hat{\tau}_t^p) $$
# - `delta_filters`: $$ \{ \gamma_t^p = \gamma_m^p \} $$
# - `param_counts`: $$ |A^p| $$
# - `filtered_delta`: $$ \sum_{t\in{A^p}} \hat{\tau}_t^p $$
# - `return`: $$ \lambda * \tau_m $$
@convert_to_recipe
def ties_sum(  # aka add_difference_ties
    *models: Tensor | LiftFlag[MergeSpace.DELTA],
    k: Hyper = 0.2,
    **kwargs,
) -> Tensor | LiftFlag[MergeSpace.DELTA]:

    # Step 1: Trim redundant parameters

    # $$ \hat{\tau}_t $$ O(N) in space
    deltas = [
        # $$ keep_topk_reset_rest_to_zero(\tau_t, k) $$
        filter_top_k(m, k)
        for m in models
    ]
    deltas = torch.stack(deltas, dim=0)

    # Step 2: Elect Final Signs.

    # $$ \gamma_t $$ 
    signs = torch.sign(deltas)

    # $$ \gamma_m^p = sgn(\sum_{t=1}^n \hat{\tau}_t^p) $$
    final_sign = torch.sign(torch.sum(deltas, dim=0)) 

    # Step 3: Disjoint merge.

    # $$ \{ \gamma_t^p = \gamma_m^p \} $$
    delta_filters = (signs == final_sign).float()

    # $$ |A^p| $$
    param_counts = torch.sum(delta_filters, dim=0)

    # $$ \sum_{t\in{A^P}} \hat{\tau}_t^p $$
    filtered_delta = (deltas * delta_filters).sum(dim=0)

    # $$ \tau_m $$
    return torch.nan_to_num(filtered_delta / param_counts)


def filter_top_k(a: Tensor, k: float):
    k = max(int((1 - k) * torch.numel(a)), 1)
    k_value, _ = torch.kthvalue(torch.abs(a.flatten()).float(), k)
    top_k_filter = (torch.abs(a) >= k_value).float()
    return a * top_k_filter


@convert_to_recipe
def tensor_sum(
    a: Tensor | SameMergeSpace,
    b: Tensor | SameMergeSpace,
    *,
    width: Hyper = 0.5,
    offset: Hyper = 0.0,
    **kwargs,
) -> Tensor | SameMergeSpace:
    if a.shape == ():
        if width > 0.5:
            return b
        return a

    start_i, end_i, region_is_inverted = ratio_to_region(width, offset, a.size(0))
    if region_is_inverted:
        b[start_i:end_i] = a[start_i:end_i]
        return b
    else:
        a[start_i:end_i] = b[start_i:end_i]
        return a


@convert_to_recipe
def top_k_tensor_sum(
    a: Tensor | SameMergeSpace,
    b: Tensor | SameMergeSpace,
    *,
    width: Hyper = 0.5,
    offset: Hyper = 0.0,
    **kwargs,
) -> Tensor | SameMergeSpace:
    a_flat = torch.flatten(a)
    a_dist = torch.msort(a_flat)
    b_indices = torch.argsort(torch.flatten(b), stable=True)
    redist_indices = torch.argsort(b_indices)

    start_i, end_i, region_is_inverted = ratio_to_region(width, offset, torch.numel(a))
    start_top_k = kth_abs_value(a_dist, start_i)
    end_top_k = kth_abs_value(a_dist, end_i)

    indices_mask = (start_top_k <= torch.abs(a_dist)) & (torch.abs(a_dist) <= end_top_k)
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
def train_difference(
    a: Tensor | SameMergeSpace,
    b: Tensor | SameMergeSpace,
    c: Tensor | SameMergeSpace,
    *,
    alpha: Hyper = 1.0,
    **kwargs,
) -> Tensor | SameMergeSpace:
    threshold = torch.maximum(torch.abs(a - c), torch.abs(b - c))
    dissimilarity = torch.clamp(torch.nan_to_num((c - a) * (b - c) / threshold**2, nan=0), 0)

    return a + (b - c) * alpha * dissimilarity


@convert_to_recipe
def multiply_quotient(
    a: Tensor | SameMergeSpace,
    b: Tensor | SameMergeSpace,
    c: Tensor | SameMergeSpace,
    *,
    alpha: Hyper = 1.0,
    **kwargs,
) -> Tensor | SameMergeSpace:
    ac_log = torch.log(a.abs()) - torch.log(c.abs())
    bc_log = torch.log(b.abs()) - torch.log(c.abs())

    b = torch.complex(b, torch.zeros_like(b))
    c = torch.complex(c, torch.zeros_like(c))

    threshold = torch.maximum(torch.abs(ac_log), torch.abs(bc_log))
    alpha *= torch.clamp(-torch.nan_to_num(ac_log * bc_log / threshold**2, nan=0), 0)

    res = a * (b / c)**alpha
    res = torch.where(torch.isnan(res), a, res)
    del a, b, c
    return res.real


@convert_to_recipe
def distribution_crossover(
    a: Tensor | SameMergeSpace,
    b: Tensor | SameMergeSpace,
    c: Tensor | SameMergeSpace,
    *,
    mean: Hyper,
    tilt: Hyper,
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
    mean: Hyper = 0.5,
    tilt: Hyper = 0.0,
    **kwargs,
) -> Tensor | SameMergeSpace:
    if mean == 0:
        return a
    if tilt == 1:
        return weighted_sum.__wrapped__(a, b, alpha=mean)

    if len(a.shape) == 0 or torch.allclose(a.half(), b.half()):
        return weighted_sum.__wrapped__(a, b, alpha=tilt)

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

    gradients = list(reversed([
        torch.linspace(0, 1, s, device=device)
        if i == 0 or s == 1 else
        # negative frequencies are in the second half of the dimension
        torch.cat([torch.linspace(0, (s - 1) // 2, s - s // 2, device=device), torch.linspace(s // 2, 1, s // 2, device=device)]) / (s // 2)
        for i, s in enumerate(reversed(shape))
    ]))

    if len(shape) > 1:
        grids = torch.meshgrid(*(g**2 for g in gradients), indexing='ij')
        mesh = (torch.stack(grids).sum(dim=0) / len(shape)).sqrt()
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
    alignment: Hyper = 1.0,
    alpha: Hyper = 0.0,
    cache: Optional[Dict[str, Dict[str, Tensor]]] = None,
    **kwargs,
) -> Tensor | SameMergeSpace:
    if alignment == 0 and alpha == 0:
        return a

    if len(a.shape) < 2 or torch.allclose(a.half(), b.half()):
        return weighted_sum.__wrapped__(a, b, alpha=alpha)

    is_conv = len(a.shape) == 4 and a.shape[-1] != 1
    if is_conv:
        shape_2d = (-1, functools.reduce(operator.mul, a.shape[2:]))
    elif len(a.shape) == 4:
        shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    else:
        shape_2d = (-1, a.shape[-1])

    a_neurons = a.reshape(*shape_2d)
    b_neurons = b.reshape(*shape_2d)
    a_centroid = a_neurons.mean(0)
    b_centroid = b_neurons.mean(0)
    a_neurons -= a_centroid
    b_neurons -= b_centroid

    alignment_is_float = alignment != round(alignment)

    if cache is not None:
        key = kwargs["key"]
        if key not in cache:
            cache[key] = {}
        cache = cache[key]

    if cache is not None and "rotation" in cache:
        rotation = transform = cache["rotation"].to(a.device, a.dtype)
    else:
        rotation = transform = orthogonal_procrustes(a_neurons, b_neurons, cancel_reflection=alignment_is_float)
        if cache is not None:
            cache["rotation"] = rotation.to("cpu", torch.float16)

    if alignment_is_float:
        transform = fractional_matrix_power(transform, alignment, cache)
    elif alignment == 0:
        transform = torch.eye(
            len(transform),
            dtype=transform.dtype,
            device=transform.device,
        )
    elif alignment != 1:
        transform = torch.linalg.matrix_power(transform, round(alignment))

    if alpha != 0:
        # interpolate the relationship between the neurons
        a_neurons = weighted_sum.__wrapped__(a_neurons, b_neurons @ rotation.T, alpha=alpha)

    a_neurons @= transform
    a_neurons += weighted_sum.__wrapped__(a_centroid, b_centroid, alpha=alignment)
    return a_neurons.reshape_as(a)


def orthogonal_procrustes(a, b, cancel_reflection: bool = False):
    svd_driver = "gesvd" if a.is_cuda else None
    u, _, v_t = torch.linalg.svd(a.T @ b, driver=svd_driver)

    if cancel_reflection:
        u[:, -1] /= torch.det(u) * torch.det(v_t)

    transform = u @ v_t
    if not torch.isfinite(u).all():
        raise ValueError(
            f"determinant error: {torch.det(transform)}. "
            'This can happen when merging on the CPU with the "rotate" method. '
            "Consider merging on a cuda device, "
            "or try setting alpha to 1 for the problematic blocks. "
            "See this related discussion for more info: "
            "https://github.com/s1dlx/meh/pull/50#discussion_r1429469484"
        )

    return transform


def fractional_matrix_power(matrix: Tensor, power: float, cache: Optional[Dict[str, Tensor]] = None):
    if cache is not None and "eigenvalues" in cache:
        complex_dtype = torch_complex_dtype_map[matrix.dtype]
        eigenvalues = cache["eigenvalues"].to(matrix.device, complex_dtype)
        eigenvectors = cache["eigenvectors"].to(matrix.device, complex_dtype)
        eigenvectors_inv = cache["eigenvectors_inv"].to(matrix.device, complex_dtype)
    else:
        eigenvalues, eigenvectors = torch.linalg.eig(matrix)
        eigenvectors_inv = torch.linalg.inv(eigenvectors)
        if cache is not None:
            cache["eigenvalues"] = eigenvalues.to("cpu", torch.complex32)
            cache["eigenvectors"] = eigenvectors.to("cpu", torch.complex32)
            cache["eigenvectors_inv"] = eigenvectors_inv.to("cpu", torch.complex32)

    eigenvalues.pow_(power)
    result = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors_inv
    return result.real.to(dtype=matrix.dtype)


torch_complex_dtype_map = {
    torch.bfloat16: torch.complex32,
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


@convert_to_recipe
def clip(
    a: Tensor | SameMergeSpace,
    *bounds: Tensor | SameMergeSpace,
    stiffness: Hyper = 0.0,
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


@convert_to_recipe
def dropout(  # aka n-supermario
    delta0: Tensor | LiftFlag[MergeSpace.DELTA],
    *deltas: Tensor | LiftFlag[MergeSpace.DELTA],
    probability: Hyper = 0.9,
    overlap: Hyper = 1.0,
    overlap_emphasis: Hyper = 0.0,
    seed: Hyper = None,
    **kwargs,
) -> Tensor | LiftFlag[MergeSpace.DELTA]:
    deltas = torch.stack((delta0,) + deltas)
    rng = np.random.default_rng(seed)

    if overlap % 2 == 1:
        masks = torch.stack([
            torch.from_numpy(rng.binomial(n=1, p=1 - probability, size=delta0.shape)).to(device=delta0.device, dtype=torch.bool)
            for _ in range(len(deltas))
        ])
    else:
        ks = np.arange(2 ** len(deltas))
        pmf = overlapping_sets_pmf(len(deltas), probability, overlap, overlap_emphasis)
        masks = torch.from_numpy(rng.choice(ks, size=delta0.shape, p=pmf)).to(delta0.device)
        masks = torch.stack([masks & 2 ** i != 0 for i in range(len(deltas))])

    final_delta = torch.zeros_like(delta0)
    for mask, delta in zip(masks, deltas):
        final_delta[mask] += delta[mask]
    return final_delta / masks.sum(0).clamp(1) / (1 - probability)


def overlapping_sets_pmf(n, p, overlap, overlap_emphasis):
    if np.isclose(overlap, round(overlap)):
        if round(overlap) % 2 == 0:
            pmf = np.array([1/n*float(bin(i).count("1") == 1) for i in range(1, 2**n)])
        else:
            pmf = np.array([0 for _ in range(1, 2**n - 1)] + [1])
    else:
        if math.floor(overlap) % 2 == 1:
            overlap = -overlap

        tan_overlap = np.tan(np.pi * (overlap - 0.5))
        pmf = np.zeros(2 ** n - 1)
        for i in range(1, 2 ** n):
            num_sets = bin(i).count("1")
            pmf[i-1] = tan_overlap*(num_sets - n/2)
        pmf = np.exp(pmf) / np.sum(np.exp(pmf))

    binomial_pmf = binom.pmf(np.arange(1, n + 1), n, p)
    expanded_binomial_pmf = np.zeros(2 ** n - 1)
    for i in range(1, 2 ** n):
        num_sets = bin(i).count("1")
        expanded_binomial_pmf[i-1] = binomial_pmf[num_sets-1] / binomial_coefficient_np(n, num_sets)
    expanded_binomial_pmf /= expanded_binomial_pmf.sum()

    pmf = weighted_sum.__wrapped__(
        pmf,
        weighted_sum.__wrapped__(pmf, expanded_binomial_pmf, alpha=1-abs(2*overlap-1)),
        alpha=overlap_emphasis,
    )
    return np.concatenate([[p], pmf * (1 - p)])


def binomial_coefficient_np(n, k):
    if k > n - k:
        k = n - k
    result = np.int64(1)
    for i in range(1, k+1):
        result = result * (n - i + 1) // i
    return result
