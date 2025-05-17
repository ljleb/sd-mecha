import functools
import math
import numpy as np
import torch
from scipy.stats import binom
from torch import Tensor
from typing import Tuple, TypeVar, Sequence
from .svd import orthogonal_procrustes, fractional_matrix_power, torch_svd_lowrank
from .ema import exchange_ema
from sd_mecha.extensions.merge_methods import merge_method, StateDict, Parameter, Return
from sd_mecha.streaming import StateDictKeyError


T = TypeVar("T")


@merge_method
def weighted_sum(
    a: Parameter(StateDict[torch.Tensor]),
    b: Parameter(StateDict[torch.Tensor]),
    alpha: Parameter(Tensor) = 0.5,
    **kwargs,
) -> Return(Tensor):
    key = kwargs["key"]

    if alpha.numel() == 1:
        alpha_float = alpha.item()
        if math.isclose(alpha_float, 0.0):
            return a[key]
        if math.isclose(alpha_float, 1.0):
            return b[key]

    return torch.lerp(a[key], b[key], alpha)


@merge_method
def n_average(
    *models: Parameter(Tensor),
) -> Return(Tensor):
    return torch.mean(torch.stack(models), dim=0)


@merge_method
def slerp(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    *,
    alpha: Parameter(Tensor) = 0.5,
) -> Return(Tensor):
    a_normalized = a / a.norm()
    b_normalized = b / b.norm()

    ab_dot = (a_normalized * b_normalized).sum().clamp(-1, 1)

    omega = torch.arccos(ab_dot)
    a_contrib = a_normalized * torch.sin((1-alpha)*omega)
    b_contrib = b_normalized * torch.sin(alpha*omega)
    res = (a_contrib + b_contrib) / torch.sin(omega)
    res *= torch.lerp(a.norm(), b.norm(), alpha)
    if res.isnan().any():
        return torch.lerp(a, b, alpha)
    return res


@merge_method
def add_difference(
    a: Parameter(StateDict[Tensor], "weight"),
    b: Parameter(StateDict[Tensor], "delta"),
    alpha: Parameter(Tensor) = 1.0,
    **kwargs,
) -> Return(Tensor, "weight"):
    key = kwargs["key"]
    if alpha.numel() == 1 and math.isclose(alpha.item(), 0.0):
        return a[key]

    b_val = b[key]  # try to load b from memory first in case it fails to merge before a
    return a[key].addcmul(b_val, alpha)


@merge_method
def subtract(
    a: Parameter(Tensor, "weight"),
    b: Parameter(Tensor, "weight"),
) -> Return(Tensor, "delta"):
    return a - b


@merge_method
def perpendicular_component(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
) -> Return(Tensor):
    norm_a = torch.linalg.norm(a)
    res = b - a * (a / norm_a * (b / norm_a)).sum()
    if res.isnan().any():
        return torch.zeros_like(a)
    return res


@merge_method
def geometric_sum(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    alpha: Parameter(Tensor) = 0.5,
) -> Return(Tensor):
    a = torch.complex(a, torch.zeros_like(a))
    b = torch.complex(b, torch.zeros_like(b))
    res = a ** (1 - alpha) * b ** alpha
    return res.real


@merge_method
def add_cosine_a(
    a: Parameter(Tensor, "weight"),
    b: Parameter(Tensor, "weight"),
    alpha: Parameter(Tensor) = 1.0,
) -> Return(Tensor, "weight"):
    a_norm = torch.nn.functional.normalize(a, dim=0)
    b_norm = torch.nn.functional.normalize(b, dim=0)
    similarity = torch.nn.functional.cosine_similarity(a_norm, b_norm, dim=0)
    return add_cosine_generic(a, b, alpha, similarity)


@merge_method
def add_cosine_b(
    a: Parameter(Tensor, "weight"),
    b: Parameter(Tensor, "weight"),
    alpha: Parameter(Tensor) = 1.0,
) -> Return(Tensor, "weight"):
    similarity = torch.nn.functional.cosine_similarity(a, b, dim=0)
    dot_product = torch.sum(a * b)
    magnitude_similarity = dot_product / (torch.norm(a) * torch.norm(b))
    combined_similarity = (similarity + magnitude_similarity) / 2.0
    return add_cosine_generic(a, b, alpha, combined_similarity)


def add_cosine_generic(a: Tensor, b: Tensor, alpha: Tensor, similarity: Tensor) -> Tensor:
    k = 1 - torch.clamp(similarity - alpha, 0, 1)
    return torch.lerp(a, b, k)


@merge_method
def tensor_sum(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    width: Parameter(float) = 0.5,
    offset: Parameter(float) = 0.0,
) -> Return(Tensor):
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


@merge_method
def top_k_tensor_sum(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    width: Parameter(float) = 1.0,
    offset: Parameter(float) = 0.0,
) -> Return(Tensor):
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


@merge_method
def train_difference_mask(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    c: Parameter(Tensor),
    alpha: Parameter(Tensor) = 1.0,
) -> Return(Tensor, "param"):
    return alpha * 1.8 * torch.nan_to_num((b - a).abs() / ((b - a).abs() + (b - c).abs()), nan=0)


@merge_method
def add_opposite_mask(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    c: Parameter(Tensor),
    alpha: Parameter(Tensor) = 1.0,
) -> Return(Tensor, "param"):
    return alpha * 2 * torch.nan_to_num((a - b).abs() / ((a - b).abs() + (a + b - 2*c).abs()), nan=0)


@merge_method
def add_strict_opposite_mask(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    c: Parameter(Tensor),
    alpha: Parameter(Tensor) = 1.0,
) -> Return(Tensor, "param"):
    threshold = torch.maximum(torch.abs(a - c), torch.abs(b - c))
    return alpha * torch.clamp(torch.nan_to_num((c - a) * (b - c) / threshold**2, nan=0), 0) * 2


@merge_method
def select_max_delta(
    a: Parameter(Tensor, "delta"),
    b: Parameter(Tensor, "delta"),
    alpha: Parameter(Tensor) = 0.5,
) -> Return(Tensor, "delta"):
    a_abs = ((a - a.mean()) / a.std()).nan_to_num(nan=0).abs()
    b_abs = ((b - b.mean()) / b.std()).nan_to_num(nan=0).abs()
    return torch.where((1 - alpha) * a_abs >= alpha * b_abs, a, b)


@merge_method
def select_max_white_delta(
    a: Parameter(Tensor, "delta"),
    b: Parameter(Tensor, "delta"),
    alpha: Parameter(Tensor) = 0.5,
) -> Return(Tensor, "delta"):
    a_norm = (a - a.mean()) / a.std(correction=0)
    b_norm = (b - b.mean()) / b.std(correction=0)

    d = torch.stack((a_norm.flatten(), b_norm.flatten()), dim=-1)
    v, vs = torch.linalg.eigh(d.cov(correction=0))
    w = vs @ torch.diag_embed(1 / v.sqrt()) @ vs.mH
    a_w, b_w = (w @ d).T

    return torch.where((1 - alpha) * a_w.abs() >= alpha * b_w.abs(), a, b)


@merge_method
def multiply_quotient(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    c: Parameter(Tensor),
    alpha: Parameter(Tensor) = 1.0,
) -> Return(Tensor):
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


@merge_method
def rotate(
    a_dict: Parameter(Tensor),
    b_dict: Parameter(Tensor),
    alignment: Parameter(float) = 1.0,
    alpha: Parameter(Tensor) = 0.0,
    **kwargs,
) -> Return(Tensor):
    key = kwargs["key"]
    if math.isclose(alignment, 0) and torch.allclose(alpha, torch.zeros_like(alpha)):
        return a_dict[key]

    if math.isclose(alignment, 1) and torch.allclose(alpha, torch.ones_like(alpha)):
        return b_dict[key]

    a = a_dict[key]
    b = b_dict[key]
    if len(a.shape) <= 1 or torch.allclose(a.half(), b.half()):
        return (1-alpha)*a + alpha*b

    is_conv = len(a.shape) == 4 and a.shape[-2:].numel() != 1
    if is_conv:
        shape_2d = a.shape[:2].numel(), a.shape[2:].numel()
    else:
        shape_2d = a.shape[:1].numel(), a.shape[1:].numel()

    a_neurons = a.reshape(*shape_2d)
    b_neurons = b.reshape(*shape_2d)
    a_centroid = a_neurons.mean(0)
    b_centroid = b_neurons.mean(0)
    a_neurons -= a_centroid
    b_neurons -= b_centroid

    cache = kwargs.get("cache")
    if cache is not None:
        key = kwargs["key"]
        if key not in cache:
            cache[key] = {}
        cache = cache[key]

    alignment_is_float = math.isclose(alignment, round(alignment))

    if cache is not None and "rotation" in cache:
        rotation = transform = cache["rotation"].to(a.device, a.dtype)
    else:
        rotation = transform = orthogonal_procrustes(a_neurons, b_neurons, cancel_reflection=alignment_is_float)
        if cache is not None:
            cache["rotation"] = rotation.to("cpu", torch.float16)

    if alignment_is_float:
        transform = fractional_matrix_power(transform, alignment, cache)
    elif math.isclose(alignment, 0):
        transform = torch.eye(
            len(transform),
            dtype=transform.dtype,
            device=transform.device,
        )
    elif not math.isclose(alignment, 1):
        transform = torch.linalg.matrix_power(transform, round(alignment))

    if not torch.allclose(alpha, torch.zeros_like(alpha)):
        # interpolate the relationship between the neurons
        a_neurons = (1-alpha)*a_neurons + alpha*b_neurons@rotation.T

    a_neurons @= transform
    a_neurons += (1-alignment)*a_centroid + alignment*b_centroid
    return a_neurons.reshape_as(a)


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


@merge_method
def copy_distribution(
    a: Parameter(Tensor),
) -> Return(Tensor):
    return torch.randn_like(a) * a.std(correction=0) + a.mean()


# Special mode "TIES-STOCK" has been implemented by setting `apply_stock` > 0.0
# Special mode "TIES-GMEDIAN" has been implemented by setting `apply_median` > 0.0
@merge_method
def ties_sum_extended(  # aka add_difference_ties
    *models: Parameter(Tensor, "delta"),
    k: Parameter(float) = 0.2,
    vote_sgn: Parameter(bool) = False,
    apply_stock: Parameter(bool) = False,
    apply_median: Parameter(bool) = False,
    cos_eps: Parameter(float) = 1e-6,
    eps: Parameter(float) = 1e-6,
    maxiter: Parameter(int) = 100,
    ftol: Parameter(float) = 1e-20,
) -> Return(Tensor, "delta"):
    filtered_delta, param_counts = ties_sum_deltas(*models, k=k, vote_sgn=vote_sgn)

    if apply_median:
        # Model Stock
        t = 1.0 if apply_stock else get_model_stock_t(torch.unbind(filtered_delta), cos_eps=cos_eps)

        filtered_delta = filtered_delta.sum(dim=0)

        # $$ \tau_m $$
        return torch.nan_to_num(filtered_delta * t / param_counts)
    else:
        # $$ \tau_m $$, but in geometric median instead of arithmetic mean. Considered to replace model stock.
        filtered_delta = geometric_median.__wrapped__(*torch.unbind(filtered_delta), eps=eps, maxiter=maxiter, ftol=ftol)

        return torch.nan_to_num(filtered_delta)


# latex notes in reference to original implementation: https://arxiv.org/abs/2306.01708
# - `delta`: $$ \hat{\tau}_t $$
# - `signs`: $$ \gamma_t $$
# - `final_sign`: $$ \gamma_m^p = sgn(\Sigma_{t=1}^n \hat{\tau}_t^p) $$
# - `delta_filters`: $$ \{ \gamma_t^p = \gamma_m^p \} $$
# - `param_counts`: $$ |A^p| $$
# - `filtered_delta`: $$ \Sigma_{t\in{A^p}} \hat{\tau}_t^p $$
# - `return`: $$ \lambda * \tau_m $$
# Special mode "TIES-SOUP" has been implemented by setting `vote_sgn` > 0.0
# - `final_sign`: $$ \gamma_m^p = sgn(\Sigma_{t=1}^n \gamma_t^p) $$
@merge_method
def ties_sum(  # aka add_difference_ties
    *models: Parameter(Tensor, "delta"),
    k: Parameter(float) = 1.0,
    vote_sgn: Parameter(bool) = False,
) -> Return(Tensor, "delta"):
    filtered_delta, param_counts = ties_sum_deltas(*models, k=k, vote_sgn=vote_sgn)

    # $$ \tau_m $$
    return torch.nan_to_num(filtered_delta.sum(dim=0) / param_counts)


def ties_sum_deltas(
    *models: Tensor,
    k: float = 0.2,
    vote_sgn: bool = False,
):
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

    # $$ \gamma_m^p = sgn(\Sigma_{t=1}^n \hat{\tau}_t^p) $$ for normal TIES
    # $$ \gamma_m^p = sgn(\Sigma_{t=1}^n \gamma_t^p) $$ if "TIES-SOUP" is activated
    final_sign = torch.sign(torch.sum(deltas if vote_sgn else signs, dim=0))

    # Step 3: Disjoint merge.

    # $$ \{ \gamma_t^p = \gamma_m^p \} $$
    delta_filters = (signs == final_sign).float()

    # $$ |A^p| $$
    param_counts = torch.sum(delta_filters, dim=0)

    # $$ \Sigma_{t\in{A^P}} \hat{\tau}_t^p $$
    # (note that the sum is not performed here directly)
    filtered_delta = deltas * delta_filters

    return filtered_delta, param_counts


def filter_top_k(a: Tensor, k: float):
    k = max(int((1 - k) * a.numel()), 1)
    k_value, _ = a.flatten().abs().float().kthvalue(k)
    top_k_filter = (a.abs() >= k_value).float()
    return a * top_k_filter


@merge_method
def dropout(  # aka n-supermario
    *deltas: Parameter(Tensor, "delta"),
    probability: Parameter(float) = 0.9,
    rescale: Parameter(float) = 1.0,
    overlap: Parameter(float) = 1.0,
    overlap_emphasis: Parameter(float) = 0.0,
    seed: Parameter(int) = None,
) -> Return(Tensor, "delta"):
    if len(deltas) == 0:
        return 0

    delta0 = deltas[0]
    deltas = torch.stack(deltas)
    rng = np.random.default_rng(seed)

    if overlap % 2 == 1:
        masks = tuple(
            torch.from_numpy(rng.binomial(n=1, p=1 - probability, size=delta0.shape)).to(device=delta0.device, dtype=torch.bool)
            for _ in range(len(deltas))
        )
    else:
        ks = np.arange(2 ** len(deltas))
        pmf = overlapping_sets_pmf(len(deltas), probability, overlap, overlap_emphasis)
        masks = torch.from_numpy(rng.choice(ks, size=delta0.shape, p=pmf)).to(delta0.device)
        masks = torch.stack([masks & 2 ** i != 0 for i in range(len(deltas))])

    final_delta = torch.zeros_like(delta0)
    for mask, delta in zip(masks, deltas):
        final_delta[mask] += delta[mask]

    if probability == 1.0:
        rescalar = 1.0
    else:
        rescalar = (1.0 - probability) ** rescale
        rescalar = rescalar if math.isfinite(rescalar) else 1
    return final_delta / masks.sum(0).clamp(1) / rescalar


def overlapping_sets_pmf(n, p, overlap: float, overlap_emphasis):
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

    pmf = torch.lerp(
        pmf,
        torch.lerp(pmf, expanded_binomial_pmf, 1-abs(2*overlap-1)),
        overlap_emphasis,
    )
    return np.concatenate([[p], pmf * (1 - p)])


# Part of TIES w/ DARE
# Hyperparameters defauled to values proposed to paper.
# Special mode "DROP" has been implemented by setting `no_rescale` > 0.0
# - `return`: $$ \hat{\delta}^t = \tilde{\delta}^t $$
@merge_method
def ties_sum_with_dropout(
    *deltas: Parameter(Tensor, "delta"),
    probability: Parameter(float) = 0.9,
    no_rescale: Parameter(bool) = False,
    k: Parameter(float) = 0.2,
    vote_sgn: Parameter(bool) = False,
    apply_stock: Parameter(bool) = False,
    cos_eps: Parameter(float) = 1e-6,
    apply_median: Parameter(bool) = False,
    eps: Parameter(float) = 1e-6,
    maxiter: Parameter(int) = 100,
    ftol: Parameter(float) = 1e-20,
    seed: Parameter(int) = None,
) -> Return(Tensor, "delta"):
    # Set seed
    generator = torch.Generator(device=deltas[0].device)
    if seed is not None:
        generator.manual_seed(seed)

    # Under "Dropout", delta will be 0 by definition. Multiply it (Hadamard product) will return 0 also.
    # $$ \tilde{\delta}^t = (1 - m^t) \odot \delta^t $$
    deltas = tuple(
        delta * torch.bernoulli(torch.full(delta.shape, 1 - probability, device=deltas[0].device, dtype=deltas[0].dtype), generator=generator)
        for delta in deltas
    )

    # $$ \tilde{\delta}^t = \tau_m = \hat{\tau}_t $$ O(N) in space
    deltas = ties_sum_extended.__wrapped__(
        *deltas,
        k=k,
        vote_sgn=vote_sgn,
        apply_stock=apply_stock,
        cos_eps=cos_eps,
        apply_median=apply_median,
        eps=eps,
        maxiter=maxiter,
        ftol=ftol,
    )

    if math.isclose(probability, 1.0):
        # Corner case
        return deltas * 0.0
    elif no_rescale:
        # Rescale
        # $$ \hat{\delta}^t = \tilde{\delta}^t / (1-p) $$
        return deltas / (1.0 - probability) 
    else:
        # No rescale
        # $$ \hat{\delta}^t = \tilde{\delta}^t $$
        return deltas


def binomial_coefficient_np(n, k):
    if k > n - k:
        k = n - k
    result = np.int64(1)
    for i in range(1, k+1):
        result = result * (n - i + 1) // i
    return result


# src: https://github.com/arcee-ai/mergekit/blob/main/mergekit/merge_methods/model_stock.py
@merge_method
def model_stock(
    *deltas: Parameter(Tensor, "delta"),
    cos_eps: Parameter(float) = 1e-6,
) -> Return(Tensor, "delta"):
    w_avg = n_average.__wrapped__(*deltas)

    # t can get inf so handle with care
    t = get_model_stock_t(deltas, cos_eps)

    # return w_h. Notice that w_0 is 0 here.
    return torch.nan_to_num(t * w_avg)


# The guess from mergekit: Average of cos(theta). Expected value is 0, somehow match with paper.
# However this may be very unstable, and the range is still -1 to 1.
def get_model_stock_t(deltas: Sequence[Tensor], cos_eps: float):
    n = len(deltas)

    # Generator function
    cos = torch.nn.CosineSimilarity(dim=-1, eps=cos_eps)
    cos_thetas = [cos(deltas[i], deltas[i + 1]) for i, _ in enumerate(deltas) if (i + 1) < n]

    # Still a vector
    cos_theta = torch.stack(cos_thetas).mean(dim=0)

    # Convert to column vector for multiplication
    t = (n * cos_theta / (1 + (n - 1) * cos_theta)).unsqueeze(-1)
    return t


# src: https://github.com/krishnap25/geom_median/blob/main/src/geom_median/torch/weiszfeld_list_of_array.py
@merge_method
def geometric_median(
    *models: Parameter(Tensor),
    eps: Parameter(float) = 1e-6,
    maxiter: Parameter(int) = 100,
    ftol: Parameter(float) = 1e-20,
) -> Return(Tensor):
    weights = torch.ones(len(models), device=models[0].device)

    # initialize median estimate at mean
    median = weighted_average_tuple(models, weights)
    new_weights = weights
    objective_value = geometric_median_objective(median, models, weights)

    # Weiszfeld iterations
    for _ in range(max(0, maxiter)):
        prev_obj_value = objective_value
        denom = torch.stack([torch.dist(p, median) for p in models])
        new_weights = weights / torch.clamp(denom, min=eps)
        median = weighted_average_tuple(models, new_weights)

        objective_value = geometric_median_objective(median, models, weights)
        if abs(prev_obj_value - objective_value) <= ftol * objective_value:
            break

    return weighted_average_tuple(models, new_weights)


def weighted_average_tuple(points: Tuple, weights):
    return torch.sum(torch.stack([p * weights[i] for i, p in enumerate(points)]), dim=0) / weights.sum()


def geometric_median_objective(median, points: Tuple, weights):
    return torch.mean(torch.stack([torch.dist(point, median) for point in points]) * weights)


@merge_method
def truncate_rank(
    a: Parameter(Tensor, merge_space="delta"),
    rank_ratio: Parameter(float) = 0.5,
) -> Return(Tensor, merge_space="delta"):
    if a.dim() < 2:
        return a

    original_shape = a.shape
    a_2d = a.flatten(start_dim=1)
    max_rank = min(a_2d.shape)
    target_rank = min(max(round(max_rank * rank_ratio), 0), max_rank)
    if target_rank == max_rank:
        return a
    if target_rank == 0:
        return torch.zeros_like(a)

    svd_driver = "gesvda" if a.is_cuda else None
    u, s, vt = torch_svd_lowrank(a_2d, q=target_rank, full_matrices=False, driver=svd_driver)
    return (u @ torch.diag(s) @ vt).reshape(original_shape)


@merge_method
def fallback(
    a: Parameter(StateDict[T]),
    default: Parameter(StateDict[T]),
    **kwargs,
) -> Return(T):
    key = kwargs["key"]
    try:
        return a[key]
    except StateDictKeyError:
        return default[key]


@merge_method
def cast(
    a: Parameter(Tensor),
    device: Parameter(str) = None,
    dtype: Parameter(str) = None,
) -> Return(Tensor):
    to_kwargs = {}
    if device is not None:
        to_kwargs["device"] = device

    if dtype is not None:
        if dtype not in cast_dtype_map:
            raise ValueError(f"Unknown dtype {dtype}. Possible values are None, {', '.join(cast_dtype_map.keys())}")
        to_kwargs["dtype"] = cast_dtype_map[dtype]

    return a.to(**to_kwargs)


cast_dtype_map = {
    "float64": torch.float64,
    "int64": torch.int64,
    "float32": torch.float32,
    "int32": torch.int32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int16": torch.int16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "int8": torch.int8,
    "bool": torch.bool,
}
if hasattr(torch, "uint8"):
    cast_dtype_map |= {
        "uint64": torch.uint64,
        "uint32": torch.uint32,
        "uint16": torch.uint16,
        "uint8": torch.uint8,
    }
cast_dtype_map_reversed = {v: k for k, v in cast_dtype_map.items()}


@merge_method
def get_dtype(
    a: Parameter(Tensor),
) -> Return(str, "param"):
    return cast_dtype_map_reversed[a.dtype]


@merge_method
def get_device(
    a: Parameter(Tensor),
) -> Return(str, "param"):
    return str(a.device)


@merge_method
def pick_component(
    a: Parameter(StateDict[T]),
    component: Parameter(str, "param"),
    **kwargs,
) -> Return(T):
    if component not in a.model_config.components():
        raise ValueError(
            f'Component "{component}" does not exist in config "{a.model_config.identifier}". '
            f"Valid components: {tuple(a.model_config.components())}"
        )

    key = kwargs["key"]
    if key in a.model_config.components()[component].keys():
        return a[key]
    else:
        raise StateDictKeyError(key)


@merge_method
def omit_component(
    a: Parameter(StateDict[T]),
    component: Parameter(str, "param"),
    **kwargs,
) -> Return(T):
    if component not in a.model_config.components():
        raise ValueError(
            f'Component "{component}" does not exist in config "{a.model_config.identifier}". '
            f"Valid components: {tuple(a.model_config.components())}"
        )

    key = kwargs["key"]
    if key in a.model_config.components()[component].keys():
        raise StateDictKeyError(key)
    else:
        return a[key]
