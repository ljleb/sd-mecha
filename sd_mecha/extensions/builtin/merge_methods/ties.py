import math
import numpy as np
import torch
from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return
from typing import Tuple, Sequence, Optional
from scipy.stats import binom, rankdata
from torch import Tensor


@merge_method
def ties_sum_with_dropout(
    *deltas: Parameter(Tensor, "delta"),
    probability: Parameter(Tensor) = 0.9,
    della_eps: Parameter(float) = 0.0,
    rescale: Parameter(bool) = True,
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
    if not deltas or math.isclose(probability, 1.0):
        return 0

    generator = torch.Generator(deltas[0].device)
    if seed is not None and seed >= 0:
        generator.manual_seed(seed)

    deltas = [delta * find_della_dropout(delta, probability, della_eps, generator) for delta in deltas]
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

    if math.isclose(probability, 1.0) or not rescale:
        rescalar = 1.0
    else:
        rescalar = 1.0 - probability
    return deltas / rescalar


def find_della_dropout(delta: Tensor, probability: Tensor, della_eps: float, generator: torch.Generator):
    if not math.isclose(della_eps, 0.0):
        rank_per_element = torch.from_numpy(rankdata(delta.abs().numpy(force=True), method="ordinal").reshape(delta.shape)).to(device=delta.device)
        ne = delta.numel()
        # center window
        delta_i = (rank_per_element/ne - (ne + 1)/(ne * 2)) * della_eps
        delta_i = delta_i.nan_to_num(nan=0, posinf=0, neginf=0)
    else:
        delta_i = 0.0

    p_min = torch.ones_like(delta) - probability
    res = torch.bernoulli((p_min + delta_i).clamp(min=0, max=1), generator=generator)
    return res


@merge_method
def ties_sum_extended(
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
    if not models:
        return 0
    if models[0].numel() == 0:
        return models[0]

    filtered_delta, param_counts = ties_sum_deltas(*models, k=k, vote_sgn=vote_sgn)

    if apply_median:
        filtered_delta = geometric_median.__wrapped__(*filtered_delta, eps=eps, maxiter=maxiter, ftol=ftol)
    else:
        t = 1.0 if apply_stock else get_model_stock_t(filtered_delta, cos_eps=cos_eps)
        filtered_delta = filtered_delta.sum(dim=0)
        filtered_delta = filtered_delta * t / param_counts

    return torch.nan_to_num(filtered_delta, nan=0, posinf=0, neginf=0)


# src: https://arxiv.org/abs/2306.01708
@merge_method
def ties_sum(
    *models: Parameter(Tensor, "delta"),
    k: Parameter(float) = 1.0,
    vote_sgn: Parameter(bool) = False,
) -> Return(Tensor, "delta"):
    if not models:
        return 0
    if models[0].numel() == 0:
        return models[0]

    filtered_delta, param_counts = ties_sum_deltas(*models, k=k, vote_sgn=vote_sgn)
    return torch.nan_to_num(filtered_delta.sum(dim=0) / param_counts, nan=0, posinf=0, neginf=0)


def ties_sum_deltas(
    *models: Tensor,
    k: float = 0.2,
    vote_sgn: bool = False,
):
    deltas = torch.stack([filter_top_k(m, k) for m in models], dim=0)
    signs = torch.sign(deltas)
    final_sign = torch.sign(torch.sum(deltas if vote_sgn else signs, dim=0))

    delta_filters = (signs == final_sign).float()
    filtered_delta = deltas * delta_filters
    param_counts = torch.sum(delta_filters, dim=0)
    return filtered_delta, param_counts


def filter_top_k(a: Tensor, k: float):
    k = max(int((1 - k) * a.numel()), 1)
    k_value, _ = a.flatten().abs().float().kthvalue(k)
    top_k_filter = (a.abs() >= k_value).float()
    return a * top_k_filter


# src: https://github.com/arcee-ai/mergekit/blob/main/mergekit/merge_methods/model_stock.py
@merge_method
def model_stock(
    *deltas: Parameter(Tensor, "delta"),
    cos_eps: Parameter(float) = 1e-6,
) -> Return(Tensor, "delta"):
    w_avg = sum(deltas) / len(deltas)
    t = get_model_stock_t(deltas, cos_eps)
    return torch.nan_to_num(t * w_avg)


def get_model_stock_t(deltas: Sequence[Tensor], cos_eps: float):
    """
    Approximate solution from mergekit: average of cos(theta).
    The expected value is 0, which accidentally corresponds with the implementation from the paper.
    This may be very unstable and the range is restricted to [-1, 1].
    """
    n = len(deltas)

    cos = torch.nn.CosineSimilarity(dim=-1, eps=cos_eps)
    cos_thetas = [cos(deltas[i], deltas[i + 1]) for i, _ in enumerate(deltas) if (i + 1) < n]
    cos_theta = torch.stack(cos_thetas).mean(dim=0)

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
    median = weighted_average(models)
    weights = new_weights = torch.ones(len(models), device=models[0].device, dtype=models[0].dtype)
    objective_value = geometric_median_objective(median, models, weights)

    # Weiszfeld iterations
    for _ in range(max(0, maxiter)):
        prev_obj_value = objective_value
        denom = torch.stack([torch.dist(p, median) for p in models])
        new_weights = weights / torch.clamp(denom, min=eps)
        median = weighted_average(models, new_weights)

        objective_value = geometric_median_objective(median, models, weights)
        if abs(prev_obj_value - objective_value) <= ftol * objective_value:
            break

    return weighted_average(models, new_weights)


def weighted_average(
    points: Sequence[float | Tensor] | Tensor,
    weights: Optional[Sequence[float | Tensor] | Tensor] = None
) -> float | Tensor:
    if weights is not None:
        return sum(p * weights[i] for i, p in enumerate(points)) / sum(weights)
    else:
        return sum(points) / len(points)


def geometric_median_objective(median, points: Tuple, weights):
    return torch.mean(torch.stack([torch.dist(point, median) for point in points]) * weights)


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
    return final_delta / sum(tuple(masks)).clamp(1) / rescalar


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


def binomial_coefficient_np(n, k):
    if k > n - k:
        k = n - k
    result = np.int64(1)
    for i in range(1, k+1):
        result = result * (n - i + 1) // i
    return result
