import math
import torch
from collections import defaultdict
from lycoris.functional import factorization as lycoris_factorization
from torch import Tensor
from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return
from sd_mecha.extensions.builtin.merge_methods.svd import svd_lowrank
from typing import Optional, Tuple


@merge_method(cache_factory=lambda: defaultdict(dict))
def truncate_kronecker(
    a: Parameter(Tensor, "delta"),
    dims: Parameter(Tensor, "param") = ((1, 1), (1, 1)),
    use_approximate_basis: Parameter(bool) = True,
    approximate_basis_iters: Parameter(int) = 2,
    approximate_basis_seed: Parameter(int) = None,
    **kwargs,
) -> Return(Tensor, "delta"):
    if a.dim() < 2:
        return a

    cache = kwargs.get("cache")
    if cache is not None:
        cache = cache[kwargs["key"]]

    w1, w2 = extract_lokr(
        a,
        dims,
        use_approximate_basis,
        approximate_basis_iters,
        approximate_basis_seed,
        cache,
    )

    res = torch.kron(w1, w2).reshape(a.shape)
    return res


def extract_lokr(
    a: torch.Tensor,
    dims: torch.Tensor,
    use_approximate_basis: bool = False,
    approximate_basis_iters: int = 0,
    approximate_basis_seed: Optional[int] = None,
    cache: Optional[dict] = None,
):
    shape_original = a.shape
    lokr_dims = solve_lokr_dims(shape_original, dims.to(a.device))

    if (
        cache is not None
        and cache.get("lokr_dims") == lokr_dims
        and cache.get("iters") == approximate_basis_iters
        and cache.get("seed") == approximate_basis_seed
        and "w1" in cache
        and "w2" in cache
    ):
        w1 = cache["w1"].to(a)
        w2 = cache["w2"].to(a)
    else:
        m1, m2, n1, n2 = lokr_dims
        shape_extra = shape_original[2:]
        shape_w1 = torch.Size((m1, n1))
        shape_w2 = torch.Size((m2, n2, *shape_extra))
        p2 = shape_extra.numel()

        a_2d = a.reshape(m1, m2, n1, n2 * p2).permute(0, 2, 1, 3).reshape(shape_w1.numel(), shape_w2.numel())

        svd_driver = "gesvda" if a.is_cuda else None
        if use_approximate_basis:
            u, s, vh = svd_lowrank(a_2d, rank=1, iters=approximate_basis_iters, seed=approximate_basis_seed, driver=svd_driver)
        else:
            u, s, vh = torch.linalg.svd(a_2d, full_matrices=False, driver=svd_driver)

        s = s[..., 0].sqrt()
        w1 = u[..., :, 0] * s
        w2 = s * vh[..., 0, :]

        w1 = w1.reshape(shape_w1)
        w2 = w2.reshape(shape_w2)

        while w1.dim() < w2.dim():
            w1 = w1.unsqueeze(-1)

        if cache is not None:
            cache["lokr_dims"] = lokr_dims
            cache["w1"] = w1.to(device="cpu", dtype=torch.float16)
            cache["w2"] = w2.to(device="cpu", dtype=torch.float16)
            if use_approximate_basis:
                cache["iters"] = approximate_basis_iters
                cache["seed"] = approximate_basis_seed
            else:
                cache.pop("iters", None)
                cache.pop("seed", None)

    return w1, w2


def solve_lokr_dims(
    shape: torch.Size | tuple[int, ...],
    target: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[int, int, int, int]:
    shape = torch.Size(shape)
    if len(shape) < 2:
        raise ValueError(f"expected shape with at least 2 dims, got {tuple(shape)}")

    dims_2d = target.new_tensor(shape[:2], dtype=torch.int64)
    out_dim = shape[0]
    in_dim = shape[1]

    if target.shape != (2, 2):
        raise ValueError(f"expected 2x2 preference ((m1, m2), (n1, n2)), got shape {tuple(target.shape)}")

    if not target.is_floating_point() and (target.prod(dim=-1) == dims_2d).all():
        return tuple(target.flatten().tolist())

    target_log = target.clamp_min(eps).log()
    s_log = (dims_2d.log() - target_log.sum(dim=-1)) * 0.5
    target_log = target_log + s_log.unsqueeze(-1)

    out_pairs = divisor_pairs(out_dim, device=target.device)
    in_pairs = divisor_pairs(in_dim, device=target.device)

    out_grid = out_pairs[:, None].expand(-1, in_pairs.shape[0], -1)  # [out_divs, in_divs, 2]
    in_grid = in_pairs[None, :].expand(out_pairs.shape[0], -1, -1)   # [out_divs, in_divs, 2]

    candidates = torch.stack((out_grid, in_grid), dim=-2)  # [out_divs, in_divs, 2, 2]
    candidates_log = candidates.log()

    cost = (candidates_log - target_log).square().sum(dim=(-1, -2))  # [out_divs, in_divs]
    balance = candidates_log.diff(dim=-1).abs().sum(dim=(-1, -2))  # Tie-break 1: prefer balanced splits.
    numel = candidates.prod(dim=-1).sum(dim=-1)  # Tie-break 2: prefer smaller pair size.

    mask = torch.isclose(cost, cost.min())

    masked_balance = torch.where(mask, balance, torch.inf)
    best_balance = masked_balance.min()
    mask = mask & (balance == best_balance)

    flat_idx = torch.where(mask, numel, torch.inf).argmin()
    i, j = torch.unravel_index(flat_idx, numel.shape)
    best = candidates[i, j].long()
    return tuple(best.flatten().tolist())


@merge_method
def ratio_lokr_dims(
    a: Parameter(torch.Tensor),
    ratio: Parameter(float, "param") = 0.5,
) -> Return(torch.Tensor, "param"):
    shape = a.shape
    if len(shape) < 2:
        raise ValueError(f"expected shape with at least 2 dims, got {tuple(shape)}")

    out_dim = int(shape[0])
    in_dim = int(shape[1])

    if out_dim <= 0 or in_dim <= 0:
        return a.new_tensor(((1, 1), (1, 1)))

    m_pairs = divisor_pairs(out_dim, device=a.device).to(torch.float64)  # [out_divs, 2]
    n_pairs = divisor_pairs(in_dim, device=a.device).to(torch.float64)   # [in_divs, 2]

    m_grid = m_pairs[:, None, :].expand(-1, n_pairs.shape[0], -1)  # [out_divs, in_divs, 2]
    n_grid = n_pairs[None, :, :].expand(m_pairs.shape[0], -1, -1)  # [out_divs, in_divs, 2]
    candidates = torch.stack((m_grid, n_grid), dim=-2)      # [out_divs, in_divs, 2, 2]
    candidates_log = candidates.log()

    log_a = candidates_log[..., 0, 0] + candidates_log[..., 1, 0]  # m1 * n1

    log_sqrt_numel = 0.5 * math.log(shape.numel())
    s = log_a - log_sqrt_numel  # [out_divs, in_divs]
    balance = candidates_log.diff(dim=-1).abs().sum(dim=(-1, -2))  # [out_divs, in_divs]

    s_min = s.min()
    s_max = s.max()
    s_range = torch.maximum(s_min.abs(), s_max.abs())

    if s_range.item() == 0.0:
        flat_idx = balance.argmin()
        i, j = torch.unravel_index(flat_idx, balance.shape)
        return candidates[i, j].to(torch.int64)

    ratio = min(max(ratio, 0.0), 1.0)
    s_target = (2.0 * ratio - 1.0) * s_range

    # Primary objective: abs(s - s_target)
    s_cost = (s - s_target).abs()
    mask = torch.is_close(s_cost, s_cost.min())

    # Tie-break: balance
    masked_balance = torch.where(mask, balance, torch.inf)
    flat_idx = masked_balance.argmin()
    i, j = torch.unravel_index(flat_idx, balance.shape)

    return candidates[i, j].to(torch.int64)


@merge_method
def lycoris_compatible_lokr_dims(
    a: Parameter(torch.Tensor),
    factor: Parameter(int) = -1,
    unbalanced_factorization: Parameter(bool) = False,
) -> Return(torch.Tensor, "param"):
    shape = a.shape
    if len(shape) < 2:
        raise ValueError(f"expected shape with at least 2 dims, got {tuple(shape)}")

    out_dim = shape[0]
    in_dim = shape[1]

    out_l, out_k = lycoris_factorization(out_dim, factor)
    in_m, in_n = lycoris_factorization(in_dim, factor)

    if unbalanced_factorization:
        out_l, out_k = out_k, out_l

    return a.new_tensor(((out_l, out_k), (in_m, in_n)), dtype=torch.int64)


def divisor_pairs(n: int, device: torch.device = None) -> torch.Tensor:
    small = []
    large = []

    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d != 0:
            continue

        q = n // d
        small.append((d, q))
        if d != q:
            large.append((q, d))

    pairs = small + large[::-1]
    return torch.tensor(pairs, device=device, dtype=torch.int64)
