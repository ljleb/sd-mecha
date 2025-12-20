import math
import torch
from torch import Tensor
from sd_mecha import merge_method, Parameter, Return
from sd_mecha.extensions.builtin.merge_methods.svd import svd_lowrank


@merge_method
def truncate_kronecker(
    a: Parameter(Tensor, merge_space="delta"),
    kronecker_ratio: Parameter(float) = 0.5,
    use_approximate_basis: Parameter(bool) = True,
    approximate_basis_iters: Parameter(int) = 2,
    approximate_basis_seed: Parameter(int) = None,
    **kwargs,
) -> Return(Tensor, merge_space="delta"):
    if a.dim() < 2:
        return a

    cache = kwargs.get("cache")
    if cache is not None:
        key = kwargs["key"]
        if key not in cache:
            cache[key] = {}
        cache = cache[key]

    shape_original = a.shape
    kron_dims = kron_dims_from_ratio(shape_original, kronecker_ratio)

    if (
        cache is not None
        and cache.get("kron_dims") == kron_dims
        and cache.get("iters") == approximate_basis_iters
        and cache.get("seed") == approximate_basis_seed
        and "w1" in cache
        and "w2" in cache
    ):
        w1 = cache["w1"].to(a)
        w2 = cache["w2"].to(a)
    else:
        m1, m2, n1, n2 = kron_dims
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
            cache["kron_dims"] = kron_dims
            cache["w1"] = w1.to(device="cpu", dtype=torch.float16)
            cache["w2"] = w2.to(device="cpu", dtype=torch.float16)
            if use_approximate_basis:
                cache["iters"] = approximate_basis_iters
                cache["seed"] = approximate_basis_seed
            else:
                cache.pop("iters", None)
                cache.pop("seed", None)

    res = torch.kron(w1, w2).reshape(shape_original)
    return res


def kron_dims_from_ratio(shape: torch.Size, kronecker_ratio: float):
    m_divisors = divisors(shape[0])
    n_divisors = divisors(shape[1])

    log_sqrt_numel = 0.5 * math.log(shape.numel())

    candidates = []
    s_min = float("inf")
    s_max = float("-inf")

    for m1 in m_divisors:
        m2 = shape[0] // m1
        for n1 in n_divisors:
            n2 = shape[1] // n1
            a = m1 * n1
            s = math.log(a) - log_sqrt_numel
            bal = abs(math.log(m1 / m2)) + abs(math.log(n1 / n2))
            candidates.append((s, bal, m1, m2, n1, n2))
            s_min = min(s_min, s)
            s_max = max(s_max, s)

    s_range = max(abs(s_min), abs(s_max))
    if s_range == 0.0:
        _, _, m1, m2, n1, n2 = min(candidates, key=lambda x: x[1])
        return m1, m2, n1, n2

    s_target = (2.0 * kronecker_ratio - 1.0) * s_range

    best = None
    best_key = (float("inf"), float("inf"))
    for s, bal, m1, m2, n1, n2 in candidates:
        key = (abs(s - s_target), bal)
        if key < best_key:
            best_key = key
            best = (m1, m2, n1, n2)

    return best


def divisors(n: int):
    ds = []
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            ds.append(d)
            if d != n // d:
                ds.append(n // d)
    return sorted(ds)
