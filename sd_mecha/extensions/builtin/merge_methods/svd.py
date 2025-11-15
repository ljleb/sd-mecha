import logging
import math
import torch
from typing import Optional, Tuple
from torch import Tensor
from sd_mecha import merge_method, Parameter, StateDict, Return


@merge_method
def rotate(
    a: Parameter(StateDict[Tensor]),
    b: Parameter(StateDict[Tensor]),
    alignment: Parameter(float) = 1.0,
    alpha: Parameter(Tensor) = 0.0,
    centralization: Parameter(float) = 1.0,
    stiefel_eps: Parameter(float) = 1e-8,
    stiefel_max_iters: Parameter(int) = 100,
    **kwargs,
) -> Return(Tensor):
    """
    Align model A with model B by an orthogonal transform.

    Useful properties: alignment=alpha=0 returns model A, whereas alignment=alpha=1 returns model B.

    :param a: model A
    :param b: model B
    :param alignment: decides how much to align a to b by an orthogonal matrix. fractional values between 0 and 1 allow to rotate model A towards model B partially. An alignment of 1 means to minimize the distance between the two by rotation alone (no scaling allowed)
    :param alpha: interpolates the scaling component of model A with model B's. This interpolates the part of model A that is not affected by the orthogonal martix
    :param centralization: how much to center the rows of model A and model B before applying the alignment. centering the rows allows to align model A to model B a lot more closely
    :param stiefel_eps: acceptable error for wide matrices with fractional alignment
    :param stiefel_max_iters: maximum number of iterations for wide matrices with fractional alignment
    :return: model A rotated towards B by an orthogonal transform Q^alignment, after centralizing model A.
    """
    key = kwargs["key"]

    if alpha.numel() == 1:
        alpha_float = alpha.item()
        if math.isclose(alignment, 0.0) and math.isclose(alpha_float, 0.0):
            return a[key]
        if math.isclose(alignment, 1.0) and math.isclose(alpha_float, 1.0):
            return b[key]

    a = a[key]
    b = b[key]
    if len(a.shape) <= 1 or torch.allclose(a.half(), b.half()):
        return torch.lerp(a, b, alpha)

    is_conv = len(a.shape) == 4 and a.shape[-2:].numel() != 1
    if is_conv:
        shape_2d = a.shape[:2].numel(), a.shape[2:].numel()
    else:
        shape_2d = a.shape[:1].numel(), a.shape[1:].numel()

    cache = kwargs.get("cache")
    if cache is not None:
        key = kwargs["key"]
        if key not in cache:
            cache[key] = {}
        cache = cache[key]

    if cache is not None:
        # if centralization is different from the cached value, invalidate cache
        if not math.isclose(cache.get("centralization", centralization), centralization):
            cache.clear()
        else:
            cache["centralization"] = centralization

    a_neurons = a.reshape(*shape_2d)
    b_neurons = b.reshape(*shape_2d)
    a_centroid = a_neurons.mean(0) * centralization
    b_centroid = b_neurons.mean(0) * centralization
    a_neurons -= a_centroid
    b_neurons -= b_centroid

    alignment_is_float = not math.isclose(alignment, round(alignment))

    if cache is not None and "transform" in cache:
        transform = cache["transform"].to(device=a.device, dtype=a.dtype)
    else:
        transform = orthogonal_procrustes(a_neurons, b_neurons, cancel_reflection=alignment_is_float)
        if cache is not None:
            cache["transform"] = transform.to(device="cpu", dtype=torch.float16)

    if alpha.numel() > 1 or not math.isclose(alpha.item(), 0):
        a_neurons = torch.lerp(a_neurons, transform(b_neurons, -1, cache, key), alpha)

    a_neurons = transform(a_neurons, alignment, cache, key, stiefel_eps=stiefel_eps, stiefel_max_iters=stiefel_max_iters)
    a_neurons += torch.lerp(a_centroid, b_centroid, alignment)
    return a_neurons.reshape_as(a)


@merge_method
def truncate_rank(
    a: Parameter(Tensor, merge_space="delta"),
    rank_ratio: Parameter(float) = 0.5,
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

    a_2d = a.flatten(start_dim=1)
    max_rank = min(a_2d.shape)
    target_rank = min(max(round(max_rank * rank_ratio), 0), max_rank)
    if target_rank == max_rank:
        return a
    if target_rank == 0:
        return torch.zeros_like(a)

    original_shape = a.shape
    if (
        cache is not None and
        "s" in cache and cache["s"].numel() >= target_rank and
        cache.get("iters", approximate_basis_iters) == approximate_basis_iters and
        cache.get("seed", approximate_basis_seed) == approximate_basis_seed
    ):
        u = cache["u"][..., :target_rank].to(a)
        s = cache["s"][..., :target_rank].to(a)
        vh = cache["vh"][..., :target_rank, :].to(a)
    else:
        svd_driver = "gesvda" if a.is_cuda else None
        if use_approximate_basis:
            u, s, vh = svd_lowrank(a_2d, rank=target_rank, iters=approximate_basis_iters, seed=approximate_basis_seed, driver=svd_driver)
        else:
            u, s, vh = torch.linalg.svd(a_2d, full_matrices=False, driver=svd_driver)
        if cache is not None:
            cache["u"] = u.to(device="cpu", dtype=torch.float16)
            cache["s"] = s.to(device="cpu", dtype=torch.float16)
            cache["vh"] = vh.to(device="cpu", dtype=torch.float16)
            if use_approximate_basis:
                cache["iters"] = approximate_basis_iters
                cache["seed"] = approximate_basis_seed
            else:
                cache.pop("iters", None)
                cache.pop("seed", None)

    return (u[..., :target_rank] * s[..., :target_rank].unsqueeze(-2) @ vh[..., :target_rank, :]).reshape(original_shape)


def orthogonal_procrustes(a, b, cancel_reflection: bool = False):
    n, p = a.shape[-2:]
    if n < p:
        svd_driver = "gesvd" if a.is_cuda else None
        u, _, vh = svd_lowrank(a.mH @ b, rank=a.shape[0], driver=svd_driver)
        return LowRankOrthogonalMatmul(u, vh)
    else:
        svd_driver = "gesvd" if a.is_cuda else None
        u, _, vh = torch.linalg.svd(a.mH @ b, driver=svd_driver)
        if cancel_reflection:
            u[..., -1] /= torch.slogdet(u @ vh)[0]

        return FullRankOrthogonalMatmul(u @ vh)


class LowRankOrthogonalMatmul:
    def __init__(self, u, vh):
        self.u = u
        self.vh = vh

    def __call__(self, x: Tensor, t: float | int = 1.0, cache: Optional[dict] = None, key: Optional[str] = None, stiefel_eps=1e-8, stiefel_max_iters=100, **_kwargs):
        def x_proj(): return x - x @ self.vh.mH @ self.vh

        if math.isclose(t, 0.0):
            return x
        elif math.isclose(t, 1.0):
            return x_proj() + x @ self.u @ self.vh
        elif math.isclose(t, -1.0):
            return x_proj() + x @ self.vh.mH @ self.u.mH
        elif math.isclose(t, round(t)):
            if t > 0:
                return x_proj() + x @ self.u @ torch.linalg.matrix_power(self.vh @ self.u, round(t) - 1) @ self.vh
            else:
                return x_proj() + x @ self.vh.mH @ torch.linalg.matrix_power(self.u.mH @ self.vh.mH, abs(round(t)) - 1) @ self.u.mH
        else:
            u = stiefel_interpolate(self.vh.mH, self.u, t, stiefel_eps, stiefel_max_iters, cache, key)
            return x_proj() + x @ u @ self.vh

    def to(self, *args, **kwargs):
        return LowRankOrthogonalMatmul(self.u.to(*args, **kwargs), self.vh.to(*args, **kwargs))


class FullRankOrthogonalMatmul:
    def __init__(self, rotation):
        self.rotation = rotation

    def __call__(self, x: Tensor, t: float | int = 1.0, cache: Optional[dict] = None, key: Optional[str] = None, **_kwargs):
        if math.isclose(t, 0.0):
            return x

        transform = fractional_orthogonal_matrix_power(self.rotation, t, cache, key)
        return x @ transform

    def to(self, *args, **kwargs):
        return FullRankOrthogonalMatmul(self.rotation.to(*args, **kwargs))


def fractional_orthogonal_matrix_power(q, t, cache=None, key=None):
    if math.isclose(t, 0.0):
        return torch.eye(q.shape[-1], device=q.device, dtype=q.dtype)
    elif math.isclose(t, 1.0):
        return q
    elif math.isclose(t, -1.0):
        return q.mH
    elif math.isclose(t, round(t)):
        return torch.linalg.matrix_power(q, round(t))
    else:
        return orthogonal_matrix_power(q, t, cache, key)


def orthogonal_matrix_power(q, power, cache=None, key=None):
    if cache is not None and "eig_v" in cache:
        eig_v = torch.view_as_complex(cache["eig_v"].to(device=q.device, dtype=q.dtype))
        eig_vs = torch.view_as_complex(cache["eig_vs"].to(device=q.device, dtype=q.dtype))
    else:
        eig_v, eig_vs = torch.linalg.eig(q)
        if cache is not None:
            cache["eig_v"] = torch.view_as_real(eig_v).to(device="cpu", dtype=torch.float16)
            cache["eig_vs"] = torch.view_as_real(eig_vs).to(device="cpu", dtype=torch.float16)

    eig_v_pow = eig_v**power
    result = eig_vs * eig_v_pow.unsqueeze(-2) @ eig_vs.mH
    if result.imag.abs().max() > 1e-6:
        logging.warning(f"imaginary residual in fractional matrix power: max|Im Q^p| = {result.imag.abs().max().item()}, key: {key}")
    return result.to(dtype=q.dtype)


# src: https://github.com/pytorch/pytorch/blob/f714599c57b3854460002335df7d67af98f12176/torch/_lowrank.py#L150
# license applies, see /LICENSE-pytorch.txt
def svd_lowrank(a: Tensor, rank: int, iters: int = 0, seed: int = None, driver: Optional[str] = None) -> Tuple[Tensor, Tensor, Tensor]:
    m, n = a.shape[-2:]

    if m < n:
        a = a.mH

    q = get_approximate_basis(a, rank, iters=iters, seed=seed)
    b = q.mH @ a
    u, s, vh = torch.linalg.svd(b, full_matrices=False, driver=driver)
    u = q @ u

    if m < n:
        u, vh = vh.mH, u.mH

    return u, s, vh


# src: https://github.com/pytorch/pytorch/blob/f714599c57b3854460002335df7d67af98f12176/torch/_lowrank.py#L12
# license applies, see /LICENSE-pytorch.txt
def get_approximate_basis(a: Tensor, rank: int, iters: int = 0, seed: int = None) -> Tensor:
    generator = None
    if seed is not None:
        generator = torch.Generator(a.device)
        generator.manual_seed(seed)

    r = torch.randn(a.shape[-1], rank, dtype=a.dtype, device=a.device, generator=generator)
    q = torch.linalg.householder_product(*torch.geqrf(a @ r))
    for i in range(iters):
        q = torch.linalg.householder_product(*torch.geqrf(a.mH @ q))
        q = torch.linalg.householder_product(*torch.geqrf(a @ q))
    return q


def orthogonal_complete(a: Tensor) -> Tensor:
    m, n = a.shape[-2:]
    if m <= n:
        return a

    proj = torch.eye(m, device=a.device, dtype=a.dtype)[:, n:] - a @ a.mH[..., n:]
    a_extension = torch.linalg.householder_product(*torch.geqrf(proj))
    return torch.cat((a, a_extension), dim=-1)


def stiefel_interpolate(a, b, t, eps=1e-8, max_iters=100, cache=None, key=None):
    delta = log_stiefel(a, b, eps, max_iters, cache, key)
    res = exp_stiefel(a, t * delta)
    return res


def exp_stiefel(u, delta):
    n, p = u.shape[-2:]

    assert n > p, "u should be tall, not square nor wide"
    k = min(n-p, p)

    a = u.mH @ delta
    q, r = qr_pos(delta - u @ a)
    q = q[..., :k]
    r = r[..., :k, :]
    w = torch.cat((
        torch.cat((a, -r.mH), -1),
        torch.cat((r, torch.zeros_like(a[..., :k, :k])), -1)
    ), -2)
    m = torch.linalg.matrix_exp(w)
    res = u @ m[..., :p, :p] + q @ m[..., p:, :p]
    return res


def log_stiefel(a, b, eps=1e-8, max_iters=100, cache=None, key=None):
    if (
        cache is not None and "log_stiefel" in cache and
        math.isclose(math.log10(cache.get("log_stiefel_eps", eps)), math.log10(eps)) and (
            cache["log_stiefel_converged"] and cache["log_stiefel_iters"] < max_iters or

            # possible optimization: start from current cache["log_stiefel"] when max_iters > cache["log_stiefel_iters"]
            not cache["log_stiefel_converged"] and cache["log_stiefel_iters"] == max_iters
        )
    ):
        return cache["log_stiefel"].to(device=a.device, dtype=a.dtype)

    original_shape = a.shape
    a = a.view(-1, original_shape[-2], original_shape[-1])
    b = b.view(-1, original_shape[-2], original_shape[-1])
    batch_size, n, p = b.shape
    assert n > p, "a and b should be tall, not square nor wide"
    k = min(n-p, p)
    assert max_iters >= 1, "max_iters should be at least 1"

    m = a.mH @ b

    q, n_mat = qr_pos(b - a @ m)
    q = q[..., :k]
    n_mat = n_mat[..., :k, :]
    v = orthogonal_complete(torch.cat((m, n_mat), dim=-2))

    r, sigma, r_hat_t = torch.linalg.svd(v[..., p:, p:], driver="gesvd" if v.is_cuda else None)
    q @= r
    v[..., p:, :p] = r.mH @ n_mat
    v[..., :p, p:] @= r_hat_t.mH
    p_arange = torch.arange(p, p+k, device=v.device)
    v[..., p:, p:].zero_()
    v[..., p_arange, p_arange] = sigma
    del r, sigma, r_hat_t, p_arange

    v[v.slogdet()[0] < 0, ..., -1] *= -1

    k_arange = torch.arange(k, device=v.device, dtype=torch.long)
    printed_error = False
    l = None
    converged = False
    for i in range(max_iters):
        l = logm(v, key)
        c = l[..., p:, p:]
        c_norm_idx = torch.linalg.matrix_norm(c).argmax()
        c_norm = torch.linalg.matrix_norm(c[c_norm_idx])
        if c_norm > 10:
            logging.warning(f"log_stiefel: very high c_norm={c_norm.item():0.3f} at iteration {i}, batch {c_norm_idx}, key: {key}")
            printed_error = True
        elif printed_error:
            logging.warning(f"log_stiefel: started converging c_norm={c_norm.item():0.3f} at iteration {i}, batch {c_norm_idx}, key: {key}")
            printed_error = False
        if c_norm <= eps:
            converged = True
            break
        elif i == max_iters - 1:
            logging.warning(f"log_stiefel: {c_norm.item()}, batch {c_norm_idx}, key: {key}")

        s = l[..., p:, :p] @ l[..., p:, :p].mH / 12
        s[..., k_arange, k_arange] -= 0.5
        g = solve_symmetric_sylvester(s, c)
        v[..., p:] @= torch.linalg.matrix_exp(g)

    delta = a @ l[..., :p, :p] + q @ l[..., p:, :p]
    res = delta.reshape(original_shape)

    if cache is not None:
        cache["log_stiefel"] = res.to(device="cpu", dtype=torch.float16)
        cache["log_stiefel_eps"] = eps
        cache["log_stiefel_iters"] = i + 1
        cache["log_stiefel_converged"] = converged

    return res


def solve_symmetric_sylvester(s, c):
    v, vs = torch.linalg.eigh(s)
    c_t = vs.mH @ c @ vs
    d = v.unsqueeze(-2) + v.unsqueeze(-1)
    if torch.any(torch.abs(d) < 1e-12):
        logging.warning("Singular Sylvester operator: some λ_i+λ_j ≈ 0")

    g_t = c_t / d
    g = vs @ g_t @ vs.mH
    return g


def logm(m, key):
    v, vs = torch.linalg.eig(m)
    v_log = v.unsqueeze(-2).log()
    res = torch.linalg.solve(vs, vs*v_log, left=False)

    max_v, _ = res.imag.abs().flatten(start_dim=-2).max(dim=-1)
    if max_v[max_v.argmax()] > 1e-4:
        logging.warning(f"imaginary residual at batch index {max_v.argmax()}: {max_v[max_v.argmax()].item()}, key: {key}")
    return res.to(m.dtype)


def qr_pos(x):
    q, r = torch.linalg.qr(x)
    s = torch.sign(r.diagonal(offset=0, dim1=-2, dim2=-1))
    s[s == 0] = 1
    return q * s.unsqueeze(-2), r / s.unsqueeze(-1)
