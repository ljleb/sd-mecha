import logging
import math
import torch
from typing import Optional, Tuple
from torch import Tensor
from sd_mecha.extensions.merge_methods import merge_method, Parameter, StateDict, Return
from collections import defaultdict


@merge_method(cache_factory=lambda: defaultdict(dict))
def rotate(
    a: Parameter(StateDict[Tensor]),
    b: Parameter(StateDict[Tensor]),
    alpha: Parameter(Tensor) = 0.0,
    centralization: Parameter(float) = 1.0,
    **kwargs,
) -> Return(Tensor):
    """
    Align model A with model B by an orthogonal transform.

    Useful property: alpha=1 returns model B.

    :param a: model A
    :param b: model B
    :param alpha: interpolates the scaling component of model A with model B's. This interpolates the part of model A that is not affected by the orthogonal martix
    :param centralization: how much to center the rows of model A and model B before applying the alignment. centering the rows allows to align model A to model B a lot more closely
    :return: model A rotated towards B by an orthogonal transform Q
    """
    key = kwargs["key"]

    if alpha.numel() == 1:
        alpha_float = alpha.item()
        if math.isclose(alpha_float, 1.0):
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

    if (cache := kwargs.get("cache")) is not None:
        cache = cache[kwargs["key"]]

        # if centralization is different from the cached value, invalidate cache
        if "centralization" in cache:
            if not math.isclose(cache["centralization"], centralization):
                cache.clear()
        else:
            cache["centralization"] = centralization

    a_2d = a.reshape(*shape_2d)
    b_2d = b.reshape(*shape_2d)
    a_centralization = a_2d.mean(0) * centralization
    b_centralization = b_2d.mean(0) * centralization
    a_2d = a_2d - a_centralization
    b_2d = b_2d - b_centralization

    if cache is not None and "transform" in cache:
        transform = cache["transform"].to(device=a.device, dtype=a.dtype)
    else:
        transform = orthogonal_procrustes(a_2d, b_2d)
        if cache is not None:
            cache["transform"] = transform.to(device="cpu", dtype=torch.float16)

    if alpha.numel() > 1 or not math.isclose(alpha.item(), 0):
        a_2d = torch.lerp(a_2d, transform(b_2d, inverse=True), alpha)

    a_2d = transform(a_2d)
    a_2d = a_2d + b_centralization
    return a_2d.reshape_as(a)


@merge_method(cache_factory=dict)
def truncate_rank(
    a: Parameter(Tensor, merge_space="delta"),
    rank: Parameter(int) = 8,
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
    target_rank = min(max(round(rank), 0), max_rank)
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


@merge_method(reuse_outputs=False)
def get_rank_from_ratio(
    a: Parameter(Tensor, {"weight", "delta"}),
    ratio: Parameter(float, "param") = 0.5,
) -> Return(int, "param"):
    shape_2d = torch.Size((a.shape[:1].numel(), a.shape[1:].numel()))
    max_rank = min(shape_2d)
    target_rank = min(max(round(max_rank * ratio), 0), max_rank)
    return target_rank


def orthogonal_procrustes(a, b, cancel_reflection: bool = False):
    n, p = a.shape[-2:]
    svd_driver = "gesvd" if a.is_cuda else None
    if n < p:
        b_q, b_r = torch.linalg.qr(b.mH)
        u, _, vh = torch.linalg.svd(a.mH @ b_r.mH, driver=svd_driver, full_matrices=False)
        return LowRankOrthogonalMatmul(u, vh @ b_q.mH)
    else:
        u, _, vh = torch.linalg.svd(a.mH @ b, driver=svd_driver, full_matrices=False)
        if cancel_reflection:
            u[..., -1] /= torch.slogdet(u @ vh)[0]

        return FullRankOrthogonalMatmul(u @ vh)


class LowRankOrthogonalMatmul:
    def __init__(self, u, vh):
        self.u = u
        self.v = vh.mH

    def __call__(
        self,
        x: Tensor,
        inverse: bool = False,
    ):
        if inverse:
            return x @ self.v @ self.u.mH
        return x @ self.u @ self.v.mH

    def to(self, *args, **kwargs):
        return LowRankOrthogonalMatmul(self.u.to(*args, **kwargs), self.v.mH.to(*args, **kwargs))


class FullRankOrthogonalMatmul:
    def __init__(self, rotation):
        self.rotation = rotation

    def __call__(
        self,
        x: Tensor,
        inverse: bool = False,
    ):
        if inverse:
            return x @ self.rotation.mH
        return x @ self.rotation

    def to(self, *args, **kwargs):
        return FullRankOrthogonalMatmul(self.rotation.to(*args, **kwargs))


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
