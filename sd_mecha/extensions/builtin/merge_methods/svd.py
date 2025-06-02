import math
import sys
import torch
from typing import Optional, Tuple
from torch import Tensor


def orthogonal_procrustes(a, b, cancel_reflection: bool = False):
    n, p = a.shape[-2:]
    if n < p:
        svd_driver = "gesvdj" if a.is_cuda else None
        u, _, vh = svd_lowrank(a.mH @ b, driver=svd_driver, rank=a.shape[0])

        return LowRankOrthogonalMatmul.create_from_svd(u, vh)
    else:
        svd_driver = "gesvd" if a.is_cuda else None
        u, _, vh = torch.linalg.svd(a.mH @ b, driver=svd_driver)
        if cancel_reflection:
            u[..., -1] /= torch.slogdet(u)[0] * torch.slogdet(vh)[0]

        return FullRankOrthogonalMatmul(u @ vh)


class LowRankOrthogonalMatmul:
    @staticmethod
    def create_from_svd(u, vh):
        n = u.shape[-2]
        eye_n = torch.eye(n, dtype=u.dtype, device=u.device)
        proj = torch.linalg.qr(torch.cat((u, vh.mH), -1)).Q
        q = u @ vh + eye_n - vh.mH @ vh
        rotation_k = proj.T @ q @ proj
        return LowRankOrthogonalMatmul(rotation_k, proj)

    def __init__(self, rotation_k, proj):
        self.rotation_k = rotation_k
        self.proj = proj

    def __call__(self, x: Tensor, t: float | int, cache: Optional[dict]):
        transform_k = fractional_orthogonal_matrix_power(self.rotation_k, t, cache)
        if transform_k is None:
            return x

        r1 = x @ self.proj
        r2 = r1 @ transform_k
        return x + (r2 - r1) @ self.proj.mH

    def to(self, *args, **kwargs):
        return LowRankOrthogonalMatmul(self.rotation_k.to(*args, **kwargs), self.proj.to(*args, **kwargs))


class FullRankOrthogonalMatmul:
    def __init__(self, rotation):
        self.rotation = rotation

    def __call__(self, x: Tensor, t: float | int, cache: Optional[dict]):
        transform = fractional_orthogonal_matrix_power(self.rotation, t, cache)
        if transform is None:
            return x
        return x @ transform

    def to(self, *args, **kwargs):
        return FullRankOrthogonalMatmul(self.rotation.to(*args, **kwargs))


def fractional_orthogonal_matrix_power(q, t, cache):
    t_is_integer = math.isclose(t, round(t))

    if math.isclose(t, 0.0):
        return None
    elif math.isclose(t, 1.0):
        return q
    elif math.isclose(t, -1.0):
        return q.mH
    elif t_is_integer:
        return torch.linalg.matrix_power(q, round(t))
    else:
        return normal_matrix_power(q, t, cache)


def normal_matrix_power(q, power, cache=None):
    if cache is not None and "eigenvalues" in cache:
        eig_v = cache["eig_v"].to(q.device, q.dtype).view_as_complex()
        eig_vs = cache["eig_vs"].to(q.device, q.dtype).view_as_complex()
    else:
        eig_v, eig_vs = torch.linalg.eig(q)
        if cache is not None:
            cache["eig_v"] = eig_v.view_as_real().to("cpu", torch.bfloat16)
            cache["eig_vs"] = eig_vs.view_as_real().to("cpu", torch.bfloat16)

    eig_v_pow = (eig_v**power).unsqueeze(-2)
    result = (eig_vs * eig_v_pow) @ eig_vs.mH
    if result.imag.abs().max() > 1e-6:
        print(f"imaginary residual in fractional matrix power: max|Im Q^p| = {result.imag.abs().max().item()}", file=sys.stderr)
    return result.to(dtype=q.dtype)


def svd_lowrank(a: Tensor, rank: int, driver: Optional[str] = None) -> Tuple[Tensor, Tensor, Tensor]:
    q = torch.linalg.householder_product(*torch.geqrf(a.mH[..., :rank])).mH
    b_t = a @ q.mH
    u, s, vh = torch.linalg.svd(b_t, driver=driver, full_matrices=False)
    vh @= q
    return u, s, vh


def orthogonal_extend(a: Tensor) -> Tensor:
    m, n = a.shape[-2:]
    if m <= n:
        return a

    proj = torch.eye(m, device=a.device, dtype=a.dtype)[:, n:] - a @ a.mH
    a_extension = torch.linalg.householder_product(*torch.geqrf(proj))
    return torch.cat((a, a_extension), dim=1)
