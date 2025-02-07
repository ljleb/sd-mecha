import torch
from torch import Tensor
from typing import Optional, Tuple, Dict


def orthogonal_procrustes(a, b, cancel_reflection: bool = False):
    if a.shape != b.shape:
        raise ValueError(f"a {tuple(a.shape)} and b {tuple(b.shape)} must have the same shape")

    if not cancel_reflection and a.shape[0] + 10 < a.shape[1]:
        svd_driver = "gesvdj" if a.is_cuda else None
        u, _, v_t = torch_svd_lowrank(a.T @ b, q=a.shape[0] + 10, driver=svd_driver, full_matrices=cancel_reflection)
    else:
        svd_driver = "gesvd" if a.is_cuda else None
        u, _, v_t = torch.linalg.svd(a.T @ b, driver=svd_driver)
        if cancel_reflection:
            u[:, -1] /= torch.slogdet(u)[0] * torch.slogdet(v_t)[0]

    transform = u @ v_t
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


# need to redefine torch.svd_lowrank to specify the svd driver and for full_matrices support
def torch_svd_lowrank(
    A: Tensor,
    q: Optional[int] = 6,
    niter: Optional[int] = 2,
    driver: Optional[str] = None,
    full_matrices: Optional[bool] = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    q = 6 if q is None else q
    m, n = A.shape[-2:]
    A_t = A.T

    # Algorithm 5.1 in Halko et al 2009, slightly modified to reduce
    # the number conjugate and transpose operations
    assert m < n or n > q
    # computing the SVD approximation of a transpose in
    # order to keep B shape minimal (the m < n case) or the V
    # shape small (the n > q case)
    Q = get_approximate_basis(A_t, q, niter=niter)
    Q_c = Q.conj()
    B_t = A @ Q_c
    assert B_t.shape[-2] == m, (B_t.shape, m)
    assert B_t.shape[-1] == q, (B_t.shape, q)
    assert B_t.shape[-1] <= B_t.shape[-2], B_t.shape
    U, S, Vh = torch.linalg.svd(B_t, driver=driver, full_matrices=full_matrices)
    V = Q @ Vh.mH
    if full_matrices:
        V = orthogonal_extend(V)

    return U, S, V.mH


def get_approximate_basis(A: Tensor, q: int, niter: Optional[int] = 2) -> Tensor:
    niter = 2 if niter is None else niter
    m, n = A.shape[-2:]
    dtype = get_floating_dtype(A)

    R = torch.randn(n, q, dtype=dtype, device=A.device)

    A_H = A.mH
    Q = torch.linalg.qr(A @ R).Q
    for i in range(niter):
        Q = torch.linalg.qr(A_H @ Q).Q
        Q = torch.linalg.qr(A @ Q).Q

    return Q


def orthogonal_extend(A: torch.Tensor) -> torch.Tensor:
    m, n = A.shape
    if m <= n:
        return A

    proj = torch.eye(m, device=A.device, dtype=A.dtype) - A @ A.mH
    proj @= torch.randn(m, m - n, device=A.device, dtype=A.dtype)
    A_extension = torch.linalg.householder_product(*torch.geqrf(proj))
    return torch.cat((A, A_extension), dim=1)


def get_floating_dtype(A):
    dtype = A.dtype
    if dtype.is_floating_point:
        return dtype
    return torch.float32
