import torch
from torch import Tensor
from typing import Optional, Tuple, Dict


def orthogonal_procrustes(a, b, cancel_reflection: bool = False):
    if a.shape != b.shape:
        raise ValueError(f"a {tuple(a.shape)} and b {tuple(b.shape)} must have the same shape")

    if a.shape[0] + 10 < a.shape[1]:
        svd_driver = "gesvdj" if a.is_cuda else None
        u, _, v_t = torch_svd_lowrank(a.T @ b, q=a.shape[0] + 10, driver=svd_driver, full_matrices=cancel_reflection)
    else:
        svd_driver = "gesvd" if a.is_cuda else None
        u, _, v_t = torch.linalg.svd(a.T @ b, driver=svd_driver)

    transform = u @ v_t
    transform_det = torch.linalg.det(transform)
    if torch.isclose(transform_det, torch.tensor(0, device=transform_det.device, dtype=transform_det.dtype)):
        raise RuntimeError(
            f"determinant error: {transform_det}. "
            'This can happen when merging on the CPU with the "rotate" method. '
            "Consider merging on a cuda device, "
            "or try setting `alignment` to 1 for the problematic blocks. "
            "See this related discussion for more info: "
            "https://github.com/s1dlx/meh/pull/50#discussion_r1429469484"
        )

    if cancel_reflection:
        transform_det = torch.linalg.det(transform)
        if transform_det < 0:
            transform -= 2 * u[:, -1:] * v_t[-1:]

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
    M_t = None
    A_t = transpose(A)

    # Algorithm 5.1 in Halko et al 2009, slightly modified to reduce
    # the number conjugate and transpose operations
    assert m < n or n > q
    # computing the SVD approximation of a transpose in
    # order to keep B shape minimal (the m < n case) or the V
    # shape small (the n > q case)
    Q = get_approximate_basis(A_t, q, niter=niter)
    Q_c = conjugate(Q)
    B_t = matmul(A, Q_c)
    assert B_t.shape[-2] == m, (B_t.shape, m)
    assert B_t.shape[-1] == q, (B_t.shape, q)
    assert B_t.shape[-1] <= B_t.shape[-2], B_t.shape
    U, S, Vh = torch.linalg.svd(B_t, driver=driver, full_matrices=full_matrices)
    V = Q.matmul(Vh.mH)
    if full_matrices:
        V = orthogonal_extend(V)

    return U, S, V.mH


def get_approximate_basis(A: Tensor, q: int, niter: Optional[int] = 2) -> Tensor:
    """Return tensor :math:`Q` with :math:`q` orthonormal columns such
    that :math:`Q Q^H A` approximates :math:`A`. If :math:`M` is
    specified, then :math:`Q` is such that :math:`Q Q^H (A - M)`
    approximates :math:`A - M`.

    .. note:: The implementation is based on the Algorithm 4.4 from
              Halko et al, 2009.

    .. note:: For an adequate approximation of a k-rank matrix
              :math:`A`, where k is not known in advance but could be
              estimated, the number of :math:`Q` columns, q, can be
              choosen according to the following criteria: in general,
              :math:`k <= q <= min(2*k, m, n)`. For large low-rank
              matrices, take :math:`q = k + 5..10`.  If k is
              relatively small compared to :math:`min(m, n)`, choosing
              :math:`q = k + 0..2` may be sufficient.

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    Args::
        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int): the dimension of subspace spanned by :math:`Q`
                 columns.

        niter (int, optional): the number of subspace iterations to
                               conduct; ``niter`` must be a
                               nonnegative integer. In most cases, the
                               default value 2 is more than enough.

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).
    """

    niter = 2 if niter is None else niter
    m, n = A.shape[-2:]
    dtype = get_floating_dtype(A)

    R = torch.randn(n, q, dtype=dtype, device=A.device)

    A_H = transjugate(A)
    Q = torch.linalg.qr(matmul(A, R)).Q
    for i in range(niter):
        Q = torch.linalg.qr(matmul(A_H, Q)).Q
        Q = torch.linalg.qr(matmul(A, Q)).Q

    return Q


def orthogonal_extend(A: torch.Tensor) -> torch.Tensor:
    m, n = A.shape
    if m <= n:
        return A

    proj = torch.eye(m, device=A.device, dtype=A.dtype) - A @ A.mH
    proj @= torch.randn(m, m - n, device=A.device, dtype=A.dtype)
    A_extension = torch.linalg.householder_product(*torch.geqrf(proj))
    return torch.cat((A, A_extension), dim=1)


def transjugate(A):
    """Return transpose conjugate of a matrix or batches of matrices."""
    return conjugate(transpose(A))


def conjugate(A):
    """Return conjugate of tensor A.

    .. note:: If A's dtype is not complex, A is returned.
    """
    if A.is_complex():
        return A.conj()
    return A


def transpose(A):
    """Return transpose of a matrix or batches of matrices."""
    ndim = len(A.shape)
    return A.transpose(ndim - 1, ndim - 2)


def matmul(A: Optional[Tensor], B: Tensor) -> Tensor:
    """Multiply two matrices.

    If A is None, return B. A can be sparse or dense. B is always
    dense.
    """
    if A is None:
        return B
    if is_sparse(A):
        return torch.sparse.mm(A, B)
    return torch.matmul(A, B)


def is_sparse(A):
    """Check if tensor A is a sparse tensor"""
    if isinstance(A, torch.Tensor):
        return A.layout == torch.sparse_coo

    error_str = "expected Tensor"
    if not torch.jit.is_scripting():
        error_str += f" but got {type(A)}"
    raise TypeError(error_str)


def get_floating_dtype(A):
    """Return the floating point dtype of tensor A.

    Integer types map to float32.
    """
    dtype = A.dtype
    if dtype in (torch.float16, torch.float32, torch.float64):
        return dtype
    return torch.float32
