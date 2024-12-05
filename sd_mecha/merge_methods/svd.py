import functools
from typing import Optional, Tuple
import torch
from torch import Tensor


def orthogonal_procrustes(a, b, cancel_reflection: bool = False):
    if a.shape[0] < a.shape[1]:
        svd_driver = "gesvdj" if a.is_cuda else None
        u, _, v = torch_svd_lowrank(a.mH @ b, driver=svd_driver, q=a.shape[0])
        vh = v.mH
        del v
    else:
        svd_driver = "gesvd" if a.is_cuda else None
        u, _, vh = torch.linalg.svd(a.mH @ b, driver=svd_driver)
        if cancel_reflection:
            u[:, -1] /= torch.slogdet(u)[0] * torch.slogdet(vh)[0]

    return u, vh


def fractional_orthogonal_matrix_power(u, v, alignment, cache):
    if cache is not None and "eigenvalues" in cache:
        eig_v = cache["eig_v"].to(u.device, u.dtype).view_as_complex()
        eig_vs = cache["eig_vs"].to(u.device, u.dtype).view_as_complex()
    else:
        eig_v, eig_vs = torch.linalg.eig(u @ v.mH)
        if cache is not None:
            cache["eig_v"] = eig_v.view_as_real().to("cpu", torch.bfloat16)
            cache["eig_vs"] = eig_vs.view_as_real().to("cpu", torch.bfloat16)

    eig_v_pow = eig_v**alignment
    result = torch.linalg.solve_ex(eig_vs, eig_vs @ torch.diag_embed(eig_v_pow), left=False)[0]
    return result.to(dtype=u.dtype)


def close_ortho_columns_full(a, b):
    original_dtype = a.dtype
    m, n = a.shape[-2:]
    assert a.shape == b.shape

    if n == m:
        return a, b, MatmulIdentity()

    if 2*n < m:
        def complement_proj_fn(a, b, x):
            x = b.mH @ x
            x = b @ x
            c = a.mH @ x
            c = a @ c
            return x - c

        def complement_proj_fn_t(a, b, y):
            y1 = a.mH @ y
            y1 = a @ y1
            y = y - y1
            y = b.mH @ y
            y = b @ y
            return y

        a_n = extend_ortho(
            a,
            functools.partial(complement_proj_fn, a, b),
            functools.partial(complement_proj_fn_t, a, b),
            m, 2*n,
        )
        b_n = extend_ortho(
            b,
            functools.partial(complement_proj_fn, b, a),
            functools.partial(complement_proj_fn_t, b, a),
            m, 2*n,
        )
        assert a_n.shape == b_n.shape

        # appending columns aligned in a criss-cross fashion might not be the best approach?
        # the idea is to close the column space of A and B so that both lie in the same vector space.
        # maybe this type of alignment prevents natural rotations?
        to_align = torch.stack([
            a_n[..., n:].mH @ b_n[..., :n],
            b_n[..., n:].mH @ a_n[..., :n],
        ], dim=0)
        svd_driver = "gesvd" if a.is_cuda else None
        u, _, vt = torch.linalg.svd(to_align, driver=svd_driver, full_matrices=False)
        r = u @ vt
        a_n[..., n:] @= r[0]
        b_n[..., n:] @= r[1]

        proj = get_shared_basis(
            a_n, b_n,
            2*n,
            device=a.device, dtype=a.dtype,
        )
        a_p = proj.mH @ a_n
        b_p = proj.mH @ b_n
        if not original_dtype.is_complex and (b_p @ a_p.mH).slogdet()[0] < 0:
            proj[-1] *= -1
            a_p = proj.mH @ a_n
            b_p = proj.mH @ b_n
    else:
        def complement_proj_fn(a, x):
            c = a.mH @ x
            c = a @ c
            return x - c

        def complement_proj_fn_t(a, y):
            c = a.mH @ y
            c = a @ c
            return y - c

        a_n = extend_ortho(
            a,
            functools.partial(complement_proj_fn, a),
            functools.partial(complement_proj_fn_t, a),
            m, m,
        )
        b_n = extend_ortho(
            b,
            functools.partial(complement_proj_fn, b),
            functools.partial(complement_proj_fn_t, b),
            m, m,
        )
        assert a_n.shape == b_n.shape

        # for 2n >= m, a projection is not useful.
        # we simply add new orthogonal columns until both matrices are square
        #  and align those in A with those in B for minimal interference between
        #  the meaningful columns during interpolation
        to_align = a_n[..., n:].mH @ b_n[..., n:]
        svd_driver = "gesvd" if a.is_cuda else None
        u, _, vt = torch.linalg.svd(to_align, driver=svd_driver, full_matrices=False)
        a_n[..., n:] @= u @ vt

        proj = MatmulIdentity()
        a_p = a_n
        b_p = b_n

        if not original_dtype.is_complex and (b_p @ a_p.mH).slogdet()[0] < 0:
            a_p[..., -1] *= -1

    return a_p, b_p, proj


class MatmulIdentity:
    def __matmul__(self, other):
        return other

    def __rmatmul__(self, other):
        return other

    @property
    def mH(self):
        return self

    @property
    def mT(self):
        return self

    @property
    def H(self):
        return self

    @property
    def T(self):
        return self

    def to(self, *args, **kwargs):
        return self


def extend_ortho(x, r, rh, input_m, target_n):
    if target_n <= x.shape[-1]:
        return x

    k = target_n - x.shape[-1]
    k_frame = memory_efficient_get_approximate_basis(r, rh, input_m, k, dtype=x.dtype, device=x.device)
    return torch.cat((x, k_frame), dim=-1)


# need to redefine torch.svd_lowrank to specify the svd driver
def torch_svd_lowrank(
    A: Tensor,
    q: Optional[int] = 6,
    niter: Optional[int] = 2,
    M: Optional[Tensor] = None,
    driver: Optional[str] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    q = 6 if q is None else q
    m, n = A.shape[-2:]
    if M is None:
        M_t = None
    else:
        M_t = transpose(M)
    A_t = transpose(A)

    # Algorithm 5.1 in Halko et al 2009, slightly modified to reduce
    # the number conjugate and transpose operations
    if m < n or n > q:
        # computing the SVD approximation of a transpose in
        # order to keep B shape minimal (the m < n case) or the V
        # shape small (the n > q case)
        Q = get_approximate_basis(A_t, q, niter=niter, M=M_t)
        Q_c = conjugate(Q)
        if M is None:
            B_t = matmul(A, Q_c)
        else:
            B_t = matmul(A, Q_c) - matmul(M, Q_c)
        assert B_t.shape[-2] == m, (B_t.shape, m)
        assert B_t.shape[-1] == q, (B_t.shape, q)
        assert B_t.shape[-1] <= B_t.shape[-2], B_t.shape
        U, S, Vh = torch.linalg.svd(B_t, driver=driver, full_matrices=False)
        V = Vh.mH
        V = Q.matmul(V)
    else:
        Q = get_approximate_basis(A, q, niter=niter, M=M)
        Q_c = conjugate(Q)
        if M is None:
            B = matmul(A_t, Q_c)
        else:
            B = matmul(A_t, Q_c) - matmul(M_t, Q_c)
        B_t = transpose(B)
        assert B_t.shape[-2] == q, (B_t.shape, q)
        assert B_t.shape[-1] == n, (B_t.shape, n)
        assert B_t.shape[-1] <= B_t.shape[-2], B_t.shape
        U, S, Vh = torch.linalg.svd(B_t, driver=driver, full_matrices=False)
        V = Vh.mH
        U = Q.matmul(U)

    return U, S, V


def get_shared_basis(a, b, max_rank, niter=2, device=None, dtype=None):
    assert a.shape == b.shape
    m, n = a.shape[-2:]

    def outer_fn(x):
        x = a.mH @ x
        x = b @ x
        return x

    def outer_fn_t(y):
        y = b.mH @ y
        y = a @ y
        return y

    basis = last_basis = None
    rank = max_rank

    while basis is None or (
        is_orthogonal((basis.mH @ b) @ (a.mH @ basis)) and
        is_orthogonal((basis.mH @ a) @ (a.mH @ basis)) and
        is_orthogonal((basis.mH @ b) @ (b.mH @ basis))
    ):
        last_basis = basis
        basis = memory_efficient_get_approximate_basis(
            outer_fn, outer_fn_t,
            m, rank, niter=niter,
            device=device, dtype=dtype,
        )
        rank -= 1

    if rank+2 < max_rank:
        print(f"optimized rank: {rank+2},\tmax case: {max_rank},\tbasis shape: {(basis if last_basis is None else last_basis).shape}")

    return basis if last_basis is None else last_basis


def is_orthogonal(a, eye=None):
    if eye is None:
        eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
    return torch.allclose(a.mH @ a, eye)


def memory_efficient_get_approximate_basis(f, fh, input_m: int, rank: int, niter=0, device=None, dtype=None):
    Q = torch.eye(input_m, rank, dtype=dtype, device=device)
    Q = torch.linalg.qr(f(Q)).Q
    for i in range(niter):
        Q = torch.linalg.qr(fh(Q)).Q
        Q = torch.linalg.qr(f(Q)).Q

    return Q


def get_approximate_basis(A: Tensor, q: int, niter: Optional[int] = 2, M: Optional[Tensor] = None) -> Tensor:
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

        M (Tensor, optional): the input tensor's mean of size
                              :math:`(*, 1, n)`.

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).
    """

    niter = 2 if niter is None else niter
    m, n = A.shape[-2:]
    dtype = A.dtype

    R = torch.eye(n, q, dtype=dtype, device=A.device)

    # The following code could be made faster using torch.geqrf + torch.ormqr
    # but geqrf is not differentiable
    A_H = transjugate(A)
    if M is None:
        Q = torch.linalg.qr(matmul(A, R)).Q
        for i in range(niter):
            Q = torch.linalg.qr(matmul(A_H, Q)).Q
            Q = torch.linalg.qr(matmul(A, Q)).Q
    else:
        M_H = transjugate(M)
        Q = torch.linalg.qr(matmul(A, R) - matmul(M, R)).Q
        for i in range(niter):
            Q = torch.linalg.qr(matmul(A_H, Q) - matmul(M_H, Q)).Q
            Q = torch.linalg.qr(matmul(A, Q) - matmul(M, Q)).Q

    return Q


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
