import torch
from scipy.optimize import linear_sum_assignment
from typing import Tuple


def balance_head_energy(
    q: torch.Tensor,
    k: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-head SPD balance.

    Parameters
    ----------
    q, k : (num_heads, head_dim, feature_dim) weight blocks
    eps : min eigenvalue for manipulating full rank SPD factors

    Returns
    -------
    q_bal, k_bal : tensors in the same shape, now satisfying
                   (q qᵀ)(k kᵀ) = I per head.
    """
    # row-Gram matrices
    a = q @ q.mH
    b = k @ k.mH

    # A^{±1/2}
    lam_a, vec_a = torch.linalg.eigh(a)
    lam_a = lam_a.clamp_min(eps)

    a_sqrt = vec_a @ torch.diag_embed(lam_a.sqrt()) @ vec_a.mH
    a_inv_sqrt = vec_a @ torch.diag_embed(lam_a.rsqrt()) @ vec_a.mH

    # C = A^{1/2} B A^{1/2};  C^{1/2}
    c = a_sqrt @ b @ a_sqrt
    lam_c, vec_c = torch.linalg.eigh(c)
    lam_c = lam_c.clamp_min(eps)
    c_sqrt = vec_c @ torch.diag_embed(lam_c.sqrt()) @ vec_c.mH

    # S = A^{-1/2} C^{1/2} A^{-1/2}      (symmetric SPD)
    s = a_inv_sqrt @ c_sqrt @ a_inv_sqrt

    # M = S^{1/2}   (still symmetric SPD ⇒ M^{-1} = M^{-H})
    lam_s, vec_s = torch.linalg.eigh(s)
    m = vec_s @ torch.diag_embed(lam_s.sqrt()) @ vec_s.mH

    q_bal = m @ q
    k_bal = torch.linalg.solve(m, k)      # m^{-1} k  == m^{-H} k

    return q_bal, k_bal


def permute_heads(
    q_a: torch.Tensor, k_a: torch.Tensor, v_a: torch.Tensor, o_a: torch.Tensor,
    q_b: torch.Tensor, k_b: torch.Tensor, v_b: torch.Tensor, o_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Permute A's attention heads to minimize the orthogonal distance between A and B.

    Inputs should already be SPD-balanced. (see `balance_head_energy()`)
    Returns permuted (q_a_aligned, k_a_aligned, v_a_aligned, o_a_aligned).
    """
    device = q_a.device

    oqk_a = _orientation(q_a, k_a)
    oqk_b = _orientation(q_b, k_b)
    ovo_a = _orientation(v_a, o_a)
    ovo_b = _orientation(v_b, o_b)

    # similarity matrix  sim_{ij} = Re tr(O_aiᵀ O_bj)
    sim_qk = torch.einsum("ihw,jhw->ij", oqk_a.conj(), oqk_b).real
    sim_vo = torch.einsum("ihw,jhw->ij", ovo_a.conj(), ovo_b).real
    sim = sim_qk + sim_vo
    perm = linear_sum_assignment((-sim).cpu().numpy())[1]  # maximize similarity
    perm = torch.as_tensor(perm, device=device)            # A → B order

    q_a_perm = q_a[perm]
    k_a_perm = k_a[perm]
    v_a_perm = v_a[perm]
    o_a_perm = o_a[perm]

    return q_a_perm, k_a_perm, v_a_perm, o_a_perm


def align_heads(
    q_a: torch.Tensor, k_a: torch.Tensor,
    q_b: torch.Tensor, k_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Align checkpoint A to checkpoint B.

    Inputs should already be SPD-balanced. (see `balance_head_energy()`)
    Returns permuted & rotated (q_a_aligned, k_a_aligned).
    """
    # batched Procrustes rotation
    min_dim = max(1, min(q_a.shape[-1], k_a.shape[-1]))
    q_s = min_dim / max(1, q_a.shape[-1])
    k_s = min_dim / max(1, k_a.shape[-1])
    s_mat = (q_b @ q_a.mH)*q_s + (k_b @ k_a.mH)*k_s
    svd_driver = "gesvd" if q_a.is_cuda else None
    u_rot, _, v_rot_h = torch.linalg.svd(s_mat, full_matrices=False, driver=svd_driver)
    r_mat = u_rot @ v_rot_h

    q_a_aligned = r_mat @ q_a
    k_a_aligned = r_mat @ k_a

    return q_a_aligned, k_a_aligned


def _orientation(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """O = U1 U2ᵀ  (thin SVD, batched)."""
    svd_driver = "gesvd" if w1.is_cuda else None
    u1 = torch.linalg.svd(w1, full_matrices=False, driver=svd_driver).U
    u2 = torch.linalg.svd(w2, full_matrices=False, driver=svd_driver).U
    return u1 @ u2.mH                        # (H, h, h)


def bundle_weight_bias(w, b):
    b = b.unsqueeze(-1)
    wb = torch.cat([w, b], dim=-1)
    return wb


def split_weight_bias(wb):
    b = wb[..., -1]
    w = wb[..., :-1]
    return w, b
