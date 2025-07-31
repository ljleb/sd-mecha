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
    a_mat = q @ q.mH
    b_mat = k @ k.mH

    # A = v diag(lam_a) vᵀ
    lam_a, vec_a = torch.linalg.eigh(a_mat)
    lam_a = lam_a.clamp_min(eps)

    a_sqrt = vec_a @ torch.diag_embed(lam_a.sqrt())  @ vec_a.mH
    a_inv_sqrt = vec_a @ torch.diag_embed(lam_a.rsqrt()) @ vec_a.mH

    # C = A^{1/2} B A^{1/2}
    c_mat = a_sqrt @ b_mat @ a_sqrt

    lam_c, vec_c = torch.linalg.eigh(c_mat)
    lam_c = lam_c.clamp_min(eps)
    c_inv_quarter = vec_c @ torch.diag_embed(lam_c.pow(-0.25)) @ vec_c.mH

    # balancing transform
    m_mat = a_inv_sqrt @ c_inv_quarter

    q_bal = m_mat @ q
    k_bal = torch.linalg.solve(m_mat, k)  # avoids explicit inverse

    return q_bal, k_bal


def permute_and_align_heads(
    q_a: torch.Tensor, k_a: torch.Tensor,
    q_b: torch.Tensor, k_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Align checkpoint A to checkpoint B.

    Inputs should already be SPD-balanced. (see `balance_head_energy()`)
    Returns permuted & rotated (q_a_aligned, k_a_aligned).
    """
    device = q_a.device
    svd_driver = "gesvd" if q_a.is_cuda else None

    # fingerprints: O = U_q U_kᵀ
    u_q_a = torch.linalg.svd(q_a, full_matrices=False, driver=svd_driver).U
    u_k_a = torch.linalg.svd(k_a, full_matrices=False, driver=svd_driver).U
    o_a = u_q_a @ u_k_a.mH

    u_q_b = torch.linalg.svd(q_b, full_matrices=False, driver=svd_driver).U
    u_k_b = torch.linalg.svd(k_b, full_matrices=False, driver=svd_driver).U
    o_b = u_q_b @ u_k_b.mH

    # similarity matrix  sim_{ij} = Re tr(O_aiᵀ O_bj)
    sim = torch.einsum("ihw,jhw->ij", o_a.conj(), o_b).real
    perm = linear_sum_assignment((-sim).cpu().numpy())[1]  # maximize similarity
    perm = torch.as_tensor(perm, device=device)            # A → B order

    q_a_perm = q_a[perm]
    k_a_perm = k_a[perm]

    # batched Procrustes rotation
    min_dim = max(1, min(q_a.shape[-1], k_a.shape[-1]))
    q_s = min_dim / max(1, q_a.shape[-1])
    k_s = min_dim / max(1, k_a.shape[-1])
    s_mat = (q_b @ q_a_perm.mH)*q_s + (k_b @ k_a_perm.mH)*k_s
    u_rot, _, v_rot_h = torch.linalg.svd(s_mat, full_matrices=False, driver=svd_driver)
    r_mat = u_rot @ v_rot_h

    q_a_aligned = r_mat @ q_a_perm
    k_a_aligned = r_mat @ k_a_perm

    return q_a_aligned, k_a_aligned


def bundle_weight_bias(w, b):
    b = b.unsqueeze(-1)
    wb = torch.cat([w, b], dim=-1)
    return wb


def split_weight_bias(wb):
    b = wb[..., -1]
    w = wb[..., :-1]
    return w, b
