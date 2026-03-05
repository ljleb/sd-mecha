import torch
from scipy.optimize import linear_sum_assignment
from typing import Tuple


def balance_head_energy(q: torch.Tensor, k: torch.Tensor, eps: float = 1e-6):
    """
    Wq, Wk: [H, 64, d] (NO bias column here)
    Returns Wq', Wk' with identical logits: (Wq'x)^T (Wk'x) == (Wq x)^T (Wk x)
    """
    a = q @ q.mH
    b = k @ k.mH

    a.diagonal()[:] += eps
    b.diagonal()[:] += eps

    v_a, vs_a = torch.linalg.eigh(a)
    v_a = v_a.clamp_min(eps)
    v_a_half = v_a.pow(0.5)
    v_a_half_neg = v_a.pow(-0.5)
    a_half = (vs_a * v_a_half.unsqueeze(-2)) @ vs_a.mH
    a_half_neg = (vs_a * v_a_half_neg.unsqueeze(-2)) @ vs_a.mH

    c = a_half @ b @ a_half
    v_c, vs_c = torch.linalg.eigh(c)
    v_c_half = v_c.clamp_min(eps).pow(0.5)
    c_half = (vs_c * v_c_half.unsqueeze(-2)) @ vs_c.mT

    s = a_half_neg @ c_half @ a_half_neg
    q = s @ q
    k = torch.linalg.solve(s, k)
    return q, k


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

    sim_q = subspace_alignment(q_a, q_b)
    sim_k = subspace_alignment(k_a, k_b)
    sim_v = subspace_alignment(v_a, v_b)
    sim_o = subspace_alignment(o_a, o_b)

    sim = sim_q + sim_k + sim_v + sim_o
    perm = linear_sum_assignment((-sim).cpu().numpy())[1]
    perm = torch.as_tensor(perm, device=device)

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


def subspace_alignment(a: torch.Tensor, b: torch.Tensor, eps=1e-12) -> torch.Tensor:
    driver = "gesvd" if a.is_cuda else None
    a_n = a / a.norm(dim=-1, keepdim=True).clamp(min=eps)
    b_n = b / b.norm(dim=-1, keepdim=True).clamp(min=eps)
    h = a_n[:, None] @ b_n[None, :].mH
    s = torch.linalg.svdvals(h, driver=driver)
    return s.log().mean(dim=-1)


def bundle_weight_bias(w, b):
    b = b.unsqueeze(-1)
    wb = torch.cat([w, b], dim=-1)
    return wb


def split_weight_bias(wb):
    b = wb[..., -1]
    w = wb[..., :-1]
    return w, b
