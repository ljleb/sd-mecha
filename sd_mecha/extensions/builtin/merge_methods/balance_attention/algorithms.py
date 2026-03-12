import torch


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


def bundle_weight_bias(w, b):
    b = b.unsqueeze(-1)
    wb = torch.cat([w, b], dim=-1)
    return wb


def split_weight_bias(wb):
    b = wb[..., -1]
    w = wb[..., :-1]
    return w, b
