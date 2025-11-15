import math
import torch
from torch import Tensor
from sd_mecha import merge_method, Parameter, Return, StateDict


@merge_method
def weighted_sum(
    a: Parameter(StateDict[Tensor]),
    b: Parameter(StateDict[Tensor]),
    alpha: Parameter(Tensor) = 0.5,
    **kwargs,
) -> Return(Tensor):
    key = kwargs["key"]

    if alpha.numel() == 1:
        alpha_float = alpha.item()
        if math.isclose(alpha_float, 0.0):
            return a[key]
        if math.isclose(alpha_float, 1.0):
            return b[key]

    return torch.lerp(a[key], b[key], alpha)


@merge_method
def n_average(
    *models: Parameter(StateDict[Tensor]),
    **kwargs,
) -> Return(Tensor):
    key = kwargs["key"]
    return sum(model[key] for model in models) / len(models)


@merge_method
def slerp(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    alpha: Parameter(Tensor) = 0.5,
) -> Return(Tensor):
    a_normalized = a / a.norm()
    b_normalized = b / b.norm()

    ab_dot = (a_normalized * b_normalized).sum().clamp(-1, 1)

    omega = torch.arccos(ab_dot)
    a_contrib = a_normalized * torch.sin((1-alpha)*omega)
    b_contrib = b_normalized * torch.sin(alpha*omega)
    res = (a_contrib + b_contrib) / torch.sin(omega)
    res *= torch.lerp(a.norm(), b.norm(), alpha)
    if res.isnan().any():
        return torch.lerp(a, b, alpha)
    return res


@merge_method
def add_difference(
    a: Parameter(StateDict[Tensor]),
    b: Parameter(StateDict[Tensor], "delta"),
    alpha: Parameter(Tensor) = 1.0,
    **kwargs,
) -> Return(Tensor):
    key = kwargs["key"]
    if alpha.numel() == 1 and math.isclose(alpha.item(), 0.0):
        return a[key]

    b_val = b[key]  # try to load b from memory first in case it fails to merge before a
    return a[key].addcmul(b_val, alpha)


@merge_method
def subtract(
    a: Parameter(Tensor, "weight"),
    b: Parameter(Tensor, "weight"),
) -> Return(Tensor, "delta"):
    return a - b


@merge_method
def perpendicular_component(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
) -> Return(Tensor):
    norm_a = torch.linalg.norm(a)
    res = b - a * (a / norm_a * (b / norm_a)).sum()
    if res.isnan().any():
        return torch.zeros_like(a)
    return res


@merge_method
def train_difference_mask(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    c: Parameter(Tensor),
    alpha: Parameter(Tensor) = 1.0,
) -> Return(Tensor, "param"):
    return alpha * 1.8 * torch.nan_to_num((b - a).abs() / ((b - a).abs() + (b - c).abs()), nan=0)


@merge_method
def add_opposite_mask(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    c: Parameter(Tensor),
    alpha: Parameter(Tensor) = 1.0,
) -> Return(Tensor, "param"):
    return alpha * 2 * torch.nan_to_num((a - b).abs() / ((a - b).abs() + (a + b - 2*c).abs()), nan=0)


@merge_method
def add_strict_opposite_mask(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    c: Parameter(Tensor),
    alpha: Parameter(Tensor) = 1.0,
) -> Return(Tensor, "param"):
    threshold = torch.maximum(torch.abs(a - c), torch.abs(b - c))
    return alpha * torch.clamp(torch.nan_to_num((c - a) * (b - c) / threshold**2, nan=0), 0) * 2


@merge_method
def geometric_sum(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    alpha: Parameter(Tensor) = 0.5,
) -> Return(Tensor):
    a = torch.complex(a, torch.zeros_like(a))
    b = torch.complex(b, torch.zeros_like(b))
    res = a ** (1 - alpha) * b ** alpha
    return res.real


@merge_method
def multiply_quotient(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    c: Parameter(Tensor),
    alpha: Parameter(Tensor) = 1.0,
) -> Return(Tensor):
    ac_log = torch.log(a.abs()) - torch.log(c.abs())
    bc_log = torch.log(b.abs()) - torch.log(c.abs())

    b = torch.complex(b, torch.zeros_like(b))
    c = torch.complex(c, torch.zeros_like(c))

    threshold = torch.maximum(torch.abs(ac_log), torch.abs(bc_log))
    alpha = alpha * torch.clamp(-torch.nan_to_num(ac_log * bc_log / threshold**2, nan=0), 0)

    res = a * (b / c)**alpha
    res = torch.where(torch.isnan(res), a, res)
    return res.real
