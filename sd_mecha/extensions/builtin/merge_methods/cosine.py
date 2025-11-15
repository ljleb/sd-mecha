import torch
from torch import Tensor
from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return


@merge_method
def add_cosine_a(
    a: Parameter(Tensor, "weight"),
    b: Parameter(Tensor, "weight"),
    alpha: Parameter(Tensor) = 1.0,
) -> Return(Tensor, "weight"):
    a_norm = torch.nn.functional.normalize(a, dim=0)
    b_norm = torch.nn.functional.normalize(b, dim=0)
    similarity = torch.nn.functional.cosine_similarity(a_norm, b_norm, dim=0)
    return add_cosine_generic(a, b, alpha, similarity)


@merge_method
def add_cosine_b(
    a: Parameter(Tensor, "weight"),
    b: Parameter(Tensor, "weight"),
    alpha: Parameter(Tensor) = 1.0,
) -> Return(Tensor, "weight"):
    similarity = torch.nn.functional.cosine_similarity(a, b, dim=0)
    dot_product = torch.sum(a * b)
    magnitude_similarity = dot_product / (torch.norm(a) * torch.norm(b))
    combined_similarity = (similarity + magnitude_similarity) / 2.0
    return add_cosine_generic(a, b, alpha, combined_similarity)


def add_cosine_generic(a: Tensor, b: Tensor, alpha: Tensor, similarity: Tensor) -> Tensor:
    k = 1 - torch.clamp(similarity - alpha, 0, 1)
    return torch.lerp(a, b, k)
