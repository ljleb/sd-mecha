from unittest.mock import patch

import pytest
import torch

import sd_mecha
from sd_mecha.extensions.builtin.merge_methods import svd


@pytest.mark.parametrize("include_base", [False, True])
def test_rotate_wrapper_builds_and_merges_a_recipe(include_base):
    a = sd_mecha.model({"weight": torch.eye(2)})
    b_weight = torch.tensor([[2., 3.], [4., 5.]])
    b = sd_mecha.model({"weight": b_weight})
    base = sd_mecha.model({"weight": torch.zeros(2, 2)}) if include_base else None

    recipe = sd_mecha.rotate(a, b, base, alpha=1.0)
    result = sd_mecha.merge(recipe, output_dtype=torch.float32, threads=0)

    torch.testing.assert_close(result["weight"], b_weight)


def test_truncate_rank_reuses_exact_factors_for_an_approximate_request():
    weight = torch.diag(torch.tensor([3., 2., 1.]))
    cache = {}
    truncate = svd.truncate_rank.__wrapped__
    truncate(weight, rank=2, use_approximate_basis=False, cache=cache, key="weight")

    with patch.object(svd, "svd_lowrank", side_effect=AssertionError("Expected cache reuse")):
        result = truncate(weight, rank=1, approximate_basis_iters=4, approximate_basis_seed=2, cache=cache, key="weight")

    torch.testing.assert_close(result, torch.diag(torch.tensor([3., 0., 0.])))


def test_truncate_rank_reuses_approximate_factors_with_the_same_iterations_and_seed():
    weight = torch.diag(torch.tensor([3., 2., 1.]))
    cache = {}
    truncate = svd.truncate_rank.__wrapped__
    truncate(weight, rank=2, approximate_basis_seed=2, cache=cache, key="weight")
    cached_factors = cache["weight"]["u"]

    truncate(weight, rank=1, use_approximate_basis=False, approximate_basis_seed=2, cache=cache, key="weight")
    assert cache["weight"]["u"] is cached_factors

    truncate(weight, rank=1, use_approximate_basis=False, approximate_basis_seed=3, cache=cache, key="weight")
    assert cache["weight"]["u"] is not cached_factors
