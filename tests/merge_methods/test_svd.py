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


def test_truncate_rank_rebuilds_factors_with_stale_cache_storage():
    weight = torch.diag(torch.tensor([3., 2., 1.]))
    delta = sd_mecha.model({"weight": weight}, merge_space="delta")
    recipe = sd_mecha.truncate_rank(delta, rank=1, use_approximate_basis=False)
    assert recipe.merge_method.create_cache() == {"device": "cpu", "dtype": torch.float16}
    cache = recipe.merge_method.create_cache(device="cpu:0", dtype=torch.float32)

    def merge():
        result = sd_mecha.merge(recipe, cache={recipe: cache}, strict_merge_space="delta", output_dtype=torch.float32, threads=0)
        torch.testing.assert_close(result["weight"], torch.diag(torch.tensor([3., 0., 0.])))

    merge()
    factors = cache["weight"]
    cached_u = factors["u"]
    assert cached_u.dtype == torch.float32
    merge()
    assert factors["u"] is cached_u

    cache["dtype"] = torch.float64
    merge()
    assert all(factors[name].dtype == torch.float64 for name in ("u", "s", "vh"))

    factors["u"] = factors["u"].to("meta")
    merge()
    assert factors["u"].device.type == "cpu"
    assert cache["device"] == "cpu:0"
    assert cache["dtype"] == torch.float64


def test_rotate_reuses_transforms_only_with_matching_centralization_and_storage():
    a = {"weight": torch.eye(3)}
    b = {"weight": torch.roll(torch.eye(3), 1, dims=1)}
    cache = svd.rotate.create_cache(dtype=torch.float32)

    def rotate(centralization):
        return svd.rotate.__wrapped__(a, b, torch.tensor(0.), centralization, cache=cache, key="weight")

    rotate(0.)
    first = cache["weight"]["transform"]
    rotate(0.)
    assert cache["weight"]["transform"] is first

    cache["dtype"] = torch.float64
    rotate(0.)
    assert cache["weight"]["transform"].rotation.dtype == torch.float64

    rotate(0.5)
    second = cache["weight"]["transform"]
    rotate(1.)
    assert cache["weight"]["transform"] is not second
    assert cache["weight"]["centralization"] == 1.
