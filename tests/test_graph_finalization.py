import pytest
import torch

import sd_mecha


@pytest.mark.parametrize("child_has_weight", [True, False])
def test_planner_checks_keys_supplied_by_nested_recipes(child_has_weight):
    config = "sd1-kohya"
    weight_key = "vae.post_quant_conv.weight"
    bias_key = "vae.post_quant_conv.bias"
    weight = torch.arange(16.0).reshape(4, 4, 1, 1)
    bias = torch.arange(4.0)
    child = sd_mecha.model({weight_key: weight} if child_has_weight else {bias_key: bias}, config)
    recipe = sd_mecha.literal({weight_key: child, bias_key: bias}, config, "weight")
    expected = {bias_key: bias}
    if child_has_weight:
        expected[weight_key] = weight

    with sd_mecha.open_graph(recipe) as graph:
        plan = graph.finalize_with_keys(check_mandatory_keys=False)
        assert plan.node_to_keys[plan.root] == expected.keys()

    result = sd_mecha.merge(recipe, threads=0, output_dtype=torch.float32)
    assert result.keys() - {"__metadata__"} == expected.keys()
    for key, value in expected.items():
        torch.testing.assert_close(result[key], value)


def test_planner_keeps_disjoint_keys_for_structural_fallback():
    left = sd_mecha.model({"custom.left": torch.tensor([1.0])})
    right = sd_mecha.model({"custom.right": torch.tensor([2.0])})
    recipe = left | right

    with sd_mecha.open_graph(recipe) as graph:
        plan = graph.finalize_with_keys()
        assert plan.node_to_keys[plan.root] == {"custom.left", "custom.right"}

    result = sd_mecha.merge(recipe, threads=0, strict_mandatory_keys=True, output_dtype=torch.float32)
    torch.testing.assert_close(result["custom.left"], torch.tensor([1.0]))
    torch.testing.assert_close(result["custom.right"], torch.tensor([2.0]))


def test_structural_planner_still_requires_both_inputs_for_subtraction():
    left = sd_mecha.model({"custom.left": torch.tensor([1.0]), "custom.common": torch.tensor([4.0])})
    right = sd_mecha.model({"custom.right": torch.tensor([2.0]), "custom.common": torch.tensor([3.0])})
    recipe = sd_mecha.subtract(left, right)

    with sd_mecha.open_graph(recipe) as graph:
        plan = graph.finalize_with_keys()
        assert plan.node_to_keys[plan.root] == {"custom.common"}


def test_planner_can_reopen_a_finalized_structural_recipe():
    recipe = sd_mecha.model({"custom.weight": torch.tensor([1.0])})
    with sd_mecha.open_graph(recipe) as graph:
        finalized = graph.finalize_root()

    with sd_mecha.open_graph(finalized) as graph:
        assert list(graph.root_candidates().model_config) == [finalized.model_config]
        plan = graph.finalize_with_keys()
        assert plan.root.model_config is finalized.model_config
        assert plan.node_to_keys[plan.root] == {"custom.weight"}
