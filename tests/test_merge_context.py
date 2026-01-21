from typing import Dict, Iterable
from unittest.mock import patch

import torch
from torch import Tensor

import sd_mecha
from sd_mecha import merge_method, Parameter, Return, StateDict
from sd_mecha.extensions import model_configs
from sd_mecha.merge_context import MergeMethodContext
from sd_mecha.recipe_nodes import RecipeNode


dtype = torch.float32
test_config = model_configs.ModelConfigImpl(
    "test1-mecha", {
        "component": model_configs.ModelComponent({
            "key": model_configs.KeyMetadata([2, 2], dtype),
        })
    }
)
model_configs.register(test_config)


test_config2 = model_configs.ModelConfigImpl(
    "test2-mecha", {
        "component": model_configs.ModelComponent({
            "key.0": model_configs.KeyMetadata([2], dtype),
            "key.1": model_configs.KeyMetadata([2], dtype),
        })
    }
)
model_configs.register(test_config2)


@merge_method(is_conversion=True)
class convert_test_to_test2:
    @staticmethod
    def map_keys(b):
        b["key.0", "key.1"] = b.keys["key"]

    def __call__(
        self,
        a: Parameter(StateDict[Tensor], model_config=test_config),
        **kwargs,
    ) -> Return(StateDict[Tensor], model_config=test_config2):
        value = a["key"]
        return {
            "key.0": value[0],
            "key.1": value[1],
        }


@merge_method(is_conversion=True)
class convert_test2_to_test:
    @staticmethod
    def map_keys(b):
        b["key"] = b.keys["key.0", "key.1"]

    def __call__(
        self,
        a: Parameter(StateDict[Tensor], model_config=test_config2),
        **kwargs,
    ) -> Return(Tensor, model_config=test_config):
        value0 = a["key.0"]
        value1 = a["key.1"]
        return torch.stack([value0, value1], dim=0)


@merge_method(register=False)
def weight_delta(
    a: Parameter(Tensor, merge_space="delta"),
    alpha: Parameter(float) = 1.0,
) -> Return(Tensor, merge_space="delta"):
    return a * alpha


def make_value(value, merge_space=None):
    return sd_mecha.literal({"key": torch.tensor([[value, value], [value, value]], dtype=dtype)}, config=test_config, merge_space=merge_space)


def make_value2(value, merge_space=None):
    return sd_mecha.literal({"key.0": torch.tensor([value, value], dtype=dtype), "key.1": torch.tensor([value, value], dtype=dtype)}, config=test_config2, merge_space=merge_space)


class CreateContextPatch:
    def __init__(self):
        self.original_fn = sd_mecha.merge_context.create_merge_method_context
        self.context = None

    def __call__(self, recipe: RecipeNode, root_keys: Iterable[str]) -> Dict[RecipeNode, MergeMethodContext]:
        self.context = self.original_fn(recipe, root_keys)
        return self.context


def patch_create_context():
    return patch("sd_mecha.merging.create_merge_method_context", new=CreateContextPatch())


def call_merge(recipe):
    sd_mecha.merge(
        recipe,
        strict_weight_space=False,
        threads=0,
        merge_device=None,
        merge_dtype=None,
        output_device=None,
        output_dtype=None,
    )


def act_assert_no_leak(recipe):
    with patch_create_context() as create_context_patch:
        call_merge(recipe)

    for node, mm_context in create_context_patch.context.items():
        num_leaked = sum(not output_ref.was_freed() for output_ref in mm_context.output_refs.values())
        assert num_leaked == 0


def test_simple_tree():
    a = make_value(5)
    b = make_value(2)
    recipe = sd_mecha.weighted_sum(a, b)

    act_assert_no_leak(recipe)


def test_binary_tree():
    a = make_value(5)
    b = make_value(2)
    c = make_value(3)
    d = make_value(9)
    m1 = sd_mecha.weighted_sum(a, b)
    m2 = sd_mecha.weighted_sum(c, d)
    recipe = sd_mecha.weighted_sum(m1, m2)

    act_assert_no_leak(recipe)


def test_simple_dag():
    a = make_value(5, "delta")
    b = sd_mecha.truncate_rank(a, rank_ratio=0.5)
    c = sd_mecha.truncate_rank(b, rank_ratio=0.25)
    d = sd_mecha.truncate_rank(b, rank_ratio=0.3)
    recipe = sd_mecha.weighted_sum(c, d)

    act_assert_no_leak(recipe)


def test_back_and_forth_conversion():
    a = make_value(5)
    b = sd_mecha.convert(a, test_config2)
    recipe = sd_mecha.convert(b, test_config)

    act_assert_no_leak(recipe)


def test_back_and_transform_and_forth_conversion():
    a = make_value(5, "delta")
    b = sd_mecha.convert(a, test_config2)
    c = sd_mecha.weighted_sum(b, weight_delta(b, 0.5))
    d = sd_mecha.convert(c, test_config)

    act_assert_no_leak(d)
