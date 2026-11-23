import warnings
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

import sd_mecha
from sd_mecha.extensions import merge_methods
from sd_mecha.extensions.builtin.merge_methods.kronecker import lokr_dims, lokr_dims_lycoris, truncate_kronecker
from sd_mecha.graph_finalization import PropagatableKeyVisitor
from sd_mecha.keys_map import RealizedKeyRelation


CONFIG = "sd1-kohya"
WEIGHT = "vae.post_quant_conv.weight"
BIAS = "vae.post_quant_conv.bias"


def merge(recipe, space="weight"):
    output = {}
    sd_mecha.merge(recipe, output=output, output_dtype=torch.float32, strict_merge_space=space, threads=0)
    return output


def test_lora_extraction_caps_rank_and_keeps_convolution_shape():
    method = merge_methods.resolve("extract_lora_sd1-kohya_lycoris_lora").instantiate()
    weight = torch.arange(12.).reshape(6, 2, 1, 1)
    relation = RealizedKeyRelation(("up", "down", "alpha"), {})
    adapter = method(weight, 8, False, 2, None, key_relation=relation)

    assert adapter["up"].shape == (6, 2, 1, 1)
    assert adapter["alpha"].item() == 2
    torch.testing.assert_close(adapter["up"].flatten(1) @ adapter["down"].flatten(1), weight.flatten(1))


def test_sparse_lora_merge_plans_only_the_available_weight():
    weight = torch.eye(4).reshape(4, 4, 1, 1)
    delta = sd_mecha.model({WEIGHT: weight}, CONFIG, "delta")
    method = merge_methods.resolve("extract_lora_sd1-kohya_lycoris_lora")
    recipe = method(delta, 4, False, 2, None)
    extraction = method.__wrapped__
    probe_sizes = []
    output_groups = PropagatableKeyVisitor._root_output_groups

    def count_groups(visitor, node):
        groups = output_groups(visitor, node)
        probe_sizes.append(len(groups))
        return groups

    with (
        patch.object(extraction, "get_output_keys", wraps=extraction.get_output_keys) as get_output_keys,
        patch.object(PropagatableKeyVisitor, "_root_output_groups", count_groups),
    ):
        result = merge(recipe)

    get_output_keys.assert_called_once_with(WEIGHT)
    assert probe_sizes == [3]
    prefix = "lycoris_vae_post_quant_conv"
    up = result[f"{prefix}.lora_up.weight"]
    down = result[f"{prefix}.lora_down.weight"]
    assert result[f"{prefix}.alpha"].item() == 4
    torch.testing.assert_close(up.flatten(1) @ down.flatten(1), weight.flatten(1))


def test_norm_extraction_and_deprecated_conversion():
    bias = torch.arange(4.)
    delta = sd_mecha.model({BIAS: bias}, CONFIG, "delta")
    adapter = sd_mecha.lycoris.resolve("norm")(delta)
    result = merge(adapter)
    torch.testing.assert_close(result["lycoris_vae_post_quant_conv.b_norm"], bias)

    with pytest.warns(UserWarning, match="converting lycoris to base is deprecated"):
        converted = merge(sd_mecha.convert(adapter, CONFIG), "delta")
    torch.testing.assert_close(converted[BIAS], bias)


def test_lokr_extraction_default_round_trips_through_application():
    w1 = torch.diag(torch.tensor([1., 2.]))
    w2 = torch.tensor([[3., 1.], [-2., 4.]])
    weight = torch.kron(w1, w2).reshape(4, 4, 1, 1)
    delta = sd_mecha.model({WEIGHT: weight}, CONFIG, "delta")
    base = sd_mecha.model({WEIGHT: torch.zeros_like(weight)}, CONFIG)
    adapter = sd_mecha.lycoris.resolve("lokr")(delta)

    extracted = merge(adapter)
    assert extracted["lycoris_vae_post_quant_conv.lokr_w1"].ndim == 2
    assert extracted["lycoris_vae_post_quant_conv.alpha"].item() == 1.0
    torch.testing.assert_close(merge(sd_mecha.apply_lycoris(base, adapter))[WEIGHT], weight)


@pytest.mark.parametrize("select_dims", [None, lokr_dims, lokr_dims_lycoris])
def test_truncate_kronecker_dimensions_and_bias_passthrough(select_dims):
    bias = torch.arange(4.)
    weight = torch.kron(torch.eye(2), torch.ones((2, 2))).reshape(4, 4, 1, 1)
    delta = sd_mecha.model({WEIGHT: weight, BIAS: bias}, CONFIG, "delta")
    dims = select_dims(delta) if select_dims is not None else None
    result = merge(truncate_kronecker(delta, dims, use_approximate_basis=False), "delta")

    torch.testing.assert_close(result[WEIGHT], weight)
    torch.testing.assert_close(result[BIAS], bias)


def test_lokr_recipe_format_version(tmp_path):
    checkpoint = tmp_path / "delta.safetensors"
    save_file({WEIGHT: torch.ones((4, 4, 1, 1))}, checkpoint)
    recipe = sd_mecha.lycoris.resolve("lokr")(sd_mecha.model(checkpoint, CONFIG, "delta"))
    serialized = sd_mecha.serialize(recipe)

    assert serialized.splitlines()[0] == "version 0.1.0"
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        restored = sd_mecha.deserialize(serialized)
    assert restored.merge_method.get_identifier() == "extract_lokr_sd1-kohya_lycoris_lora"


@pytest.mark.parametrize(
    "old_name, arguments, algorithm, adapter_format, ratio",
    [
        ("extract_lycoris_lora", "", "lora", "lycoris", None),
        ("extract_kohya_lora", " rank=2", "lora", "kohya", None),
        ("extract_lora_'sd1-kohya_lycoris_lora'", " 2 false 2 null", "lora", "lycoris", None),
        ("extract_lycoris_lokr", "", "lokr", "lycoris", 0.5),
        ("extract_kohya_lokr", " kronecker_ratio=0.25", "lokr", "kohya", 0.25),
        ("extract_lokr_'sd1-kohya_kohya_lora'", " 0.75", "lokr", "kohya", 0.75),
    ],
)
def test_deserialize_deprecated_extraction_method(tmp_path, old_name, arguments, algorithm, adapter_format, ratio):
    checkpoint = tmp_path / "delta.safetensors"
    save_file({WEIGHT: torch.ones((4, 4, 1, 1))}, checkpoint)
    source = (
        "version 0.1.0\n"
        f'model "{checkpoint}" model_config="{CONFIG}" merge_space="delta"\n'
        f'merge "{old_name}" &0{arguments}'
    )
    replacement = f"extract_{algorithm}_{CONFIG}_{adapter_format}_lora"
    with pytest.warns(FutureWarning, match=r"removed in sd-mecha 1\.3\.0") as caught:
        recipe = sd_mecha.deserialize(source)

    assert old_name in str(caught[0].message)
    assert replacement in str(caught[0].message)
    assert recipe.merge_method.get_identifier() == replacement
    if ratio is not None:
        dims = recipe.bound_args.arguments["dims"]
        expected_dims = lokr_dims(recipe.bound_args.arguments["base"], ratio)
        assert sd_mecha.serialize(dims, finalize=False) == sd_mecha.serialize(expected_dims, finalize=False)

    serialized = sd_mecha.serialize(recipe)
    assert old_name not in serialized
    assert replacement in serialized
