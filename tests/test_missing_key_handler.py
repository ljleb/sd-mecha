import torch
import pytest
import sd_mecha
from safetensors.torch import load_file, save_file
from sd_mecha.extensions import model_configs
from sd_mecha.streaming import OutSafetensorsDict


def _register_aux(config):
    try:
        model_configs.register_aux(config)
    except ValueError:
        pass


config_without_handler = model_configs.ModelConfigImpl(
    "testmissing-withouthandler",
    {
        "component": model_configs.ModelComponent({
            "key": model_configs.KeyMetadata([2, 2], torch.float32),
        }),
    },
)
_register_aux(config_without_handler)


config_with_handler = model_configs.ModelConfigImpl(
    "testmissing-withhandler",
    {
        "component": model_configs.ModelComponent({
            "key": model_configs.KeyMetadata([2, 2], torch.float32),
        }),
    },
    missing_key_handler="tests.stubs.missing_key_handler.reconstruct_test_key",
)
_register_aux(config_with_handler)


def test_model_config_rejects_optional_generated_keys():
    with pytest.raises(ValueError, match="cannot be both optional and generated"):
        model_configs.from_yaml("""
identifier: testmissing-invalid
components:
  component:
    key: {shape: [1], dtype: int64, optional: true, generated: true}
""")


def test_merge_without_missing_key_handler_still_skips_non_finite_key():
    recipe = sd_mecha.model(
        {"key": torch.full((2, 2), float("nan"), dtype=torch.float32)},
        config=config_without_handler,
    )

    output = sd_mecha.merge(
        recipe,
        threads=0,
        merge_device=None,
        merge_dtype=None,
        output_device="cpu",
        output_dtype=torch.float32,
    )

    assert "key" not in output


def test_merge_uses_missing_key_handler_for_non_finite_key():
    recipe = sd_mecha.model(
        {"key": torch.full((2, 2), float("nan"), dtype=torch.float32)},
        config=config_with_handler,
    )

    output = sd_mecha.merge(
        recipe,
        threads=0,
        merge_device=None,
        merge_dtype=None,
        output_device="cpu",
        output_dtype=torch.float64,
    )

    assert torch.equal(output["key"], torch.full((2, 2), 42, dtype=torch.float64))


def test_merge_uses_missing_key_handler_for_key_absent_from_the_plan():
    recipe = sd_mecha.literal({}, config=config_with_handler)
    output = sd_mecha.merge(recipe, threads=0, output_dtype=torch.float64)

    torch.testing.assert_close(output["key"], torch.full((2, 2), 42., dtype=torch.float64))


@pytest.mark.parametrize("config_id", ["sdxl-sgm", "sdxl-kohya", "sdxl-kohya_but_diffusers"])
def test_merge_fills_missing_sdxl_final_clip_l_layer(config_id):
    if config_id == "sdxl-sgm":
        clip_l = "conditioner.embedders.0.transformer.text_model"
        clip_g_final_layer = "conditioner.embedders.1.model.transformer.resblocks.31."
        clip_g_existing = f"{clip_g_final_layer}ln_1.bias"
    else:
        clip_l = "te1.text_model"
        clip_g_final_layer = "te2.text_model.encoder.layers.31."
        clip_g_existing = f"{clip_g_final_layer}layer_norm1.bias"

    config = model_configs.resolve(config_id)
    keys = config.keys()
    clip_l_final_layer = f"{clip_l}.encoder.layers.11."
    clip_l_existing = f"{clip_l_final_layer}layer_norm1.bias"
    positions = f"{clip_l}.embeddings.position_ids"
    recipe = sd_mecha.model({
        clip_l_existing: torch.ones(keys[clip_l_existing].shape),
        clip_g_existing: torch.ones(keys[clip_g_existing].shape),
        positions: torch.full((1, 77), -1, dtype=torch.int64),
    }, config)
    output = sd_mecha.merge(recipe, threads=0, output_dtype=torch.float16)

    layer_keys = [key for key in keys if key.startswith(clip_l_final_layer)]
    assert len(layer_keys) == 16
    for key in layer_keys:
        expected = torch.full(keys[key].shape, float(key == clip_l_existing), dtype=torch.float16)
        torch.testing.assert_close(output[key], expected)

    assert {key for key in output if key.startswith(clip_g_final_layer)} == {clip_g_existing}
    torch.testing.assert_close(output[clip_g_existing], torch.ones(keys[clip_g_existing].shape, dtype=torch.float16))
    torch.testing.assert_close(output[positions], torch.arange(77, dtype=torch.int64)[None, :])
    assert f"{clip_l}.encoder.layers.10.layer_norm1.bias" not in output


def test_merge_regenerates_sd1_position_ids_from_aliased_input():
    alias = "cond_stage_model.transformer.embeddings.position_ids"
    key = "cond_stage_model.transformer.text_model.embeddings.position_ids"
    recipe = sd_mecha.model({alias: torch.full((1, 77), -1, dtype=torch.int64)}, "sd1-ldm")
    output = sd_mecha.merge(recipe, threads=0, output_dtype=torch.float16)

    torch.testing.assert_close(output[key], torch.arange(77, dtype=torch.int64)[None, :])
    assert alias not in output


@pytest.mark.parametrize("config_id", ["moss_image-openmoss", "moss_image-comfyui"])
def test_merge_regenerates_moss_image_rope_bands_in_float32(config_id):
    key = "rope.bands"
    alias = "transformer.rope.bands"
    recipe = sd_mecha.model({alias: torch.zeros(16)}, config_id)
    output = sd_mecha.merge(recipe, threads=0, output_dtype=torch.float16)

    expected = 1 / (10000.0 ** (torch.arange(16, dtype=torch.float32) / 16))
    torch.testing.assert_close(output[key], expected, atol=0, rtol=0)
    assert alias not in output


@pytest.mark.parametrize("include_positions", [False, True])
def test_generated_keys_are_ignored_on_input_and_written_with_the_configured_dtype(tmp_path, monkeypatch, include_positions):
    prefix = "conditioner.embedders.0.transformer.text_model"
    positions = f"{prefix}.embeddings.position_ids"
    missing = f"{prefix}.encoder.layers.11.layer_norm1.bias"
    non_finite = f"{prefix}.encoder.layers.11.layer_norm2.bias"
    config = model_configs.ModelConfigImpl(
        "testmissing-generated",
        {"clip_l": {
            positions: model_configs.KeyMetadata([1, 3], torch.int64, aliases=("positions",), generated=True),
            missing: model_configs.KeyMetadata([3], torch.float32, optional=True),
            non_finite: model_configs.KeyMetadata([3], torch.float32),
            "ordinary": model_configs.KeyMetadata([3], torch.float32),
        }},
        missing_key_handler=(
            "sd_mecha.extensions.builtin.model_configs.sdxl_missing_keys."
            "sdxl_sgm_missing_key_handler"
        ),
    )
    _register_aux(config)
    weights = {"ordinary": torch.ones(3), non_finite: torch.full((3,), float("inf"))}
    if include_positions:
        weights["positions"] = torch.tensor([[2049, 2050, 2051]], dtype=torch.int64)
    source = tmp_path / "source.safetensors"
    save_file(weights, source)
    model = sd_mecha.model(source, config)

    @sd_mecha.merge_method(register=False)
    def override(a: sd_mecha.Parameter(torch.Tensor), **kwargs) -> sd_mecha.Return(torch.Tensor):
        assert kwargs["key"] != positions
        return a + 1

    recipe = override(model)
    with sd_mecha.open_graph(recipe) as graph:
        plan = graph.finalize_with_keys()
    assert plan.node_to_keys[plan.root] == {"ordinary", non_finite}
    assert all(positions not in keys and "positions" not in keys for keys in plan.node_to_keys.values())

    estimates = {}
    init_buffer = OutSafetensorsDict._init_buffer

    def capture_header(writer, header):
        estimates.update(header)
        return init_buffer(writer, header)

    monkeypatch.setattr(OutSafetensorsDict, "_init_buffer", capture_header)
    target = tmp_path / "result.safetensors"
    sd_mecha.merge(recipe, threads=0, strict_mandatory_keys=True, output_dtype=torch.float16, output=target)

    output = load_file(target)
    assert estimates[positions].dtype == torch.int64
    assert estimates["ordinary"].dtype == torch.float16
    torch.testing.assert_close(output[positions], torch.arange(3, dtype=torch.int64).reshape(1, 3))
    torch.testing.assert_close(output[missing], torch.zeros(3, dtype=torch.float16))
    torch.testing.assert_close(output[non_finite], torch.zeros(3, dtype=torch.float16))
    torch.testing.assert_close(output["ordinary"], torch.full((3,), 2., dtype=torch.float16))


def test_generated_keys_are_discarded_from_multi_output_key_groups():
    config = model_configs.ModelConfigImpl(
        "testmissing-generated_group",
        {"component": {
            "key": model_configs.KeyMetadata([2, 2], torch.float32, generated=True),
            "ordinary": model_configs.KeyMetadata([2, 2], torch.float32),
        }},
        missing_key_handler="tests.stubs.missing_key_handler.reconstruct_test_key",
    )
    _register_aux(config)

    @sd_mecha.merge_method(register=False)
    class group_outputs:
        @staticmethod
        def map_keys(builder):
            builder["key", "ordinary"] = builder.keys["ordinary"]

        def __call__(self, ordinary: sd_mecha.Parameter(torch.Tensor)) -> sd_mecha.Return(sd_mecha.StateDict[torch.Tensor]):
            return {"key": ordinary, "ordinary": ordinary + 1}

    model = sd_mecha.model({"ordinary": torch.zeros(2, 2)}, config)
    output = sd_mecha.merge(group_outputs(model), threads=0, output_dtype=torch.float32, strict_mandatory_keys=True)
    torch.testing.assert_close(output["key"], torch.full((2, 2), 42.))
    torch.testing.assert_close(output["ordinary"], torch.ones(2, 2))


def test_merge_accepts_a_literal_root():
    recipe = sd_mecha.literal(
        {"key": torch.full((2, 2), 7, dtype=torch.float32)},
        config=config_without_handler,
    )

    output = sd_mecha.merge(
        recipe,
        threads=0,
        merge_device=None,
        merge_dtype=None,
        output_device="cpu",
        output_dtype=torch.float32,
    )

    assert torch.equal(output["key"], torch.full((2, 2), 7, dtype=torch.float32))


def test_merge_rejects_empty_result_unless_allowed():
    recipe = sd_mecha.literal({}, config=config_without_handler)

    with pytest.raises(RuntimeError, match="produces no output keys"):
        sd_mecha.merge(recipe, threads=0, merge_device=None, merge_dtype=None)

    output = sd_mecha.merge(
        recipe,
        threads=0,
        merge_device=None,
        merge_dtype=None,
        allow_empty_merge=True,
    )
    assert set(output) == {"__metadata__"}
