import math

import pytest

import sd_mecha


@pytest.mark.parametrize("filename", ["model.safetensors", "model epoch=5.safetensors"])
def test_model_path_round_trips_through_recipe_file(tmp_path, filename):
    checkpoint = tmp_path / filename
    recipe_file = tmp_path / "merge.mecha"
    sd_mecha.serialize(sd_mecha.model(checkpoint, "sd1-kohya"), output=recipe_file)

    assert sd_mecha.deserialize_path(recipe_file).path == checkpoint


@pytest.mark.parametrize(
    "old_config, new_config",
    [
        ("anima-diffuser", "anima-diffusion_pipe"),
        ("anima-diffuser_kohya_lora", "anima-diffusion_pipe_lycoris_lora"),
        ("anima-diffuser_kohya_lokr", "anima-diffusion_pipe_lycoris_lora"),
        ("anima-diffuser_kohya_loha", "anima-diffusion_pipe_lycoris_lora"),
    ],
)
def test_deprecated_anima_config_alias_round_trips_as_canonical(tmp_path, old_config, new_config):
    checkpoint = tmp_path / "anima.safetensors"
    canonical = sd_mecha.serialize(sd_mecha.model(checkpoint, new_config))
    legacy = canonical.replace(f'model_config="{new_config}"', f'model_config="{old_config}"')

    with pytest.warns(FutureWarning, match=rf"'{old_config}'.*removed in sd-mecha 1\.3\.0"):
        restored = sd_mecha.deserialize(legacy)

    assert restored.model_config.identifier == new_config
    assert sd_mecha.serialize(restored) == canonical


@pytest.mark.parametrize("algorithm", ["lora", "lokr", "loha"])
@pytest.mark.parametrize("argument", ["&0", "lora=&0"])
def test_deprecated_anima_adapter_conversion_round_trips_as_canonical(tmp_path, algorithm, argument):
    checkpoint = tmp_path / "anima-adapter.safetensors"
    old_config = f"anima-diffuser_kohya_{algorithm}"
    new_config = "anima-diffusion_pipe_lycoris_lora"
    old_method = f"convert_'{old_config}'_to_base"
    new_method = f"convert_'{new_config}'_to_base"
    canonical_model = sd_mecha.serialize(sd_mecha.model(checkpoint, new_config))
    legacy_model = canonical_model.replace(f'model_config="{new_config}"', f'model_config="{old_config}"')

    with pytest.warns(FutureWarning, match=r"removed in sd-mecha 1\.3\.0") as caught:
        restored = sd_mecha.deserialize(f'{legacy_model}\nmerge "{old_method}" {argument}')

    assert len(caught) == 2
    assert old_method in str(caught[1].message)
    assert new_method in str(caught[1].message)
    assert sd_mecha.serialize(restored, finalize=False) == f'{canonical_model}\nmerge "{new_method}" &0'


def test_kohya_application_recipe_uses_canonical_implementation(tmp_path):
    base = sd_mecha.serialize(sd_mecha.model(tmp_path / "base.safetensors", "sd1-kohya"))
    kohya = "sd1-kohya_kohya_lora"
    lycoris = "sd1-kohya_lycoris_lora"
    old_method = f"apply_lycoris_{kohya}"
    adapter = f'model "{tmp_path / "adapter.safetensors"}" model_config="{kohya}" merge_space="weight"'
    with pytest.warns(FutureWarning) as caught:
        restored = sd_mecha.deserialize(f'{base}\n{adapter}\nmerge "{old_method}" &0 &1')
    assert kohya in str(caught[0].message)
    assert old_method in str(caught[1].message)

    saved = sd_mecha.serialize(restored, finalize=False)
    assert kohya not in saved
    assert f"apply_lycoris_{lycoris}" in saved
    assert sd_mecha.serialize(sd_mecha.deserialize(saved), finalize=False) == saved


@pytest.mark.parametrize("source_is_legacy", (False, True))
def test_deprecated_adapter_prefix_converters_restore_the_canonical_input(tmp_path, source_is_legacy):
    old = "sd1-kohya_kohya_lora"
    canonical = "sd1-kohya_lycoris_lora"
    source, target = (old, canonical) if source_is_legacy else (canonical, old)
    method = f"convert_'{source}'_to_'{target}'"
    checkpoint = tmp_path / "adapter.safetensors"
    saved = sd_mecha.serialize(sd_mecha.model(checkpoint, canonical))
    legacy = saved.replace(f'model_config="{canonical}"', f'model_config="{source}"')

    with pytest.warns(FutureWarning) as caught:
        restored = sd_mecha.deserialize(f'{legacy}\nmerge "{method}" &0')

    assert method in str(caught[-1].message)
    assert sd_mecha.serialize(restored) == saved


def test_deserialize_accepts_blank_lines_between_instructions():
    recipe = sd_mecha.literal({"key": 7}, config="singleton-mecha", merge_space="param")
    serialized = sd_mecha.serialize(recipe)
    with_blank_lines = "\n \n".join(serialized.splitlines()) + "\n"

    assert sd_mecha.deserialize(with_blank_lines).value_dict == {"key": 7}


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_float_round_trips(value):
    recipe = sd_mecha.literal({"key": value}, config="singleton-mecha", merge_space="param")

    restored = sd_mecha.deserialize(sd_mecha.serialize(recipe)).value_dict["key"]

    if math.isnan(value):
        assert math.isnan(restored)
    else:
        assert restored == value
