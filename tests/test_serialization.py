import math
import warnings

import pytest
import torch
from safetensors.torch import save_file

import sd_mecha
from sd_mecha.extensions import merge_methods


def test_serializing_device_casts_warns_once_and_preserves_them(tmp_path):
    base = sd_mecha.model(tmp_path / "model.safetensors", "sd1-kohya")
    recipe = base.to(device="cpu").to(device=torch.device("cpu"), dtype="float64")
    path = tmp_path / "recipe.mecha"

    with pytest.warns(UserWarning, match="contains explicit device casts.*remap_devices=None") as caught:
        saved = sd_mecha.serialize(recipe, finalize=False, output=path)

    assert len(caught) == 1
    assert path.read_text() == saved
    assert '"cpu"' in saved
    assert '"float64"' in saved
    with pytest.warns(UserWarning, match="contains explicit device casts"):
        assert sd_mecha.serialize(sd_mecha.deserialize_path(path), finalize=False) == saved


@pytest.mark.parametrize("device_argument", ['"remote:9"', 'device="remote:9"', "&2"])
def test_deserialize_can_override_devices_before_validation(tmp_path, device_argument):
    saved = "\n".join((
        "version 0.1.0",
        'model "base.safetensors" model_config="sd1-kohya" merge_space="weight"',
        'dict key="remote:9"',
        'literal &1 model_config="singleton-mecha" merge_space="param"',
        f'merge "cast" &0 {device_argument} dtype="float64"',
        'merge "cast" &3 device="remote:8"',
    ))
    with pytest.raises(ValueError, match="Invalid value 'remote:9' for parameter 'device'"):
        sd_mecha.deserialize(saved)

    path = tmp_path / "device-casts.mecha"
    path.write_text(saved)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cleared = sd_mecha.serialize(sd_mecha.deserialize_path(path, remap_devices=None), finalize=False)
    with pytest.warns(UserWarning, match="contains explicit device casts"):
        replaced = sd_mecha.serialize(sd_mecha.deserialize(saved, remap_devices="cpu"), finalize=False)
    assert '"remote:' not in cleared + replaced
    assert '"float64"' in cleared and '"float64"' in replaced
    assert '"cpu"' not in cleared and '"cpu"' in replaced
    assert cleared.count('merge "cast"') == 1
    assert replaced.count('merge "cast"') == 2

    with pytest.raises(ValueError, match="Invalid value 'remote:7' for parameter 'device'"):
        sd_mecha.deserialize(saved, remap_devices="remote:7")

    with pytest.raises(ValueError, match="Invalid value 'remote:8' for parameter 'device'"):
        sd_mecha.deserialize(saved, remap_devices={"remote:9": "cpu"})
    remapped = sd_mecha.deserialize_path(path, remap_devices={"remote:9": "cpu", "remote:8": "cpu:0"})
    with pytest.warns(UserWarning, match="contains explicit device casts"):
        remapped_saved = sd_mecha.serialize(remapped, finalize=False)
    assert '"remote:' not in remapped_saved
    assert all(f'"{value}"' in remapped_saved for value in ("cpu", "cpu:0", "float64"))


def test_remap_devices_rewrites_per_key_literals_without_chaining():
    saved = "\n".join((
        "version 0.1.0",
        'model "base.safetensors" model_config="sdxl-supermerger_blocks" merge_space="weight"',
        'dict IN00="remote:9" IN01="cpu" IN02="cpu:0"',
        'literal &1 model_config="sdxl-supermerger_blocks" merge_space="param"',
        'merge "cast" &0 device=&2',
    ))
    recipe = sd_mecha.deserialize(saved, remap_devices={"remote:9": "cpu", "cpu": "cpu:0"})
    devices = recipe.bound_args.arguments["device"]
    assert devices.value_dict == {"IN00": "cpu", "IN01": "cpu:0", "IN02": "cpu:0"}
    assert devices.model_config.identifier == "sdxl-supermerger_blocks"
    assert devices.merge_space == "param"


def test_remap_devices_requires_string_pairs():
    with pytest.raises(TypeError, match="remap_devices must map device name strings to device name strings"):
        sd_mecha.deserialize("version 0.1.0", remap_devices={"cpu": None})


def test_device_override_leaves_dtype_only_casts_alone(tmp_path):
    base = sd_mecha.model(tmp_path / "base.safetensors", "sd1-kohya")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        saved = sd_mecha.serialize(base.to(dtype="float64"), finalize=False)
        restored = sd_mecha.deserialize(saved, remap_devices="cpu")
        assert sd_mecha.serialize(restored, finalize=False) == saved


def test_rotate_recipe_with_default_legacy_alignment_is_migrated():
    saved = '\n'.join((
        'version 0.1.0',
        'model "a.safetensors" model_config="sd1-kohya" merge_space="weight"',
        'model "b.safetensors" model_config="sd1-kohya" merge_space="weight"',
        'dict key=1.0',
        'literal &2 model_config="singleton-mecha" merge_space=null',
        'merge "rotate" &0 &1 &3 0.25 0.5 1e-8 100',
    ))

    with pytest.warns(FutureWarning, match=r"alignment parameter has no effect.*1\.1\.8.*raise an error.*1\.3\.0"):
        recipe = sd_mecha.deserialize(saved)

    assert tuple(recipe.bound_args.arguments) == ("a", "b", "alpha", "centralization")
    migrated = sd_mecha.serialize(recipe, finalize=False)
    assert sd_mecha.serialize(sd_mecha.deserialize(migrated), finalize=False) == migrated


def test_rotate_recipe_ignores_custom_legacy_alignment(tmp_path):
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    expected = torch.tensor([[0., -1.], [1., 0.]])
    save_file({"matrix": torch.eye(2)}, a)
    save_file({"matrix": expected}, b)
    saved = '\n'.join((
        'version 0.1.0',
        f'model "{a}" merge_space="weight"',
        f'model "{b}" merge_space="weight"',
        'merge "rotate" &0 &1 alignment=0.5 alpha=0.0 centralization=0.0',
    ))

    with pytest.warns(FutureWarning, match=r"alignment parameter has no effect.*1\.1\.8.*raise an error.*1\.3\.0"):
        recipe = sd_mecha.deserialize(saved)
    restored = sd_mecha.deserialize(sd_mecha.serialize(recipe, finalize=False))

    output = sd_mecha.merge(restored, output_dtype=torch.float32, threads=0)
    torch.testing.assert_close(output["matrix"], expected)


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
        ("moss_image-diffusers", "moss_image-openmoss"),
        ("moss_image-diffusers_lycoris_lora", "moss_image-openmoss_lycoris_lora"),
        ("moss_image-diffusers_kohya_lora", "moss_image-openmoss_lycoris_lora"),
    ],
)
def test_deprecated_config_alias_round_trips_as_canonical(tmp_path, old_config, new_config):
    checkpoint = tmp_path / "model.safetensors"
    canonical = sd_mecha.serialize(sd_mecha.model(checkpoint, new_config))
    legacy = canonical.replace(f'model_config="{new_config}"', f'model_config="{old_config}"')

    with pytest.warns(FutureWarning, match=rf"'{old_config}'.*removed in sd-mecha 1\.3\.0"):
        restored = sd_mecha.deserialize(legacy)

    assert restored.model_config.identifier == new_config
    assert sd_mecha.serialize(restored) == canonical


@pytest.mark.parametrize(
    "method, sources, extra_args",
    [
        ("convert_moss_image_openmoss_to_comfyui", (("openmoss", "weight"),), ()),
        ("convert_moss_image_comfyui_to_openmoss", (("comfyui", "weight"),), ()),
        ("convert_moss_image_blocks_to_openmoss", (("blocks", "param"),), ()),
        ("convert_'moss_image-openmoss_lycoris_lora'_to_base", (("openmoss_lycoris_lora", "weight"),), ()),
        ("apply_lycoris_moss_image-openmoss_lycoris_lora", (("openmoss", "weight"), ("openmoss_lycoris_lora", "weight")), ()),
        ("extract_lora_moss_image-openmoss_lycoris_lora", (("openmoss", "delta"),), (1, False, 1, 0)),
        ("extract_lokr_moss_image-openmoss_lycoris_lora", (("openmoss", "delta"),), ()),
        ("extract_norm_moss_image-openmoss_lycoris_lora", (("openmoss", "delta"),), ()),
    ],
)
def test_deprecated_moss_method_round_trips_as_canonical(tmp_path, method, sources, extra_args):
    inputs = [
        sd_mecha.model(tmp_path / f"{layout}.safetensors", f"moss_image-{layout}", merge_space)
        for layout, merge_space in sources
    ]
    if method.startswith("extract_lokr_"):
        extra_args = (merge_methods.resolve("lokr_dims")(inputs[0]),)
    recipe = merge_methods.resolve(method)(*inputs, *extra_args)
    canonical = sd_mecha.serialize(recipe, finalize=False)
    legacy_method = method.replace("openmoss", "diffusers")
    legacy = canonical.replace(f'merge "{method}"', f'merge "{legacy_method}"')

    with pytest.warns(FutureWarning, match=r"Recipe merge method .* will be removed in sd-mecha 1\.3\.0") as caught:
        restored = sd_mecha.deserialize(legacy)

    assert legacy_method in str(caught[0].message)
    assert method in str(caught[0].message)
    assert sd_mecha.serialize(restored, finalize=False) == canonical


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
