import math

import pytest

import sd_mecha


@pytest.mark.parametrize("filename", ["model.safetensors", "model epoch=5.safetensors"])
def test_model_path_round_trips_through_recipe_file(tmp_path, filename):
    checkpoint = tmp_path / filename
    recipe_file = tmp_path / "merge.mecha"
    sd_mecha.serialize(sd_mecha.model(checkpoint, "sd1-kohya"), output=recipe_file)

    assert sd_mecha.deserialize_path(recipe_file).path == checkpoint


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
