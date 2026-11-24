import pathlib
import tempfile
import typing
import pytest
import safetensors.torch
import torch
import sd_mecha
from typing import TypeVar, Mapping
from sd_mecha import merge_method, Parameter, Return, StateDict, StateDictKeyError


A = TypeVar("A")
B = TypeVar("B")


def assert_equal_in_merge_method(expected: A, actual_literal: B, t: type[A] | TypeVar):
    return_t = next(iter(typing.get_args(t))) if typing.get_args(t) else t

    @merge_method(register=False)
    def compare_value(
        actual: Parameter(t, model_config="sdxl-supermerger_blocks"),
        **kwargs,
    ) -> Return(return_t, model_config="sdxl-supermerger_blocks"):
        nonlocal expected

        try:
            if isinstance(expected, Mapping):
                expected_value = expected[kwargs["key"]]
            else:
                expected_value = expected
        except KeyError as e:
            raise StateDictKeyError(str(e)) from e

        if isinstance(actual, Mapping):
            actual = actual[kwargs["key"]]

        if isinstance(actual, torch.Tensor):
            assert torch.allclose(expected_value, actual)
        else:
            assert actual == expected_value and isinstance(actual, type(expected_value))

        return actual

    sd_mecha.merge(
        compare_value(actual_literal),
        strict_merge_space=None,
        threads=0,
        merge_device=None,
        merge_dtype=None,
        output_device=None,
        output_dtype=None,
    )


def test_value_to_node__float_to_tensor():
    actual = 1.0
    expected = torch.tensor(1.0)
    assert_equal_in_merge_method(expected, actual, torch.Tensor)


def test_value_to_node__str():
    actual = "hello!"
    expected = "hello!"
    with pytest.warns(DeprecationWarning, match=r"str as a merge method parameter type.*removed in sd-mecha 1\.3\.0"):
        assert_equal_in_merge_method(expected, actual, str)


def test_bare_str_parameter_is_deprecated():
    with pytest.warns(DeprecationWarning, match=r"removed in sd-mecha 1\.3\.0"):
        @merge_method(register=False)
        def echo(message: str) -> Return(str, "param"):
            return message


@pytest.mark.parametrize("annotation", [typing.Literal[()], typing.Literal[1], typing.Literal["a", 1]])
def test_literal_parameter_requires_string_choices(annotation):
    with pytest.raises(TypeError, match="one or more string choices"):
        Parameter(annotation)


def test_literal_parameter_validates_defaults():
    with pytest.raises(ValueError, match="Invalid value 'typo' for parameter 'mode'"):
        @merge_method(register=False)
        def invalid_default(mode: Parameter(typing.Literal["a"]) = "typo") -> Return(str, "param"):
            return mode


def test_literal_parameter_validates_arguments():
    @merge_method(register=False)
    def choose(*, mode: typing.Literal["a", "b"]) -> Return(str, "param"):
        return mode

    assert choose(mode=sd_mecha.literal("a")).merge_method is choose
    singleton = sd_mecha.literal("a", config="singleton-mecha", merge_space="param")
    assert choose(mode=singleton).bound_args.kwargs["mode"].bound_args.args[0] is singleton
    with pytest.raises(ValueError, match="Invalid value 'typo' for parameter 'mode'.*'a', 'b'"):
        choose(mode="typo")

    for value in (1, pathlib.Path("a"), {"key": "a"}, sd_mecha.literal({"IN00": "a"}), choose(mode="a")):
        with pytest.raises(TypeError, match="Parameter 'mode' requires a (constant )?string"):
            choose(mode=value)


def test_literal_parameter_dispatch_prefers_the_narrower_implementation(monkeypatch):
    from sd_mecha.extensions import merge_methods

    monkeypatch.setattr(merge_methods, "_module_state", merge_methods._module_state.copy())

    @merge_method(is_interface=True)
    def select_mode(mode: Parameter(typing.Literal["a", "b"]) = "a") -> Return(str, "param"):
        ...

    @merge_method(implements=select_mode)
    def broad_mode(mode: Parameter(typing.Literal["a", "b"])) -> Return(str, "param"):
        return mode

    @merge_method(implements=select_mode)
    def narrow_mode(mode: Parameter(typing.Literal["a"])) -> Return(str, "param"):
        return mode

    assert select_mode().merge_method is narrow_mode
    assert select_mode("b").merge_method is broad_mode
    with pytest.raises(ValueError, match="Invalid value 'typo' for parameter 'mode'"):
        select_mode("typo")

    with pytest.raises(TypeError, match="Parameter 'mode' must declare a Literal subset"):
        @merge_method(implements=select_mode)
        def unrelated_mode(mode: Parameter(typing.Literal["a", "c"])) -> Return(str, "param"):
            return mode

    @merge_method(implements=select_mode)
    def other_narrow_mode(mode: Parameter(typing.Literal["a"])) -> Return(str, "param"):
        return mode

    with pytest.raises(TypeError, match="Ambiguous implementations.*narrow_mode, other_narrow_mode"):
        select_mode("a")


def test_cast_dtype_choices_round_trip(tmp_path):
    path = tmp_path / "weights.safetensors"
    reference_path = tmp_path / "reference.safetensors"
    safetensors.torch.save_file({"IN00": torch.ones(1), "IN01": torch.ones(1)}, path)
    safetensors.torch.save_file({"IN00": torch.ones(1, dtype=torch.float64), "IN01": torch.ones(1)}, reference_path)
    base = sd_mecha.model(path, "sdxl-supermerger_blocks")
    reference = sd_mecha.model(reference_path, "sdxl-supermerger_blocks", merge_space="param")

    for dtype, expected in (
        (None, (torch.float32, torch.float32)),
        ("float64", (torch.float64, torch.float64)),
        (torch.float64, (torch.float64, torch.float64)),
        ({"IN00": "float64", "IN01": "float32"}, (torch.float64, torch.float32)),
        (sd_mecha.get_dtype(reference), (torch.float64, torch.float32)),
    ):
        recipe = base.to(dtype=dtype)
        restored = sd_mecha.deserialize(sd_mecha.serialize(recipe, finalize=False))
        result = sd_mecha.merge(restored, threads=0, merge_dtype=None, output_dtype=None)
        assert (result["IN00"].dtype, result["IN01"].dtype) == expected


def test_cast_device_choices_include_only_visible_accelerator_indices(monkeypatch):
    from sd_mecha.extensions.builtin.merge_methods.logistics import _cast_device_choices

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    devices = _cast_device_choices()
    assert {"cpu", "cuda", "cuda:0", "cuda:1"} <= set(devices)
    assert "cuda:2" not in devices
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert "cuda" not in _cast_device_choices()


def test_cast_dtype_validates_values_at_recipe_creation_and_merge():
    base = sd_mecha.model({"IN00": torch.ones(1)}, "sdxl-supermerger_blocks")

    with pytest.raises(ValueError, match="Invalid value 'fp32' for parameter 'dtype'"):
        sd_mecha.cast(base, dtype="fp32")
    with pytest.raises(TypeError, match="Parameter 'dtype' requires a string"):
        sd_mecha.cast(base, dtype=32)
    with pytest.raises(ValueError, match=r"Invalid value 'fp32' for parameter .*IN00"):
        sd_mecha.cast(base, dtype={"IN00": "fp32"})

    @merge_method(register=False)
    def invalid_dtype(a: Parameter(torch.Tensor, "weight")) -> Return(str, "param"):
        return "fp32"

    recipe = sd_mecha.cast(base, dtype=invalid_dtype(base))
    with pytest.raises(ValueError, match="Invalid value 'fp32' for parameter 'dtype'"):
        sd_mecha.merge(recipe, threads=0)


def test_value_to_node__float_dict_to_tensor():
    actual = {"IN00": float(1.0), "IN01": float(2.0)}
    expected = {"IN00": torch.tensor(1.0), "IN01": torch.tensor(2.0)}
    assert_equal_in_merge_method(expected, actual, torch.Tensor)


def test_value_to_node__float_dict_to_tensor_dict():
    actual = {"IN00": float(1.0), "IN01": float(2.0)}
    expected = {"IN00": torch.tensor(1.0), "IN01": torch.tensor(2.0)}
    assert_equal_in_merge_method(expected, actual, StateDict[torch.Tensor])


def test_value_to_node__int_dict_to_tensor_dict():
    actual = {"IN00": int(1), "IN01": int(2)}
    expected = {"IN00": torch.tensor(1.0), "IN01": torch.tensor(2.0)}
    assert_equal_in_merge_method(expected, actual, StateDict[torch.Tensor])


def test_value_to_node__tensor_dict_to_tensor_dict():
    actual = {"IN00": torch.tensor(1.0), "IN01": torch.tensor(2.0)}
    expected = {"IN00": torch.tensor(1.0), "IN01": torch.tensor(2.0)}
    assert_equal_in_merge_method(expected, actual, StateDict[torch.Tensor])


def test_value_to_node__float_to_int():
    actual = float(1.5)
    expected = int(1)
    assert_equal_in_merge_method(expected, actual, int)


def test_value_to_node__int_to_float():
    actual = int(1)
    expected = float(1.0)
    assert_equal_in_merge_method(expected, actual, float)


def test_value_to_node__path_to_tensor():
    tmp = tempfile.mktemp(suffix=".safetensors")
    actual = pathlib.Path(tmp)
    try:
        expected = {"IN00": torch.tensor(0.0)}
        safetensors.torch.save_file(expected, tmp)
        assert_equal_in_merge_method(expected, actual, torch.Tensor)
    finally:
        pathlib.Path(tmp).unlink(missing_ok=True)


def test_value_to_node__str_dict_to_str():
    actual = {"IN00": "hello!", "IN01": "hello2!"}
    expected = {"IN00": "hello!", "IN01": "hello2!"}
    assert_equal_in_merge_method(expected, actual, str)


def test_value_to_node__inconsistent_dict_type():
    value = {"IN00": torch.tensor(1.0), "IN01": 2.0}
    with pytest.raises(TypeError):
        assert_equal_in_merge_method(value, value, torch.Tensor)


def test_value_to_node__path_to_str():
    tmp = tempfile.mktemp(suffix=".safetensors")
    actual = pathlib.Path(tmp)
    expected = tmp
    with pytest.raises(TypeError):
        assert_equal_in_merge_method(expected, actual, str)


def test_value_to_node__str_dict_to_str_dict():
    actual = {"IN00": "hello!", "IN01": "hello2!"}
    expected = {"IN00": "hello!", "IN01": "hello2!"}
    assert_equal_in_merge_method(expected, actual, StateDict[str])


T = TypeVar("T")


def test_value_to_node__int_to_type_var():
    actual = {"IN00": 1, "IN01": 2}
    expected = {"IN00": 1, "IN01": 2}
    assert_equal_in_merge_method(expected, actual, T)


def test_value_to_node__int_to_type_var_dict():
    actual = {"IN00": 1, "IN01": 2}
    expected = {"IN00": 1, "IN01": 2}
    assert_equal_in_merge_method(expected, actual, StateDict[T])


def test_value_to_node__tensor_to_float():
    actual = {"IN00": torch.tensor(1.0), "IN01": torch.tensor(2.0)}
    expected = {"IN00": float(1.0), "IN01": float(2.0)}
    assert_equal_in_merge_method(expected, actual, float)


def test_value_to_node__tensor_to_int():
    actual = {"IN00": torch.tensor(1), "IN01": torch.tensor(2)}
    expected = {"IN00": int(1), "IN01": int(2)}
    assert_equal_in_merge_method(expected, actual, int)


def test_value_to_node__path_to_type_var():
    tmp = tempfile.mktemp(suffix=".safetensors")
    actual = pathlib.Path(tmp)
    try:
        expected = {"IN00": torch.tensor(0.0)}
        safetensors.torch.save_file(expected, tmp)
        assert_equal_in_merge_method(expected, actual, T)
    finally:
        pathlib.Path(tmp).unlink(missing_ok=True)


def test_value_to_node__path_to_type_var_dict():
    tmp = tempfile.mktemp(suffix=".safetensors")
    actual = pathlib.Path(tmp)
    try:
        expected = {"IN00": torch.tensor(0.0)}
        safetensors.torch.save_file(expected, tmp)
        assert_equal_in_merge_method(expected, actual, StateDict[T])
    finally:
        pathlib.Path(tmp).unlink(missing_ok=True)
