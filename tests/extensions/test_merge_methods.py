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
        strict_weight_space=False,
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
    assert_equal_in_merge_method(expected, actual, str)


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
