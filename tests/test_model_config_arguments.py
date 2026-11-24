from typing import Literal, get_args

import pytest
import torch
from safetensors.torch import save_file
from torch import Tensor

import sd_mecha
from sd_mecha import Parameter, ParameterInfo, Return, StateDict, merge_method, param_ref
from sd_mecha.extensions import merge_methods, model_configs


@pytest.fixture
def configs(monkeypatch):
    monkeypatch.setattr(merge_methods, "_module_state", merge_methods._module_state.copy())
    monkeypatch.setattr(model_configs, "_model_configs_registry_base", model_configs._model_configs_registry_base.copy())
    configs = {}
    for name in ("base", "source", "intermediate", "adapter"):
        config = model_configs.ModelConfigImpl(
            f"config_ref_{name}-mecha",
            {"weights": {name: model_configs.KeyMetadata([1], torch.float32)}},
        )
        model_configs.register(config)
        configs[name] = config
    return configs


def register_converter(identifier, source, target, input_space, output_space):
    source_key, = source.keys()
    target_key, = target.keys()

    @merge_method(identifier=identifier, is_conversion=True)
    class Converter:
        @staticmethod
        def map_keys(builder):
            builder[target_key] = builder.keys[source_key]

        def __call__(self, value: Parameter(StateDict[Tensor], input_space, source)) -> Return(
            Tensor, output_space, target,
        ):
            return value[source_key]

    return Converter


def model_file(tmp_path, config, value, space="weight", infer=False):
    key, = config.keys()
    path = tmp_path / f"{config.identifier}-{space}.safetensors"
    save_file({key: torch.tensor([value], dtype=torch.float32)}, path)
    return sd_mecha.model(path, None if infer else config, space)


@pytest.mark.parametrize(
    "selector, expected_key",
    [(sd_mecha.pick_component, "custom.input"), (sd_mecha.omit_component, "custom.output")],
)
def test_component_selectors_use_literal_choices_from_the_model_config(configs, tmp_path, selector, expected_key):
    config = model_configs.ModelConfigImpl(
        "custom_components-mecha",
        {
            "custom_in": {"custom.input": model_configs.KeyMetadata([1], torch.float32)},
            "custom_out": {"custom.output": model_configs.KeyMetadata([1], torch.float32)},
        },
    )
    model_configs.register(config)
    path = tmp_path / "custom.safetensors"
    save_file({key: torch.ones(1) for key in config.keys()}, path)

    explicit = sd_mecha.model(path, config)
    recipe = selector(explicit, "custom_in")
    assert get_args(recipe.merge_method.get_input_types().args[1]) == ("custom_in", "custom_out")
    restored = sd_mecha.deserialize(sd_mecha.serialize(recipe))
    output = sd_mecha.merge(restored, threads=0)
    assert set(output) - {"__metadata__"} == {expected_key}

    per_key = selector(explicit, {"custom.input": "custom_in", "custom.output": "custom_out"})
    output = sd_mecha.merge(per_key, threads=0)
    expected_keys = set(config.keys()) if selector is sd_mecha.pick_component else set()
    assert set(output) - {"__metadata__"} == expected_keys

    invalid_component = r"Invalid value 'unknown' for parameter 'component'.*custom_in.*custom_out"
    with pytest.raises(ValueError, match=invalid_component):
        selector(explicit, "unknown")

    @merge_method(register=False)
    def compute_component(a: Parameter(Tensor, "weight")) -> Return(str, "param"):
        return "unknown"

    with pytest.raises(ValueError, match=invalid_component):
        sd_mecha.merge(selector(explicit, compute_component(explicit)), threads=0)

    inferred = sd_mecha.model(path)
    assert expected_key in sd_mecha.merge(selector(inferred, "custom_in"), threads=0)
    invalid_recipe = selector(inferred, "unknown")
    with pytest.raises(ValueError, match=invalid_component):
        sd_mecha.merge(invalid_recipe, threads=0)


def test_mapped_literal_type_uses_referenced_argument_metadata(configs, tmp_path):
    def type_for(info):
        assert info.model_config == configs["base"]
        return Literal["gain"] if info.merge_space == "weight" else Literal["difference"]

    @merge_method(register=False)
    def choose_mode(
        a: Parameter(Tensor, {"weight", "delta"}),
        mode: Parameter(param_ref("a").map(type_for)),
    ) -> Return(Tensor, "weight"):
        return a

    source = model_file(tmp_path, configs["base"], 1, "delta")
    assert get_args(choose_mode(source, "difference").merge_method.get_input_types().args[1]) == ("difference",)
    with pytest.raises(ValueError, match="Invalid value 'gain' for parameter 'mode'"):
        choose_mode(source, "gain")


def test_implicit_conversion_skips_incompatible_route(configs, tmp_path):
    base_config, source_config, intermediate = (configs[name] for name in ("base", "source", "intermediate"))
    incompatible = register_converter("ref_incompatible", source_config, base_config, "delta", "delta")
    step_one = register_converter("ref_step_one", source_config, intermediate, "weight", "weight")
    step_two = register_converter("ref_step_two", intermediate, base_config, "weight", "delta")

    @merge_method
    def ref_add(
        base: Parameter(Tensor, "weight"),
        diff: Parameter(Tensor, "delta", param_ref("base").model_config, allow_conversion=True),
    ) -> Return(Tensor, "weight", param_ref("base").model_config):
        return base + diff

    base = model_file(tmp_path, base_config, 3, infer=True)
    source = model_file(tmp_path, source_config, 4)
    explicit = sd_mecha.convert(source, base)
    assert explicit.merge_method is step_two

    recipe = ref_add(base, source)
    converted = recipe.bound_args.arguments["diff"]
    assert converted.merge_method is step_two
    assert converted.bound_args.args[0].merge_method is step_one
    serialized = sd_mecha.serialize(recipe)
    assert incompatible.identifier not in serialized
    assert step_one.identifier in serialized
    assert step_two.identifier in serialized
    assert isinstance(sd_mecha.deserialize(serialized).bound_args.arguments["diff"], sd_mecha.RecipeNode)
    assert sd_mecha.serialize(sd_mecha.deserialize(serialized)) == serialized
    output = sd_mecha.merge(sd_mecha.deserialize(serialized), threads=0)
    torch.testing.assert_close(output["base"], torch.tensor([7.0], dtype=output["base"].dtype))


def test_direct_match_beats_conversion_and_ambiguity_requires_explicit_route(configs, tmp_path):
    base_config, source_config = configs["base"], configs["source"]
    for name in ("ref_choice_a", "ref_choice_b"):
        register_converter(name, source_config, base_config, "weight", "delta")

    @merge_method(register=False)
    def add(
        base: Parameter(Tensor, "weight"),
        diff: Parameter(Tensor, "delta", param_ref("base").model_config, allow_conversion=True),
    ) -> Return(Tensor, "weight"):
        return base + diff

    base = model_file(tmp_path, base_config, 1)
    delta = model_file(tmp_path, base_config, 2, "delta")
    assert add(base, delta).bound_args.arguments["diff"] is delta
    with pytest.raises(TypeError, match=r"Ambiguous implicit conversion.*'diff'.*"):
        add(base, model_file(tmp_path, source_config, 3))


def test_reference_mapping_on_interface_uses_named_argument_and_converts_one_input(configs, tmp_path):
    base_config, adapter_config, source_config = (configs[name] for name in ("base", "adapter", "source"))
    converter = register_converter("ref_convert_base", source_config, base_config, "weight", "weight")
    resolved = []

    def working_config(info: ParameterInfo):
        assert info.merge_space == "weight"
        resolved.append(info.model_config.identifier)
        return (base_config.identifier,)

    @merge_method(is_interface=True)
    def ref_apply(
        base: Parameter(
            Tensor, "weight", param_ref("adapter").map(working_config), allow_conversion=True,
        ),
        adapter: Parameter(Tensor, "weight"),
    ) -> Return(Tensor, "weight", param_ref("base").model_config):
        ...

    @merge_method(implements=ref_apply)
    class ref_apply_impl:
        @staticmethod
        def map_keys(builder):
            builder["base"] = builder.base.keys["base"] & builder.adapter.keys["adapter"]

        def __call__(
            self,
            base: Parameter(Tensor, "weight", base_config),
            adapter: Parameter(Tensor, "weight", adapter_config),
        ) -> Return(Tensor, "weight", base_config):
            return base + adapter

    adapter = model_file(tmp_path, adapter_config, 6)
    base = model_file(tmp_path, source_config, 2)
    recipe = ref_apply(adapter=adapter, base=base)
    assert recipe.merge_method.identifier == ref_apply_impl.identifier
    assert recipe.bound_args.arguments["base"].merge_method is converter
    assert recipe.bound_args.arguments["adapter"] is adapter
    assert resolved == [adapter_config.identifier]
    saved = sd_mecha.serialize(recipe)
    restored = sd_mecha.deserialize(saved)
    assert sd_mecha.serialize(restored) == saved
    output = sd_mecha.merge(restored, threads=0)
    torch.testing.assert_close(output["base"], torch.tensor([8.0], dtype=output["base"].dtype))


def test_registration_checks_reference_and_opt_in(configs):
    config = configs["base"]
    with pytest.raises(TypeError, match=r"param_ref\('missing'\).*parameter"):
        @merge_method(register=False)
        def bad_reference(a: Parameter(Tensor, model_config=param_ref("missing").model_config)) -> Return(Tensor):
            return a

    with pytest.raises(TypeError, match="Only one parameter"):
        @merge_method(register=False)
        def multiple(
            a: Parameter(Tensor, model_config=config, allow_conversion=True),
            b: Parameter(Tensor, model_config=config, allow_conversion=True),
        ) -> Return(Tensor):
            return a + b


@pytest.mark.parametrize("allow_conversion", (False, True))
def test_parameter_cannot_reference_its_own_config(allow_conversion):
    with pytest.raises(TypeError, match=r"Parameter 'a' cannot reference its own model config"):
        @merge_method(register=False)
        def self_reference(
            a: Parameter(Tensor, model_config=param_ref("a").model_config, allow_conversion=allow_conversion),
        ) -> Return(Tensor):
            return a


def test_same_config_conversion_obeys_shared_merge_space(configs, tmp_path):
    config = configs["base"]
    converter = register_converter("ref_same_config", config, config, "weight", "delta")

    def output_config(info: ParameterInfo):
        assert info.merge_spaces == {"delta"}
        return (info.model_config,)

    @merge_method
    def ref_add_shared_space(
        base: Parameter(Tensor),
        diff: Parameter(Tensor, model_config=param_ref("base").model_config, allow_conversion=True),
    ) -> Return(Tensor, model_config=param_ref("diff").map(output_config)):
        return base + diff

    base = model_file(tmp_path, config, 3, "delta")
    source = model_file(tmp_path, config, 5)
    recipe = ref_add_shared_space(base, source)
    assert recipe.bound_args.arguments["diff"].merge_method is converter
    saved = sd_mecha.serialize(recipe)
    restored = sd_mecha.deserialize(saved)
    assert sd_mecha.serialize(restored) == saved
    result = sd_mecha.merge(restored, strict_merge_space="delta", threads=0)
    torch.testing.assert_close(result["base"], torch.tensor([8.0], dtype=result["base"].dtype))


def test_model_reference_requires_opt_in_before_converting(configs, tmp_path):
    target, source = configs["base"], configs["source"]
    register_converter("ref_without_opt_in", source, target, "weight", "delta")

    @merge_method(register=False)
    def add(
        base: Parameter(Tensor, "weight"),
        diff: Parameter(Tensor, "delta", param_ref("base").model_config),
    ) -> Return(Tensor, "weight"):
        return base + diff

    with pytest.raises(TypeError, match=r"Parameter 'diff' requires .*config_ref_base-mecha"):
        add(model_file(tmp_path, target, 1), model_file(tmp_path, source, 2))
