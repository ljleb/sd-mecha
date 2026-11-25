from __future__ import annotations

import warnings
from typing import Iterable

import fuzzywuzzy.process
import torch

from sd_mecha.conversion import convert
from sd_mecha.extensions import merge_methods, model_configs
from sd_mecha.extensions.merge_methods import MergeMethod, Parameter, Return, StateDict, merge_method, param_ref
from sd_mecha.extensions.model_configs import ModelConfig
from sd_mecha.graph_finalization import open_graph
from sd_mecha.keys_map import KeyMapBuilder, KeysAccessor, ReqExpr
from sd_mecha.recipe_nodes import RecipeNode, RecipeNodeOrValue
from sd_mecha.typing_ import subclasses

from ._common import LycorisAlgorithm, LycorisModelConfig, StateDictKeyHelper, _algos_by_name, extraction_interface_name
from .lora import lora
from .lokr import lokr
from .norm import norm
from .loha import loha
from .full import full
from .boft import boft


__all__ = [
    "lora",
    "lokr",
    "norm",
    "loha",
    "full",
    "boft",
    "apply",
    "to_base",
]


def _register_all_lycoris_configs():
    algorithms = subclasses(LycorisAlgorithm)
    for base_config_id in (
        "sdxl-kohya",
        "sdxl-kohya_but_diffusers",
        "sd1-kohya",
        "anima-diffusion_pipe",
        "kolors-diffusers",
        "moss_image-openmoss",
    ):
        base_config = model_configs.resolve(base_config_id)
        lyco_config = LycorisModelConfig(base_config, algorithms)
        model_configs.register_aux(lyco_config)

        define_application(lyco_config)
        define_conversion(lyco_config)
        for algorithm in algorithms:
            if algorithm.extract_method is not None:
                algorithm.implement_extract_method(lyco_config, base_config)


def _adapter_base_config(config: ModelConfig) -> tuple[ModelConfig, ...]:
    return (config.base_config,) if isinstance(config, LycorisModelConfig) else ()


def _adapter_configs_for(base_config: ModelConfig) -> tuple[ModelConfig, ...]:
    return tuple(
        config for config in model_configs.get_all_aux()
        if isinstance(config, LycorisModelConfig) and config.base_config == base_config
    )


def _recipe_model_config(recipe: RecipeNode) -> ModelConfig:
    config = recipe.model_config
    if config is None:
        with open_graph(recipe, root_only=True, solve_merge_space=False) as graph:
            config = graph.root_candidates().model_config.finalize()
    if config.identifier == "structural":
        raise TypeError("Could not infer a known model config. Specify the input's model config explicitly.")
    return config


def to_base(recipe: RecipeNodeOrValue) -> RecipeNode:
    """Convert an adapter to the key format of its associated base model.

    Args:
        recipe: An adapter recipe, or a recipe already in a base model's key format.

    Returns:
        The conversion recipe. Supported additive updates produce a delta. If the input
        already uses the base key format, return it directly. BOFT updates need the base
        weights, so use apply_lycoris with the base checkpoint to apply them.
    """
    recipe = merge_methods.value_to_node(recipe)
    config = _recipe_model_config(recipe)
    return convert(recipe, config.base_config if isinstance(config, LycorisModelConfig) else config)


@merge_method(identifier="apply_lycoris", is_interface=True)
def apply(
    base: Parameter(
        torch.Tensor, "weight", param_ref("lyco").map(lambda info: _adapter_base_config(info.model_config)),
        allow_conversion=True,
    ),
    lyco: Parameter(torch.Tensor, "weight", param_ref("base").map(lambda info: _adapter_configs_for(info.model_config))),
) -> Return(torch.Tensor, "weight", param_ref("base").model_config):
    """Apply an adapter in the model layout it targets.

    Args:
        base: The base model. A registered conversion can bring it into the adapter's layout.
        lyco: The adapter to apply to the base weights. Inference accepts both lycoris_ and lora_ keys.

    Returns:
        A weight-space recipe in the adapter's base layout. Use convert() to select another
        output layout.
    """
    ...


def _adapter_keys(
    keys: KeysAccessor, lyco_config: LycorisModelConfig, base_key: str,
    algorithms: Iterable[type[LycorisAlgorithm]],
) -> ReqExpr | None:
    inputs = None
    for algorithm in algorithms:
        if (algorithm_inputs := algorithm.build_input_keys(keys, lyco_config, base_key)) is not None:
            inputs |= algorithm_inputs
    return inputs


def define_application(lyco_config: LycorisModelConfig):
    lyco_config_id = lyco_config.identifier
    base_config = lyco_config.base_config

    @merge_method(
        identifier=f"apply_lycoris_{lyco_config_id}",
        implements=apply,
        globals=globals(),
        locals=locals(),
    )
    class ApplyLycorisToBase:
        @staticmethod
        def map_keys(b: KeyMapBuilder):
            for base_key in base_config.keys():
                lyco_inputs = _adapter_keys(b.lyco.keys, lyco_config, base_key, lyco_config.algorithms)
                b[base_key] = (lyco_inputs | b.base.keys[()]) & b.base.keys[base_key]

        def __call__(
            self,
            base: Parameter(StateDict[torch.Tensor], "weight", base_config),
            lyco: Parameter(StateDict[torch.Tensor], "weight", lyco_config),
            **kwargs,
        ) -> Return(torch.Tensor, "weight", base_config):
            base_key = kwargs["key"]
            apply_fn = kwargs["key_relation"].meta
            if apply_fn is None:
                return base[base_key]

            sd_helper = StateDictKeyHelper(base, lyco, base_key)
            target_shape = base_config.keys()[base_key].shape
            return apply_fn(sd_helper, target_shape)


def define_conversion(lyco_config: LycorisModelConfig):
    lyco_config_id = lyco_config.identifier
    base_config = lyco_config.base_config
    base_config_id = base_config.identifier
    algorithms = tuple(algo for algo in lyco_config.algorithms if algo.supports_delta_conversion)

    @merge_method(
        identifier=f"convert_'{lyco_config_id}'_to_base",
        is_conversion=True,
        globals=globals(),
        locals=locals(),
    )
    class LycorisToBase:
        @staticmethod
        def map_keys(b: KeyMapBuilder):
            for base_key in base_config.keys():
                lyco_inputs = _adapter_keys(b.keys, lyco_config, base_key, algorithms)
                if lyco_inputs is not None:
                    b[base_key] = lyco_inputs

        @staticmethod
        def _check_first_input_keys(keys, reachable_keys):
            config_keys = lyco_config.keys()
            boft_keys = (key for key in keys if key.endswith(".oft_blocks") and key in config_keys)
            if next(reachable_keys(boft_keys), None) is not None:
                warnings.warn(
                    "Adapter-to-delta conversion skipped BOFT updates because they require the base weights. "
                    "Use sd_mecha.apply_lycoris(base, adapter) to apply them.",
                    stacklevel=3,
                )

        def __call__(
            self,
            lyco: Parameter(StateDict[torch.Tensor], "weight", lyco_config_id),
            **kwargs,
        ) -> Return(torch.Tensor, "delta", base_config_id):
            base_key = kwargs["key"]
            apply_fn = kwargs["key_relation"].meta

            sd_helper = StateDictKeyHelper(None, lyco, base_key)
            target_shape = base_config.keys()[base_key].shape
            return apply_fn(sd_helper, target_shape)


def resolve(identifier: str) -> MergeMethod:
    """Look up the extraction method for an adapter algorithm.

    Args:
        identifier: The algorithm name, such as `lora`, `lokr`, `norm`, or `boft`.

    Returns:
        The merge method that builds an extraction recipe. BOFT takes the base and target weights.
        LoRA, LoKr, and Norm take a weight difference.

    Raises:
        KeyError: If the algorithm has no registered extraction method.
    """
    if identifier in _algos_by_name:
        return merge_methods.resolve(extraction_interface_name(identifier))

    suggestions = fuzzywuzzy.process.extractOne(identifier, _algos_by_name.keys())
    postfix = ""
    if suggestions is not None:
        postfix = f". Nearest match is '{suggestions[0]}'"
    raise KeyError(f"unknown lycoris algorithm: {identifier}{postfix}")


_register_all_lycoris_configs()
