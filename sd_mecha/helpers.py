import dataclasses
import logging
import pathlib
import torch
from .streaming import StateDictKeyError
from .extensions.merge_spaces import MergeSpace
from .extensions.model_configs import ModelConfig, resolve as resolve_model_config
from .extensions import merge_methods
from .merging import merge
from .conversion import convert
from .recipe_nodes import ClosedModelRecipeNode, LiteralValue, LiteralRecipeNode, RecipeNode, RecipeNodeOrValue
from typing import Any, Optional, MutableMapping, Mapping


def model(
    state_dict: str | pathlib.Path | Mapping[str, torch.Tensor],
    config: Optional[str | ModelConfig] = None,
    merge_space: str | MergeSpace = "weight",
) -> RecipeNode:
    """
    Create a recipe node representing a state dict.

    Args:
        state_dict:
            Path to a `.safetensors` file or an already loaded state dict.
        config:
            Model config or an identifier thereof.
        merge_space:
            The merge space in which the model is expected to be.

    Returns:
        RecipeNode: A node that can be used in recipe graphs.
    """
    if merge_space is None:
        raise ValueError("merge space cannot be None")

    if isinstance(state_dict, Mapping):
        if state_dict and not isinstance(first_value := next(iter(state_dict.values())), torch.Tensor):
            raise ValueError(f"state dict must contain values of type Tensor, not {type(first_value)}")
        return LiteralRecipeNode(state_dict, model_config=config, merge_space=merge_space)
    if isinstance(state_dict, str):
        state_dict = pathlib.Path(state_dict)
    return ClosedModelRecipeNode(state_dict, model_config=config, merge_space=merge_space)


def literal(
    value: LiteralValue,
    config: Optional[ModelConfig | str] = None,
    merge_space: Optional[str | MergeSpace] = None,
) -> RecipeNode:
    """
    Create a recipe node wrapping tensors or some builtin python objects.

    This is used to access recipe node properties on python objects directly, i.e.:
    ```python
    sd_mecha.literal({...}) | 3
    ```
    There is typically no need to wrap inputs into recipe nodes manually as this function is implicitly applied whenever needed.

    Args:
        value:
            The literal value to wrap into a recipe node.
        config:
            Model config or an identifier thereof.
        merge_space:
            The merge space in which the literal is expected to be.

    Returns:
        RecipeNode: A recipe node representing the literal value.
    """
    if isinstance(config, str):
        config = resolve_model_config(config)

    initial_config = config
    if not isinstance(value, dict):
        value = {"key": value}
        initial_config = resolve_model_config("singleton-mecha")

    res = LiteralRecipeNode(value, model_config=initial_config, merge_space=merge_space)
    if config != initial_config:
        res = merge_methods.resolve("convert_singleton")(res)
    return res


def set_log_level(level: str = "INFO"):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)


@dataclasses.dataclass
class Defaults:
    merge_device: Optional[str | torch.device] = ...,
    merge_dtype: Optional[torch.dtype] = ...,
    output_device: Optional[str | torch.device] = ...,
    output_dtype: Optional[torch.dtype] = ...,
    threads: Optional[int] = ...,
    total_buffer_size: int = ...,
    strict_merge_space: MergeSpace | str = ...,
    strict_mandatory_keys: bool = ...,
    check_extra_keys: bool = ...,
    check_finite_output: bool = ...,
    omit_non_finite_inputs: bool = ...,
    memoize_intermediates: bool = ...,
    validate_mm_contract: bool = ...,
    cache: Mapping[RecipeNode, Any] = ...,
    tqdm: type = ...,
    output: Optional[MutableMapping[str, torch.Tensor]] | pathlib.Path | str = ...,

    def convert(
        self,
        recipe: RecipeNodeOrValue,
        config: str | ModelConfig | RecipeNode,
    ):
        """
        Convert a recipe or model from one model config to another.

        See `sd_mecha.convert` for more information.
        """
        return convert(recipe, config)

    def merge(
        self,
        recipe: RecipeNodeOrValue,
        *,
        merge_device: Optional[str | torch.device] = ...,
        merge_dtype: Optional[torch.dtype] = ...,
        output_device: Optional[str | torch.device] = ...,
        output_dtype: Optional[torch.dtype] = ...,
        threads: Optional[int] = ...,
        total_buffer_size: int = ...,
        strict_merge_space: MergeSpace | str = ...,
        strict_mandatory_keys: bool = ...,
        check_extra_keys: bool = ...,
        check_finite_output: bool = ...,
        omit_non_finite_inputs: bool = ...,
        memoize_intermediates: bool = ...,
        validate_mm_contract: bool = ...,
        cache: Mapping[RecipeNode, Any] = ...,
        tqdm: type = ...,
        output: Optional[MutableMapping[str, torch.Tensor]] | pathlib.Path | str = ...,
    ) -> Optional[MutableMapping[str, torch.Tensor]]:
        """
        Materialize a state dict from a recipe graph and optionally save it to a file.

        See `sd_mecha.merge` for more information.
        """
        if merge_device is ...:
            merge_device = self.merge_device
        if merge_dtype is ...:
            merge_dtype = self.merge_dtype
        if output_device is ...:
            output_device = self.output_device
        if output_dtype is ...:
            output_dtype = self.output_dtype
        if threads is ...:
            threads = self.threads
        if total_buffer_size is ...:
            total_buffer_size = self.total_buffer_size
        if strict_merge_space is ...:
            strict_merge_space = self.strict_merge_space
        if strict_mandatory_keys is ...:
            strict_mandatory_keys = self.strict_mandatory_keys
        if check_extra_keys is ...:
            check_extra_keys = self.check_extra_keys
        if check_finite_output is ...:
            check_finite_output = self.check_finite_output
        if omit_non_finite_inputs is ...:
            omit_non_finite_inputs = self.omit_non_finite_inputs
        if memoize_intermediates is ...:
            memoize_intermediates = self.memoize_intermediates
        if validate_mm_contract is ...:
            validate_mm_contract = self.validate_mm_contract
        if cache is ...:
            cache = self.cache
        if tqdm is ...:
            tqdm = self.tqdm
        if output is ...:
            output = self.output

        return merge(
            recipe,
            merge_device=merge_device,
            merge_dtype=merge_dtype,
            output_device=output_device,
            output_dtype=output_dtype,
            threads=threads,
            total_buffer_size=total_buffer_size,
            strict_merge_space=strict_merge_space,
            strict_mandatory_keys=strict_mandatory_keys,
            check_extra_keys=check_extra_keys,
            check_finite_output=check_finite_output,
            omit_non_finite_inputs=omit_non_finite_inputs,
            memoize_intermediates=memoize_intermediates,
            validate_mm_contract=validate_mm_contract,
            cache=cache,
            tqdm=tqdm,
            output=output,
        )


def skip_key(key: str) -> None:
    """
    Skip merging a key from within a merge method.

    This simply raises StateDictKeyError
    :param key:
    :return:
    """
    raise StateDictKeyError(key)
