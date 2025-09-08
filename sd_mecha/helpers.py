import logging
import pathlib
import torch

from .streaming import StateDictKeyError
from .extensions.merge_spaces import MergeSpace
from .extensions.model_configs import ModelConfig
from .merging import merge
from .conversion import convert
from .recipe_nodes import ModelRecipeNode, LiteralRecipeNode, RecipeNode, RecipeNodeOrValue
from .extensions.merge_methods import NonDictLiteralValue
from typing import Optional, List, MutableMapping, Iterable, Mapping


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
        ModelRecipeNode: A node that can be used in recipe graphs.
    """
    if merge_space is None:
        raise ValueError("merge space cannot be None")

    if isinstance(state_dict, Mapping):
        if state_dict and not isinstance(first_value := next(iter(state_dict.values())), torch.Tensor):
            raise ValueError(f"state dict must contain values of type Tensor, not {type(first_value)}")
        return LiteralRecipeNode(state_dict, model_config=config, merge_space=merge_space)
    if isinstance(state_dict, str):
        state_dict = pathlib.Path(state_dict)
    return ModelRecipeNode(state_dict, model_config=config, merge_space=merge_space)


def literal(
    value: NonDictLiteralValue | dict,
    config: Optional[ModelConfig | str] = None,
    merge_space: Optional[str | MergeSpace] = None,
) -> LiteralRecipeNode:
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
        LiteralRecipeNode: A recipe node representing the literal value.
    """
    return LiteralRecipeNode(value, model_config=config, merge_space=merge_space)


def set_log_level(level: str = "INFO"):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)


class Defaults:
    """
    Convenience class for common recipe operations to reduce repetition in recipe scripts.
    """

    def __init__(
        self,
        model_dirs: pathlib.Path | str | Iterable[pathlib.Path | str] = ...,
        merge_device: Optional[str | torch.device] = ...,
        merge_dtype: Optional[torch.dtype] = ...,
        output_device: Optional[str | torch.device] = ...,
        output_dtype: Optional[torch.dtype] = ...,
        threads: Optional[int] = ...,
        total_buffer_size: int = ...,
        strict_weight_space: bool = ...,
        check_finite: bool = ...,
        omit_extra_keys: bool = ...,
        omit_ema: bool = ...,
        check_mandatory_keys: bool = ...,
        tqdm: type = ...,
    ):
        """
        Args:
            See documentation for `sd_mecha.merge` or `sd_mecha.conversion` for a description of each parameter.
        """
        self.__model_dirs = model_dirs
        self.__merge_device = merge_device
        self.__merge_dtype = merge_dtype
        self.__output_device = output_device
        self.__output_dtype = output_dtype
        self.__threads = threads
        self.__total_buffer_size = total_buffer_size
        self.__strict_weight_space = strict_weight_space
        self.__check_finite = check_finite
        self.__omit_extra_keys = omit_extra_keys
        self.__omit_ema = omit_ema
        self.__check_mandatory_keys = check_mandatory_keys
        self.__tqdm = tqdm

    def convert(
        self,
        recipe: RecipeNodeOrValue,
        config: str | ModelConfig | RecipeNode,
        model_dirs: pathlib.Path | str | Iterable[pathlib.Path | str] = ...,
    ):
        """
        Convert a recipe or model from one model config to another.

        See `sd_mecha.convert` for more information.
        """
        if model_dirs is ...:
            model_dirs = self.__model_dirs
        model_dirs = self._model_dirs_to_pathlib_list(model_dirs)
        return convert(recipe, config, model_dirs)

    def merge(
        self,
        recipe: RecipeNodeOrValue,
        *,
        fallback_model: Optional[RecipeNodeOrValue] = ...,
        merge_device: Optional[str | torch.device] = ...,
        merge_dtype: Optional[torch.dtype] = ...,
        output_device: Optional[str | torch.device] = ...,
        output_dtype: Optional[torch.dtype] = ...,
        threads: Optional[int] = ...,
        total_buffer_size: int = ...,
        model_dirs: pathlib.Path | str | Iterable[pathlib.Path | str] = ...,
        strict_weight_space: bool = ...,
        check_finite: bool = ...,
        omit_extra_keys: bool = ...,
        omit_ema: bool = ...,
        check_mandatory_keys: bool = ...,
        tqdm: type = ...,
        output: MutableMapping[str, torch.Tensor] | pathlib.Path | str = ...,
    ) -> Optional[MutableMapping[str, torch.Tensor]]:
        """
        Materialize a state dict from a recipe graph and optionally save it to a file.

        See `sd_mecha.merge` for more information.
        """
        if merge_device is ...:
            merge_device = self.__merge_device
        if merge_dtype is ...:
            merge_dtype = self.__merge_dtype
        if output_device is ...:
            output_device = self.__output_device
        if output_dtype is ...:
            output_dtype = self.__output_dtype
        if threads is ...:
            threads = self.__threads
        if total_buffer_size is ...:
            total_buffer_size = self.__total_buffer_size
        if model_dirs is ...:
            model_dirs = self.__model_dirs
        model_dirs = self._model_dirs_to_pathlib_list(model_dirs)
        if strict_weight_space is ...:
            strict_weight_space = self.__strict_weight_space
        if check_finite is ...:
            check_finite = self.__check_finite
        if omit_extra_keys is ...:
            omit_extra_keys = self.__omit_extra_keys
        if omit_ema is ...:
            omit_ema = self.__omit_ema
        if check_mandatory_keys is ...:
            check_mandatory_keys = self.__check_mandatory_keys

        return merge(
            recipe,
            fallback_model=fallback_model,
            merge_device=merge_device,
            merge_dtype=merge_dtype,
            output_device=output_device,
            output_dtype=output_dtype,
            threads=threads,
            total_buffer_size=total_buffer_size,
            model_dirs=model_dirs,
            strict_weight_space=strict_weight_space,
            check_finite=check_finite,
            omit_extra_keys=omit_extra_keys,
            omit_ema=omit_ema,
            check_mandatory_keys=check_mandatory_keys,
            tqdm=tqdm,
            output=output,
        )

    @staticmethod
    def _model_dirs_to_pathlib_list(models_dir):
        if models_dir is ...:
            models_dir = []
        if not isinstance(models_dir, List):
            models_dir = [models_dir]
        models_dir = list(models_dir)
        for i in range(len(models_dir)):
            if isinstance(models_dir[i], str):
                models_dir[i] = pathlib.Path(models_dir[i])
            if models_dir[i] is not None:
                models_dir[i] = models_dir[i].absolute()

        return models_dir


def skip_key(key: str) -> None:
    """
    Skip merging a key from within a merge method.

    This simply raises StateDictKeyError
    :param key:
    :return:
    """
    raise StateDictKeyError(key)
