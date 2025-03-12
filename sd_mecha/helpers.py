import logging
import pathlib
import torch
from .extensions.model_configs import ModelConfig
from .recipe_merging import merge
from .conversion import convert
from .recipe_nodes import ModelRecipeNode, LiteralRecipeNode, RecipeNode, RecipeNodeOrValue
from .extensions.merge_methods import NonDictLiteralValue
from typing import Optional, List, MutableMapping, Iterable


def model(path: str | pathlib.Path, config: Optional[ModelConfig | str] = None, merge_space: str = "weight") -> ModelRecipeNode:
    """
    Create a `ModelRecipeNode` referring to a safetensors file or directory of a diffusers model.

    Args:
        path (str or Path):
            Path to a `.safetensors` file (or directory in certain formats).
        config (str, optional):
            A model config identifier if known, or None to let `sd_mecha` infer it.
        merge_space (str):
            The merge space in which the model is expected to be ("weight" by default).

    Returns:
        ModelRecipeNode: A node that can be used in recipe graphs.
    """
    if isinstance(path, str):
        path = pathlib.Path(path)
    return ModelRecipeNode(path, model_config=config, merge_space=merge_space)


def literal(value: NonDictLiteralValue | dict, config: Optional[ModelConfig | str] = None) -> LiteralRecipeNode:
    """
    Wrap raw python objects into a literal recipe node with an optional model config.

    Useful when you need to use recipe node properties on python objects directly, i.e.:
    ```python
    sd_mecha.literal({...}) | 3
    ```
    It is implicitly used when passing a dictionary or a single scalar as input to merge methods.

    Args:
        value:
            The literal data to wrap.
        config (str, optional):
            Model config or an identifier thereof, if relevant.

    Returns:
        LiteralRecipeNode: A recipe node representing the literal value.
    """
    return LiteralRecipeNode(value, model_config=config)


def set_log_level(level: str = "INFO"):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)


class Defaults:
    """
    Convenience wrapper around common recipe operations with custom default values.
    """

    def __init__(
        self,
        model_dirs: pathlib.Path | str | Iterable[pathlib.Path | str] = ...,
        merge_device: Optional[str] = ...,
        merge_dtype: Optional[torch.dtype] = ...,
        output_device: Optional[str | torch.device] = ...,
        output_dtype: Optional[torch.dtype] = torch.float16,
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
            merge_device (optional):
                Device to load intermediate tensors onto while merging (e.g., "cpu" or "cuda").
            merge_dtype (optional):
                Torch dtype for intermediate merges (e.g., `torch.float32`, `torch.float64`).
            output_device (optional):
                Final output device (e.g., "cpu").
            output_dtype (optional):
                Final dtype for the saved model.
            threads (optional):
                Number of threads to spawn for parallel merges. Defaults to a reasonable guess.
            total_buffer_size (optional):
                Total byte size of the buffers for all safetensors state dicts (input and output).
            model_dirs (optional):
                One or more directories to search for model files if `recipe` references relative paths.
            strict_weight_space (optional):
                If True, verifies that merges occur in "weight" space. If False, merges can happen
                in other merge spaces (like "delta" or "param").
            check_finite (optional):
                If True, warns if any non-finite values appear in the final merged tensors.
            tqdm (optional):
                A custom progress-bar factory. By default, uses `tqdm.tqdm`.
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

        This is a convenience wrapper for `sd_mecha.convert`, using the paths set in `model_dirs`.

        Args:
            recipe:
                A `RecipeNode` or dictionary representing the input model or partial recipe.
            config (str or ModelConfig or RecipeNode):
                The desired output config, or a node referencing that config.
            model_dirs (Iterable[Path], optional):
                Directories to resolve relative model paths, if needed.

        Returns:
            A new recipe node describing the entire conversion path. If no path is found, raises `ValueError`.
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
            check_mandatory_keys=check_mandatory_keys,
            omit_ema=omit_ema,
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
