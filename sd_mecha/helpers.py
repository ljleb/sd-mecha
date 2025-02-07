import logging
import pathlib
import torch
import tqdm
from .extensions.model_configs import ModelConfig
from .recipe_merging import merge_and_save
from .conversion import convert
from .recipe_serializer import serialize
from .recipe_nodes import ModelRecipeNode, LiteralRecipeNode, RecipeNode, RecipeNodeOrValue
from .extensions.merge_methods import NonDictLiteralValue
from typing import Optional, List, MutableMapping, Iterable


def serialize_and_save(
    recipe: RecipeNode,
    output_path: pathlib.Path | str,
):
    serialized = serialize(recipe)

    if not isinstance(output_path, pathlib.Path):
        output_path = pathlib.Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".mecha")
    output_path = output_path.absolute()

    logging.info(f"Saving recipe to {output_path}")
    with open(output_path, "w") as f:
        f.write(serialized)


def model(path: str | pathlib.Path, config: Optional[str] = None, merge_space: str = "weight") -> ModelRecipeNode:
    if isinstance(path, str):
        path = pathlib.Path(path)
    return ModelRecipeNode(path, model_config=config, merge_space=merge_space)


def literal(value: NonDictLiteralValue | dict, config: Optional[str] = None) -> LiteralRecipeNode:
    return LiteralRecipeNode(value, model_config=config)


def set_log_level(level: str = "INFO"):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)


class RecipeMerger:
    def __init__(
        self,
        model_dirs: Optional[pathlib.Path | str | List[pathlib.Path | str]] = ...,
        merge_device: Optional[str] = ...,
        merge_dtype: Optional[torch.dtype] = ...,
        output_device: Optional[str | torch.device] = ...,
        output_dtype: Optional[torch.dtype] = torch.float16,
        threads: Optional[int] = None,
        total_buffer_size: int = 2**28,
        strict_weight_space: bool = True,
        check_finite: bool = True,
        tqdm: type = tqdm,
    ):
        self.__model_dirs = model_dirs
        self.__merge_device = merge_device
        self.__merge_dtype = merge_dtype
        self.__output_device = output_device
        self.__output_dtype = output_dtype
        self.__threads = threads
        self.__total_buffer_size = total_buffer_size
        self.__strict_weight_space = strict_weight_space
        self.__check_finite = check_finite
        self.__tqdm = tqdm

    def convert(self, recipe: RecipeNode, config: str | ModelConfig | RecipeNode):
        return convert(recipe, config, self._model_dirs_to_pathlib_list(self.__model_dirs))

    def merge_and_save(
        self,
        recipe: RecipeNodeOrValue,
        output: MutableMapping[str, torch.Tensor] | pathlib.Path | str = ...,
        fallback_model: Optional[RecipeNodeOrValue] = ...,
        merge_device: Optional[str] = ...,
        merge_dtype: Optional[torch.dtype] = ...,
        output_device: Optional[str | torch.device] = ...,
        output_dtype: Optional[torch.dtype] = ...,
        threads: Optional[int] = ...,
        total_buffer_size: int = ...,
        model_dirs: Iterable[pathlib.Path] = ...,
        strict_weight_space: bool = ...,
        check_finite: bool = ...,
        tqdm: type = ...,
    ):
        if merge_device is None:
            merge_device = self.__merge_device
        if merge_dtype is None:
            merge_dtype = self.__merge_dtype
        if output_device is None:
            output_device = self.__output_device
        if output_dtype is None:
            output_dtype = self.__output_dtype
        if threads is None:
            threads = self.__threads
        if total_buffer_size is None:
            total_buffer_size = self.__total_buffer_size
        if model_dirs is None:
            model_dirs = self._model_dirs_to_pathlib_list(self.__model_dirs)
        if strict_weight_space is None:
            strict_weight_space = self.__strict_weight_space
        if check_finite is None:
            check_finite = self.__check_finite

        return merge_and_save(
            recipe,
            output,
            fallback_model,
            merge_device,
            merge_dtype,
            output_device,
            output_dtype,
            threads,
            total_buffer_size,
            model_dirs,
            strict_weight_space,
            check_finite,
            tqdm,
        )

    @staticmethod
    def _model_dirs_to_pathlib_list(models_dir):
        if models_dir is None:
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
