import dataclasses
import functools
import logging
import pathlib
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from sd_mecha.model_detection import DetermineConfigVisitor
from sd_mecha.recipe_nodes import RecipeVisitor
from sd_mecha.streaming import InSafetensorsDict, OutSafetensorsDict
from sd_mecha import extensions, recipe_nodes, recipe_serializer
from tqdm import tqdm
from typing import Optional, Mapping, MutableMapping, Dict, Set


class RecipeMerger:
    def __init__(
        self, *,
        models_dir: Optional[pathlib.Path | str] = None,
        default_device: str = "cpu",
        default_dtype: Optional[torch.dtype] = torch.float64,
    ):
        if isinstance(models_dir, str):
            models_dir = pathlib.Path(models_dir)
        if models_dir is not None:
            models_dir = models_dir.absolute()
        self.__base_dir = models_dir

        self.__default_device = default_device
        self.__default_dtype = default_dtype

    def merge_and_save(
        self, recipe: extensions.merge_method.RecipeNodeOrPath, *,
        output: MutableMapping[str, torch.Tensor] | pathlib.Path | str = "merge",
        fallback_model: Optional[Mapping[str, torch.Tensor] | recipe_nodes.ModelRecipeNode | pathlib.Path | str] = None,
        save_dtype: Optional[torch.dtype] = torch.float16,
        threads: Optional[int] = None,
        total_buffer_size: int = 2**28,
    ):
        recipe = extensions.merge_method.path_to_node(recipe)
        if recipe.merge_space != recipe_nodes.MergeSpace.BASE:
            raise ValueError(f"recipe should be in model merge space, not {str(recipe.merge_space).split('.')[-1]}")
        if isinstance(fallback_model, (str, pathlib.Path)):
            fallback_model = extensions.merge_method.path_to_node(fallback_model)
        elif not isinstance(fallback_model, (recipe_nodes.ModelRecipeNode, Mapping, type(None))):
            raise ValueError(f"fallback_model should be a simple model or None, not {type(fallback_model)}")
        extensions.merge_method.clear_model_paths_cache()

        fallback_is_recipe = isinstance(fallback_model, recipe_nodes.ModelRecipeNode)
        fallback_in_recipe = fallback_is_recipe and fallback_model in recipe
        total_files_open = (
            recipe.accept(recipe_nodes.ModelsCountVisitor()) +
            int(isinstance(output, (str, pathlib.Path))) +
            int(fallback_is_recipe and not fallback_in_recipe)
        )
        buffer_size_per_file = total_buffer_size // total_files_open
        if threads is None:
            threads = total_files_open

        load_input_dicts_visitor = LoadInputDictsVisitor(
            self.__base_dir,
            buffer_size_per_file,
        )
        recipe.accept(load_input_dicts_visitor)
        if fallback_is_recipe:
            fallback_model.accept(load_input_dicts_visitor)

        model_config = recipe.accept(DetermineConfigVisitor())
        if fallback_is_recipe:
            model_config = model_config.intersect(fallback_model.accept(DetermineConfigVisitor()))
            fallback_model = fallback_model.state_dict

        output = self.__normalize_output_to_dict(
            output,
            model_config.get_minimal_dummy_header(),
            model_config.get_keys_to_merge(),
            recipe_serializer.serialize(recipe),
            buffer_size_per_file // threads,
            save_dtype,
        )

        progress = tqdm(total=len(model_config.keys()), desc="Merging recipe")
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for key in model_config.keys():
                key_merger = model_config.get_key_merger(key, recipe, fallback_model, self.__default_device, self.__default_dtype)
                key_merger = self.__track_output(key_merger, output, key, save_dtype or self.__default_dtype)
                key_merger = self.__track_progress(key_merger, key, model_config.get_shape(key), progress)
                futures.append(executor.submit(key_merger))

            for future in as_completed(futures):
                future.result()

        progress.close()
        if isinstance(output, OutSafetensorsDict):
            output.close()

    def __normalize_output_to_dict(
        self,
        output: MutableMapping[str, torch.Tensor] | pathlib.Path | str,
        merged_header: Dict[str, dict],
        keys_to_merge: Set[str],
        serialized_recipe: str,
        buffer_size_per_thread: int,
        dtype: torch.dtype,
    ):
        if isinstance(output, (str, pathlib.Path)):
            if not isinstance(output, pathlib.Path):
                output = pathlib.Path(output)
            if not output.is_absolute():
                output = self.__base_dir / output
            if not output.suffix:
                output = output.with_suffix(".safetensors")
            logging.info(f"Saving to {output}")

            output = OutSafetensorsDict(
                output,
                merged_header,
                keys_to_merge,
                serialized_recipe,
                buffer_size_per_thread,
                dtype,
            )
        return output

    def __track_progress(self, f, key, key_shape, progress):
        @functools.wraps(f)
        def track_progress(*args, **kwargs):
            progress.set_postfix({"key": key, "shape": key_shape})
            res = f(*args, **kwargs)
            progress.update()
            return res
        return track_progress

    def __track_output(self, f, output, key, save_dtype):
        @functools.wraps(f)
        def track_output(*args, **kwargs):
            output[key] = f(*args, **kwargs).to(save_dtype)
        return track_output


@dataclasses.dataclass
class LoadInputDictsVisitor(RecipeVisitor):
    __base_dir: pathlib.Path
    __buffer_size_per_dict: int

    def visit_model(self, node: recipe_nodes.ModelRecipeNode):
        node.state_dict = self.__load_dict(node)

    def visit_parameter(self, _node: recipe_nodes.ParameterRecipeNode):
        return

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode):
        for model in node.models:
            model.accept(self)

    def __load_dict(
        self,
        node: recipe_nodes.ModelRecipeNode,
    ) -> InSafetensorsDict:
        if node.state_dict is not None:
            return node.state_dict

        path = node.path
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        if not path.is_absolute():
            path = self.__base_dir / path
        if not path.suffix:
            path = path.with_suffix(".safetensors")
        return InSafetensorsDict(path, self.__buffer_size_per_dict)
