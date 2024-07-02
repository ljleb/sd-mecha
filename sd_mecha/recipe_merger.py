import copy
import dataclasses
import functools
import gc
import logging
import operator
import pathlib
import threading
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from sd_mecha.model_detection import DetermineConfigVisitor
from sd_mecha.recipe_nodes import RecipeVisitor
from sd_mecha.streaming import InSafetensorsDict, OutSafetensorsDict
from sd_mecha import extensions, recipe_nodes, recipe_serializer, builtin_model_types
from tqdm import tqdm
from typing import Optional, Mapping, MutableMapping, Dict, Set, List, Iterable, Tuple


class RecipeMerger:
    def __init__(
        self, *,
        models_dir: Optional[pathlib.Path | str | List[pathlib.Path | str]] = None,
        default_device: str = "cpu",
        default_dtype: Optional[torch.dtype] = torch.float64,
        tqdm: type = tqdm,
    ):
        if models_dir is None:
            models_dir = []
        if not isinstance(models_dir, List):
            models_dir = [models_dir]
        for i in range(len(models_dir)):
            if isinstance(models_dir[i], str):
                models_dir[i] = pathlib.Path(models_dir[i])
            if models_dir[i] is not None:
                models_dir[i] = models_dir[i].absolute()

        self.__base_dirs = models_dir
        self.__default_device = default_device
        self.__default_dtype = default_dtype
        self.__tqdm = tqdm

    def merge_and_save(
        self, recipe: extensions.merge_method.RecipeNodeOrPath, *,
        output: MutableMapping[str, torch.Tensor] | pathlib.Path | str = "merge",
        fallback_model: Optional[Mapping[str, torch.Tensor] | recipe_nodes.ModelRecipeNode | pathlib.Path | str] = None,
        save_device: Optional[str] = "cpu",
        save_dtype: Optional[torch.dtype] = torch.float16,
        threads: Optional[int] = None,
        total_buffer_size: int = 2 ** 28,
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
            self.__base_dirs,
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

        thread_local_data = threading.local()
        progress = self.__tqdm(total=len(model_config.keys()), desc="Merging recipe")
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for key in model_config.keys():
                key_merger = model_config.get_key_merger(key, recipe, fallback_model, self.__default_device, self.__default_dtype)
                key_merger = self.__track_output(key_merger, output, key, save_dtype, save_device)
                key_merger = self.__track_progress(key_merger, key, model_config.get_shape(key), progress)
                key_merger = self.__wrap_thread_context(key_merger, thread_local_data)
                futures.append(executor.submit(key_merger))

            for future in as_completed(futures):
                if future.exception() is not None:
                    for future_to_cancel in futures:
                        future_to_cancel.cancel()
                    raise future.exception()
                future.result()

        progress.close()
        if isinstance(output, OutSafetensorsDict):
            output.close()
        recipe.accept(CloseInputDictsVisitor())

        gc.collect()
        torch.cuda.empty_cache()

    def merge_and_save_lora(
        self, recipe: extensions.merge_method.RecipeNodeOrPath, *,
        output: MutableMapping[str, torch.Tensor] | pathlib.Path | str = "merge",
        rank: int = 32,
        conv_rank: Optional[int] = None,
        svd_device: Optional[str] = None,
        svd_dtype: Optional[torch.dtype] = None,
        save_device: Optional[str] = "cpu",
        save_dtype: Optional[torch.dtype] = torch.float16,
        threads: Optional[int] = None,
        total_buffer_size: int = 2 ** 28,
    ):
        if recipe.model_arch.identifier not in ["sd1", "sdxl"]:
            raise ValueError("currently, only the sd1 and sdxl architectures are supported for lora extraction")

        if conv_rank is None:
            conv_rank = rank

        recipe = extensions.merge_method.path_to_node(recipe)
        if recipe.merge_space != recipe_nodes.MergeSpace.DELTA:
            raise ValueError(f"recipe should be in delta merge space, not {str(recipe.merge_space).split('.')[-1]}")
        extensions.merge_method.clear_model_paths_cache()

        total_files_open = (
            recipe.accept(recipe_nodes.ModelsCountVisitor()) +
            int(isinstance(output, (str, pathlib.Path)))
        )
        buffer_size_per_file = total_buffer_size // total_files_open
        if threads is None:
            threads = total_files_open

        load_input_dicts_visitor = LoadInputDictsVisitor(
            self.__base_dirs,
            buffer_size_per_file,
        )
        recipe.accept(load_input_dicts_visitor)

        model_config = recipe.accept(DetermineConfigVisitor())
        minimal_dummy_header = get_lora_header_from_model_header(model_config.get_minimal_dummy_header(), rank, conv_rank, recipe.model_arch.identifier)
        keys_to_merge = get_lora_keys_from_model_keys(model_config.get_keys_to_merge(), recipe.model_arch.identifier, prefix_only=False).intersection(set(minimal_dummy_header))

        output = self.__normalize_output_to_dict(
            output,
            minimal_dummy_header,
            keys_to_merge,
            recipe_serializer.serialize(recipe),
            buffer_size_per_file // threads,
            save_dtype,
        )

        thread_local_data = threading.local()
        keys_to_process = filter_lora_keys(list(model_config.keys()), recipe.model_arch.identifier)
        progress = self.__tqdm(total=len(keys_to_process), desc="Merging recipe")
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for key in keys_to_process:
                key_merger = model_config.get_key_merger(key, recipe, None, self.__default_device, self.__default_dtype)
                key_merger = self.__track_lora_output(key_merger, output, key, rank, conv_rank, recipe.model_arch.identifier, svd_dtype, svd_device, save_dtype, save_device)
                key_merger = self.__track_progress(key_merger, key, model_config.get_shape(key), progress)
                key_merger = self.__wrap_thread_context(key_merger, thread_local_data)
                futures.append(executor.submit(key_merger))

            for future in as_completed(futures):
                if future.exception() is not None:
                    for future_to_cancel in futures:
                        future_to_cancel.cancel()
                    raise future.exception()
                future.result()

        progress.close()
        if isinstance(output, OutSafetensorsDict):
            output.close()
        recipe.accept(CloseInputDictsVisitor())

        gc.collect()
        torch.cuda.empty_cache()

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
                output = self.__base_dirs[0] / output
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

    def __track_output(self, f, output, key, save_dtype, save_device):
        if save_dtype is None:
            save_dtype = self.__default_dtype

        if save_device is None:
            to_kwargs = {"dtype": save_dtype},
        else:
            to_kwargs = {"dtype": save_dtype, "device": save_device}

        @functools.wraps(f)
        def track_output(*args, **kwargs):
            output[key] = f(*args, **kwargs).to(**to_kwargs)

        return track_output

    def __track_lora_output(self, f, output, key, rank, conv_rank, model_arch, svd_dtype, svd_device, save_dtype, save_device):
        save_to_kwargs = {}
        if save_dtype is not None:
            save_to_kwargs["dtype"] = save_dtype
        if save_device is not None:
            save_to_kwargs["device"] = save_device

        svd_to_kwargs = {}
        if svd_dtype is not None:
            svd_to_kwargs["dtype"] = svd_dtype
        if svd_device is not None:
            svd_to_kwargs["device"] = svd_device

        @functools.wraps(f)
        def track_lora_output(*args, **kwargs):
            full_rank_outputs = f(*args, **kwargs).to(**svd_to_kwargs)
            is_true_conv = len(full_rank_outputs.shape) == 4 and functools.reduce(operator.mul, full_rank_outputs.shape[-2:]) != 1

            lora_keys = builtin_model_types.get_lora_keys_from_model_key(key, model_arch=model_arch)
            for i, lora_key in enumerate(lora_keys):
                dim = full_rank_outputs.shape[0] // len(lora_keys)
                t_start = dim * i
                t_end = dim * (i + 1)
                full_rank_output = full_rank_outputs[t_start:t_end]

                up, down, alpha = extract_lora_up_down(full_rank_output, conv_rank if is_true_conv else rank)
                output[f"{lora_key}.lora_up.weight"] = up.to(**save_to_kwargs)
                output[f"{lora_key}.lora_down.weight"] = down.to(**save_to_kwargs)
                output[f"{lora_key}.alpha"] = alpha.to(**save_to_kwargs)

        return track_lora_output

    def __wrap_thread_context(self, f, ctx):
        @functools.wraps(f)
        def thread_context(*args, **kwargs):
            if torch.cuda.is_available():
                if not hasattr(ctx, 'cuda_stream'):
                    ctx.cuda_stream = torch.cuda.Stream()
                with torch.cuda.stream(ctx.cuda_stream):
                    return f(*args, **kwargs)
            else:
                return f(*args, **kwargs)

        return thread_context


def filter_lora_keys(keys, model_arch: str):
    keys = copy.deepcopy(keys)
    for k in keys.copy():
        try:
            builtin_model_types.get_lora_keys_from_model_key(k, model_arch)
        except KeyError:
            if isinstance(keys, dict):
                del keys[k]
            else:
                keys.remove(k)
    return keys


def get_lora_keys_from_model_keys(keys, model_arch: str, prefix_only: bool = True):
    lora_keys = set()
    for k in keys:
        try:
            new_keys = builtin_model_types.get_lora_keys_from_model_key(k, model_arch)
            lora_keys |= set(new_keys)
        except KeyError:
            continue

    if not prefix_only:
        full_lora_keys = set()
        for lora_key in lora_keys:
            full_lora_keys.add(f"{lora_key}.lora_up.weight")
            full_lora_keys.add(f"{lora_key}.lora_down.weight")
            full_lora_keys.add(f"{lora_key}.alpha")
        lora_keys = full_lora_keys

    return lora_keys


def get_lora_header_from_model_header(header: dict, rank: int, conv_rank: int, model_arch: str):
    lora_header = {}
    for k, v in header.items():
        if len(v["shape"]) < 2:
            continue

        try:
            new_keys = builtin_model_types.get_lora_keys_from_model_key(k, model_arch)
        except KeyError:
            continue

        is_true_conv = len(v["shape"]) == 4 and functools.reduce(operator.mul, v["shape"][-2:]) != 1
        for new_key in new_keys:
            up_v = copy.deepcopy(v)
            lora_header[f"{new_key}.lora_up.weight"] = up_v
            up_v["shape"] = [v["shape"][0] // len(new_keys), conv_rank if is_true_conv else rank, *[1] * len(v["shape"][2:])]

            down_v = copy.deepcopy(v)
            lora_header[f"{new_key}.lora_down.weight"] = down_v
            down_v["shape"] = [conv_rank if is_true_conv else rank, v["shape"][1], *v["shape"][2:]]

            alpha_v = copy.deepcopy(v)
            lora_header[f"{new_key}.alpha"] = alpha_v
            alpha_v["shape"] = [1]

    return lora_header


def extract_lora_up_down(a: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    is_conv = len(a.shape) == 4
    kernel_size = (1, 1)
    input_dim = a.shape[1]
    if is_conv:
        kernel_size = a.shape[-2:]

    if a.device.type.startswith("cuda"):
        driver = "gesvd"
    else:
        driver = None

    u, s, vt = torch.linalg.svd(a.flatten(start_dim=1), driver=driver, full_matrices=False)
    u = u[:, :rank] @ torch.diag(s[:rank])
    vt = vt[:rank, :]

    if is_conv:
        u = u.reshape(*u.shape, 1, 1)
        vt = vt.reshape(vt.shape[0], input_dim, kernel_size[0], kernel_size[1])

    return u, vt, torch.tensor(vt.shape[0], dtype=a.dtype, device=a.device)


@dataclasses.dataclass
class LoadInputDictsVisitor(RecipeVisitor):
    __base_dirs: List[pathlib.Path]
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
            for base_dir in self.__base_dirs:
                path_attempt = base_dir / path
                if not path_attempt.suffix:
                    path_attempt = path_attempt.with_suffix(".safetensors")
                if path_attempt.exists():
                    path = path_attempt
                    break

        return InSafetensorsDict(path, self.__buffer_size_per_dict)


@dataclasses.dataclass
class CloseInputDictsVisitor:
    def visit_model(self, node: recipe_nodes.ModelRecipeNode):
        if node.state_dict is not None:
            node.state_dict.close()
        node.state_dict = None

    def visit_parameter(self, _node: recipe_nodes.ParameterRecipeNode):
        return

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode):
        for model in node.models:
            model.accept(self)
