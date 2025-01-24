import contextlib
import dataclasses
import functools
import gc
import itertools
import logging
import os
import pathlib
import sys
import threading
import typing
import torch
from contextlib import nullcontext
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from sd_mecha.extensions.merge_method import MergeMethod, StateDict, T as MergeMethodT
from sd_mecha.extensions.model_config import ModelConfig
from sd_mecha.recipe_nodes import RecipeVisitor, LiteralRecipeNode, RecipeNode
from sd_mecha.streaming import OutSafetensorsDict, TensorMetadata, StateDictKeyError
from sd_mecha import extensions, recipe_nodes, recipe_serializer
from tqdm import tqdm
from typing import Optional, Mapping, MutableMapping, List, Iterable, Tuple, Dict, TypeVar
from sd_mecha.typing_ import is_subclass


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

    def convert(self, recipe: RecipeNode, config: str | ModelConfig):
        from sd_mecha.conversion import convert
        return convert(recipe, config, self.__base_dirs)

    def merge_and_save(
        self,
        recipe: extensions.merge_method.RecipeNodeOrValue,
        output: MutableMapping[str, torch.Tensor] | pathlib.Path | str = "merge.safetensors",
        fallback_model: Optional[extensions.merge_method.RecipeNodeOrValue] = None,
        save_device: Optional[str | torch.device] = "cpu",
        save_dtype: Optional[torch.dtype] = torch.float16,
        threads: Optional[int] = None,
        total_buffer_size: int = 2**28,
        strict_weight_space: bool = True,
        check_finite: bool = True,
    ):
        recipe = extensions.merge_method.value_to_node(recipe)
        if fallback_model is not None:
            recipe = recipe | fallback_model

        if save_device is not None or save_dtype is not None:
            recipe = recipe.to(device=save_device, dtype=save_dtype)

        if threads is not None and (threads < 0 or not isinstance(threads, int)):
            raise RuntimeError("threads should be a non-negative integer or None")

        total_files_open = (
            recipe.accept(recipe_nodes.ModelsCountVisitor()) +
            int(isinstance(output, (str, pathlib.Path)))
        )
        buffer_size_per_file = total_buffer_size // max(1, total_files_open)
        if threads is None:
            threads = min(max(total_files_open, 2), os.cpu_count(), 8)

        with open_input_dicts(recipe, self.__base_dirs, buffer_size_per_file):
            model_config = recipe.model_config
            if strict_weight_space and recipe.merge_space != "weight":
                raise ValueError(f"recipe should be in 'weight' space, not '{recipe.merge_space.identifier}'")

            recipe_keys = model_config.keys
            output = self.__normalize_output_to_dict(
                output,
                recipe_keys,
                recipe_serializer.serialize(recipe),
                buffer_size_per_file // max(1, threads),
            )
            if threads == 0:
                thread_local_data = SimpleNamespace()
                executor = ThisThreadExecutor()
            else:
                thread_local_data = threading.local()
                executor = ThreadPoolExecutor(max_workers=threads)

            progress = self.__tqdm(total=len(recipe_keys), desc="Merging recipe")
            with executor:
                futures = []
                for key in recipe_keys:
                    fn = recipe.accept
                    fn = self.__track_output(fn, output, key, check_finite)
                    fn = self.__track_progress(fn, key, recipe_keys[key].shape, progress)
                    fn = self.__wrap_thread_context(fn, thread_local_data)
                    futures.append(executor.submit(fn, KeyMergeVisitor(key)))

                for future in as_completed(futures):
                    if future.exception() is not None:
                        for future_to_cancel in futures:
                            future_to_cancel.cancel()
                        raise future.exception()
                    future.result()

            progress.close()
            if isinstance(output, OutSafetensorsDict):
                output.close()

    def __normalize_output_to_dict(
        self,
        output: MutableMapping[str, torch.Tensor] | pathlib.Path | str,
        merged_header: Mapping[str, TensorMetadata],
        serialized_recipe: str,
        buffer_size_per_thread: int,
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
                serialized_recipe,
                buffer_size_per_thread,
            )
        return output

    def __track_output(self, f, output, key, check_finite: bool):
        @functools.wraps(f)
        def track_output(*args, **kwargs):
            try:
                res = f(*args, **kwargs)
                if check_finite and isinstance(res, torch.Tensor) and not res.isfinite().all():
                    logging.warning(f"there are non finite values in key '{key}'")
                output[key] = res
            except StateDictKeyError as k:
                logging.debug(f"skipping key {k}")
        return track_output

    def __track_progress(self, f, key, key_shape, progress):
        @functools.wraps(f)
        def track_progress(*args, **kwargs):
            progress.set_postfix({"key": key} | ({"shape": list(key_shape)} if key_shape is not None else {}))
            res = f(*args, **kwargs)
            progress.update()
            return res
        return track_progress

    def __wrap_thread_context(self, f, ctx):
        @functools.wraps(f)
        def thread_context(*args, **kwargs):
            if torch.cuda.is_available():
                if not hasattr(ctx, 'cuda_stream'):
                    setattr(ctx, "cuda_stream", torch.cuda.Stream())
                with torch.cuda.stream(ctx.cuda_stream):
                    return f(*args, **kwargs)
            else:
                return f(*args, **kwargs)

        return thread_context


class ThisThreadExecutor(nullcontext):
    def submit(self, fn, /, *args, **kwargs):
        result = Future()
        result.set_result(fn(*args, **kwargs))
        return result


@contextlib.contextmanager
def open_input_dicts(recipe: recipe_nodes.RecipeNode, base_dirs: Iterable[pathlib.Path] = (), buffer_size_per_dict: int = 2**28):
    recipe.accept(LoadInputDictsVisitor(base_dirs, buffer_size_per_dict))
    yield recipe
    recipe.accept(CloseInputDictsVisitor())
    gc.collect()
    torch.cuda.empty_cache()


@dataclasses.dataclass
class LoadInputDictsVisitor(RecipeVisitor):
    base_dirs: Iterable[pathlib.Path]
    buffer_size_per_dict: int
    dicts_cache: Dict[str, Mapping[str, torch.Tensor]] = dataclasses.field(default_factory=dict)

    def visit_literal(self, node: LiteralRecipeNode):
        if isinstance(node.value, Mapping) and node.model_config is None:
            node.model_config = infer_model_configs(node.value, None)[0]
        return node

    def visit_model(self, node: recipe_nodes.ModelRecipeNode):
        node.state_dict, node_path = self.__load_dict(node)
        if node.model_config is None:
            node.model_config = infer_model_configs(node.state_dict, node_path)[0]

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode):
        for model in itertools.chain(node.args, node.kwargs.values()):
            model.accept(self)

    def __load_dict(self, node: recipe_nodes.ModelRecipeNode):
        if node.state_dict is not None:
            return node.state_dict, node.path

        path = node.path
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        if not path.is_absolute():
            for base_dir in self.base_dirs:
                path_attempt = base_dir / path
                if not path_attempt.suffix:
                    path_attempt = path_attempt.with_suffix(".safetensors")
                if path_attempt.exists():
                    path = path_attempt
                    break

        cache_key = str(path.resolve())
        if cache_key not in self.dicts_cache:
            matching_formats = []
            for model_format in extensions.model_format.get_all():
                if model_format.matches(path):
                    matching_formats.append(model_format)

            if len(matching_formats) > 1:
                raise RuntimeError(f"ambiguous format ({', '.join(f.identifier for f in matching_formats)}) for model {path}")
            if len(matching_formats) < 1:
                raise RuntimeError(f"no matching format found for model {path}")

            self.dicts_cache[cache_key] = matching_formats[0].get_read_dict(path, self.buffer_size_per_dict)

        return self.dicts_cache[cache_key], path


def infer_model_configs(state_dict: Iterable[str], path: Optional[pathlib.Path]) -> List[ModelConfig]:
    state_dict_set = set(state_dict)
    configs_affinity = {}
    for model_config in extensions.model_config.get_all():
        matched_keys = state_dict_set.intersection(model_config.keys)
        # heuristic: accept config only if we match more than 90% of the keys of the state dict
        if len(matched_keys) >= len(state_dict_set) * 0.9:
            configs_affinity[model_config] = len(matched_keys)
        # heuristic: break early if we match more than 90% of the keys of a config
        if len(matched_keys) == len(state_dict_set) and len(matched_keys) >= len(model_config.keys) * 0.9:
            break

    best_configs = sorted(configs_affinity, key=configs_affinity.get, reverse=True)
    if not best_configs or configs_affinity[best_configs[0]] == 0:
        raise ValueError(f"No configuration was found for the given state dict{' ' + str(path) if path is not None else ''}")

    return best_configs


@dataclasses.dataclass
class CloseInputDictsVisitor(RecipeVisitor):
    def visit_literal(self, node: LiteralRecipeNode):
        pass

    def visit_model(self, node: recipe_nodes.ModelRecipeNode):
        if node.state_dict is not None:
            node.state_dict.close()
        node.state_dict = None

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode):
        for model in itertools.chain(node.args, node.kwargs.values()):
            model.accept(self)


@dataclasses.dataclass
class KeyMergeVisitor(RecipeVisitor):
    key: str

    def visit_literal(self, node: LiteralRecipeNode):
        value = node.value
        if isinstance(node.value, Mapping):
            try:
                value = value[self.key]
            except KeyError as e:
                raise StateDictKeyError(str(e)) from e
        if isinstance(value, torch.Tensor | str | int | float | bool):
            return value
        raise RuntimeError(f"Unexpected literal node value of type {type(value)}")

    def visit_model(self, node: recipe_nodes.ModelRecipeNode) -> torch.Tensor:
        try:
            return node.state_dict[self.key]
        except KeyError as e:
            raise StateDictKeyError(str(e)) from e

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode) -> torch.Tensor:
        merged_args, merged_kwargs = self.__visit_deeper_first(node.args, node.kwargs, node.merge_method)
        return node.merge_method.merge_key(
            merged_args,
            merged_kwargs,
            self.key,
            node.cache,
        )

    def __visit_deeper_first(
        self,
        node_args: Tuple[recipe_nodes.RecipeNode, ...],
        node_kwargs: Dict[str, recipe_nodes.RecipeNode],
        merge_method: MergeMethod,
    ):
        def depth_of_value(index) -> int:
            nodes = node_args if isinstance(index, int) else node_kwargs
            return nodes[index].accept(recipe_nodes.ModelDepthRecipeVisitor())

        error_holder = ErrorHolder()
        merged = {}
        input_types = merge_method.get_input_types().as_dict(len(node_args))
        indices = itertools.chain(range(len(node_args)), node_kwargs.keys())

        for index in sorted(indices, key=depth_of_value, reverse=True):
            nodes = node_args if isinstance(index, int) else node_kwargs
            if is_subclass(input_types[index], StateDict):
                expected_type = next(iter(typing.get_args(input_types[index]) or (MergeMethodT,)))
                merged[index] = error_holder.intercept(MergeNodeWrapperStateDict, nodes[index], expected_type, self)
            else:
                merged[index] = cast_node_value(error_holder.intercept(nodes[index].accept, self), input_types[index])

        merged_args = [merged.get(index) for index in range(len(node_args))]
        merged_kwargs = {k: v for k, v in merged.items() if not isinstance(k, int)}
        error_holder.try_raise()
        return merged_args, merged_kwargs


class ErrorHolder:
    def __init__(self):
        self.exc_info = None

    def intercept(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            self.exc_info = sys.exc_info()
            return None

    def try_raise(self):
        if self.exc_info:
            raise self.get_error()

    def get_error(self):
        if self.exc_info:
            return self.exc_info[1].with_traceback(self.exc_info[2])
        else:
            return None


class MergeNodeWrapperStateDict(StateDict):
    def __init__(
        self,
        merge_node: recipe_nodes.MergeRecipeNode,
        expected_type: type,
        original_merge_visitor: KeyMergeVisitor,
    ):
        self.merge_node = merge_node
        self.expected_type = expected_type
        self.original_merge_visitor = original_merge_visitor

    def __getitem__(self, key):
        key_merger = dataclasses.replace(self.original_merge_visitor, key=key)
        return cast_node_value(self.merge_node.accept(key_merger), self.expected_type)

    def __len__(self):
        return len(self.compute_keys())

    def __iter__(self):
        return iter(self.keys())

    def keys(self) -> Iterable[str]:
        return self.compute_keys().keys()

    @property
    def model_config(self) -> ModelConfig:
        return self.merge_node.model_config

    def compute_keys(self):
        return self.model_config.keys


def cast_node_value(value, expected_type):
    if value is None:
        return value

    try:
        if issubclass(typing.get_origin(expected_type) or expected_type, StateDict):
            expected_type = (typing.get_args(expected_type) + (MergeMethodT,))[0]
    except TypeError:
        pass

    if isinstance(expected_type, TypeVar) or isinstance(value, expected_type):
        return value
    if isinstance(expected_type, str):
        raise RuntimeError(f"cannot implicitly convert {type(value)} to {expected_type}")
    if issubclass(expected_type, int):
        return int(value)
    if issubclass(expected_type, float):
        return float(value)
    if issubclass(expected_type, torch.Tensor):
        return torch.tensor(value, dtype=torch.float32)
    return value
