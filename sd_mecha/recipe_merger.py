import dataclasses
import functools
import gc
import itertools
import logging
import os
import pathlib
import sys
import threading
import torch
from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from sd_mecha.extensions.merge_method import MergeMethod, StateDict
from sd_mecha.extensions.model_config import ModelConfig
from sd_mecha.recipe_nodes import RecipeVisitor
from sd_mecha.streaming import OutSafetensorsDict, TensorMetadata, StateDictKeyError
from sd_mecha import extensions, recipe_nodes, recipe_serializer
from tqdm import tqdm
from typing import Optional, Mapping, MutableMapping, List, Iterable, Tuple, Dict


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
        self,
        recipe: extensions.merge_method.RecipeNodeOrLiteral,
        output: MutableMapping[str, torch.Tensor] | pathlib.Path | str = "merge.safetensors",
        fallback_model: Optional[extensions.merge_method.RecipeNodeOrLiteral] = None,
        save_device: Optional[str] = "cpu",
        save_dtype: Optional[torch.dtype] = torch.float16,
        threads: Optional[int] = None,
        total_buffer_size: int = 2**28,
        strict_weight_space: bool = True,
    ):
        recipe = extensions.merge_method.value_to_node(recipe)
        if fallback_model is not None:
            recipe = extensions.merge_method.resolve("fallback").create_recipe(recipe, fallback_model)

        if threads is not None and (threads < 0 or not isinstance(threads, int)):
            raise RuntimeError("threads should be a non-negative integer or None")

        total_files_open = (
            recipe.accept(recipe_nodes.ModelsCountVisitor()) +
            int(isinstance(output, (str, pathlib.Path)))
        )
        buffer_size_per_file = total_buffer_size // total_files_open
        if threads is None:
            threads = min(max(total_files_open, 2), os.cpu_count(), 8)

        load_input_dicts_visitor = LoadInputDictsVisitor(
            self.__base_dirs,
            buffer_size_per_file,
        )
        recipe.accept(load_input_dicts_visitor)

        model_config = recipe.model_config
        if strict_weight_space and recipe.merge_space != "weight":
            raise ValueError(f"recipe should be in 'weight' space, not '{recipe.merge_space}'")

        # ideally we use keys in the order the model config lists them
        # model configs should be responsible for default key ordering
        #   (in some cases we can optimize the key ordering from input models, disregard default order in this case)
        # however I cannot be bothered to update the current config files
        # hence we force lexicographic ordering here for the time being :>
        #   (it's the exact right ordering most of the time)
        recipe_keys = OrderedDict(sorted(model_config.compute_keys().items(), key=lambda t: t[0]))

        keys_to_merge = model_config.compute_keys_to_merge()
        output = self.__normalize_output_to_dict(
            output,
            recipe_keys,
            keys_to_merge,
            recipe_serializer.serialize(recipe),
            buffer_size_per_file // max(1, threads),
            save_dtype,
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
                key_merger = self.__get_key_merger(key, recipe)
                key_merger = self.__track_output(key_merger, output, key, save_dtype, save_device)
                key_merger = self.__track_progress(key_merger, key, recipe_keys[key].shape, progress)
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
        merged_header: Mapping[str, TensorMetadata],
        keys_to_merge: Iterable[str],
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

            merged_header = {
                k: dataclasses.replace(v, dtype=dtype) if k in keys_to_merge else v
                for k, v in merged_header.items()
            }

            output = OutSafetensorsDict(
                output,
                merged_header,
                serialized_recipe,
                buffer_size_per_thread,
            )
        return output

    def __get_key_merger(self, key, recipe):
        key_merger = KeyMerger(recipe)
        return functools.partial(key_merger.merge_and_save, key)

    def __track_output(self, f, output, key, save_dtype, save_device):
        if save_dtype is None:
            save_dtype = self.__default_dtype

        to_kwargs = {"dtype": save_dtype}
        if save_device is not None:
            to_kwargs["device"] = save_device

        @functools.wraps(f)
        def track_output(*args, **kwargs):
            try:
                output[key] = f(*args, **kwargs).to(**to_kwargs)
            except StateDictKeyError as k:
                logging.debug(f"skipping key {k}")
        return track_output

    def __track_progress(self, f, key, key_shape, progress):
        @functools.wraps(f)
        def track_progress(*args, **kwargs):
            progress.set_postfix({"key": key, "shape": list(key_shape)})
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


@dataclasses.dataclass
class LoadInputDictsVisitor(RecipeVisitor):
    base_dirs: List[pathlib.Path]
    buffer_size_per_dict: int
    dicts_cache: Dict[str, Mapping[str, torch.Tensor]] = dataclasses.field(default_factory=dict)

    def visit_model(self, node: recipe_nodes.ModelRecipeNode):
        node.state_dict, node_path = self.__load_dict(node)
        if node.model_config is None:
            node.model_config = self.__detect_model_config(node.state_dict, node_path)

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode):
        for model in node.inputs:
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

    def __detect_model_config(self, state_dict: Iterable[str], path: pathlib.Path):
        configs_affinity = {}
        for model_config in extensions.model_config.get_all():
            state_dict_set = set(state_dict)
            matched_keys = state_dict_set.intersection(model_config.compute_keys())
            configs_affinity[model_config.identifier] = len(matched_keys)
            if len(matched_keys) == len(state_dict_set):
                break

        best_config = max(configs_affinity, key=configs_affinity.get)
        best_config = extensions.model_config.resolve(best_config)
        if configs_affinity[best_config.identifier] == 0:
            raise ValueError(f"No configuration matches any key of {path}")

        return best_config


@dataclasses.dataclass
class CloseInputDictsVisitor(RecipeVisitor):
    def visit_model(self, node: recipe_nodes.ModelRecipeNode):
        if node.state_dict is not None:
            node.state_dict.close()
        node.state_dict = None

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode):
        for model in node.inputs:
            model.accept(self)


@dataclasses.dataclass
class KeyMerger:
    recipe: recipe_nodes.RecipeNode

    def merge_and_save(
        self,
        key: str,
    ) -> torch.Tensor:
        key_merger = KeyMergeVisitor(
            key,
        )
        return self.recipe.accept(key_merger)


@dataclasses.dataclass
class KeyMergeVisitor(RecipeVisitor):
    key: str

    def visit_model(self, node: recipe_nodes.ModelRecipeNode) -> torch.Tensor:
        try:
            return node.state_dict[self.key]
        except KeyError as e:
            raise StateDictKeyError(str(e)) from e

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode) -> torch.Tensor:
        merged_args, merged_kwargs, error = self.__visit_deeper_first(node.input_args, node.input_kwargs, node.merge_method)
        try:
            if error: raise error

            return node.merge_method.merge_key(
                merged_args,
                merged_kwargs,
                self.key,
            )
        except StateDictKeyError:
            for node_child, merged in zip(
                itertools.chain(node.input_args, node.input_kwargs.keys()),
                itertools.chain(merged_args, merged_kwargs.keys()),
            ):
                if node_child.model_config == node.model_config and node_child.merge_space == node.merge_space:
                    if isinstance(merged, str):
                        merged = merged_kwargs[merged]

                    if isinstance(merged, torch.Tensor):
                        return merged
                    elif isinstance(merged, StateDict):
                        return merged[self.key]
            raise

    def __visit_deeper_first(
        self,
        node_args: Tuple[recipe_nodes.RecipeNode, ...],
        node_kwargs: Dict[str, recipe_nodes.RecipeNode],
        merge_method: MergeMethod,
    ):
        def depth_of_value(index) -> int:
            nodes = node_args if isinstance(index, int) else node_kwargs
            return nodes[index].accept(recipe_nodes.DepthRecipeVisitor())

        error_holder = ErrorHolder()
        merged = {}
        input_types = merge_method.get_input_types(len(node_args))
        indices = itertools.chain(range(len(node_args)), node_kwargs.keys())
        for index in sorted(indices, key=depth_of_value, reverse=True):
            nodes = node_args if isinstance(index, int) else node_kwargs
            if issubclass(input_types[index], torch.Tensor):
                merged[index] = error_holder.intercept(nodes[index].accept, self)
            else:
                merged[index] = error_holder.intercept(MergeNodeMappingWrapper, nodes[index], self)

        merged_args = [merged.get(index) for index in range(len(node_args))]
        merged_kwargs = {k: v for k, v in merged if not isinstance(k, int)}
        return merged_args, merged_kwargs, error_holder.get_error()


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


class MergeNodeMappingWrapper(StateDict):
    def __init__(self, merge_node: recipe_nodes.MergeRecipeNode, original_merge_visitor: KeyMergeVisitor):
        self.merge_node = merge_node
        self.original_merge_visitor = original_merge_visitor
        self.to_args = (), {}

        self.__keys_to_merge = None
        self.__keys_to_copy = None
        self.__keys = None

    def to(self, *args, **kwargs):
        self.to_args = args, kwargs
        return self

    def __getitem__(self, key):
        key_merger = dataclasses.replace(self.original_merge_visitor, key=key)
        return self.merge_node.accept(key_merger).to(*self.to_args[0], **self.to_args[1])

    def __len__(self):
        return len(self.compute_keys())

    def __iter__(self):
        return iter(self.keys())

    def keys(self) -> Iterable[str]:
        return self.compute_keys().keys()

    @property
    def model_config(self) -> ModelConfig:
        return self.merge_node.model_config

    def compute_keys_to_merge(self):
        if self.__keys_to_merge is None:
            self.__keys_to_merge = self.model_config.compute_keys_to_merge()
        return self.__keys_to_merge

    def compute_keys_to_copy(self):
        if self.__keys_to_copy is None:
            self.__keys_to_copy = self.model_config.compute_keys_to_copy()
        return self.__keys_to_copy

    def compute_keys(self):
        if self.__keys is None:
            self.__keys = self.model_config.compute_keys()
        return self.__keys
