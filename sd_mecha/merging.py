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
from .extensions import model_configs, model_formats
from .extensions.merge_methods import MergeMethod, StateDict, T as MergeMethodT, value_to_node
from .extensions.model_configs import ModelConfig, StructuralModelConfig
from .recipe_nodes import RecipeVisitor, LiteralRecipeNode, RecipeNode, MergeRecipeNode, ModelRecipeNode, RecipeNodeOrValue, NonDictLiteralValue
from .streaming import OutSafetensorsDict, TensorMetadata, StateDictKeyError
from . import recipe_nodes, serialization
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from contextlib import nullcontext
from tqdm import tqdm as tqdm_original
from types import SimpleNamespace
from typing import Optional, Mapping, MutableMapping, List, Iterable, Tuple, Dict, TypeVar
from .typing_ import is_subclass


def merge(
    recipe: RecipeNodeOrValue,
    fallback_model: Optional[RecipeNodeOrValue] = ...,
    merge_device: Optional[str | torch.device] = ...,
    merge_dtype: Optional[torch.dtype] = ...,
    output_device: Optional[str | torch.device] = ...,
    output_dtype: Optional[torch.dtype] = ...,
    threads: Optional[int] = ...,
    total_buffer_size: int = ...,
    model_dirs: Iterable[pathlib.Path] = ...,
    strict_weight_space: bool = ...,
    check_finite: bool = ...,
    tqdm: type = ...,
    *,
    output: Optional[MutableMapping[str, torch.Tensor]] | pathlib.Path | str = ...,
) -> Optional[MutableMapping[str, torch.Tensor]]:
    """
    Merge a recipe graph into a final state dict and optionally save it to a file.

    This function streams each key from the underlying safetensors or dictionaries,
    applies all instructions in the `recipe`, and writes the resulting data to `output`.

    Args:
        recipe:
            A `RecipeNode`, python literal, or dictionary describing how to merge or transform multiple models.
        fallback_model (optional):
            A secondary recipe or model to provide values for any keys missing from `recipe`.
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
        output (optional):
            Where to store the merged state dict. Can be a filesystem path (string or
            `Path`) ending with `.safetensors`, an in-memory dict-like object, or None.
            If it is None or omitted, an empty dict is created and returned when the merge completes.

    Returns:
        None, or the in-memory dictionary if `output` is either a MutableMapping or None.
    """
    if output is ...:
        output = None
    if fallback_model is ...:
        fallback_model = None
    if merge_device is ...:
        merge_device = "cpu"
    if merge_dtype is ...:
        merge_dtype = torch.float64
    if output_device is ...:
        output_device = "cpu"
    if output_dtype is ...:
        output_dtype = torch.float16
    if threads is ...:
        threads = None
    if total_buffer_size is ...:
        total_buffer_size = 2**28
    if model_dirs is ...:
        model_dirs = ()
    else:
        model_dirs = list(model_dirs)
    if strict_weight_space is ...:
        strict_weight_space = True
    if check_finite is ...:
        check_finite = True
    if tqdm is ...:
        tqdm = tqdm_original

    recipe = value_to_node(recipe)
    if fallback_model is not None:
        recipe = recipe | fallback_model

    if merge_device is not None or merge_dtype is not None:
        recipe = recipe.accept(CastInputDicts(merge_device, merge_dtype))

    if output_device is not None or output_dtype is not None:
        recipe = recipe.to(device=output_device, dtype=output_dtype)

    if threads is not None and (threads < 0 or not isinstance(threads, int)):
        raise RuntimeError("threads should be a non-negative integer or None")

    total_files_open = (
        recipe.accept(recipe_nodes.ModelsCountVisitor()) +
        int(isinstance(output, (str, pathlib.Path)))
    )
    buffer_size_per_file = total_buffer_size // max(1, total_files_open)
    if threads is None:
        threads = min(max(total_files_open, 2), os.cpu_count(), 8)

    if threads == 0:
        thread_local_data = SimpleNamespace()
        executor = ThisThreadExecutor()
    else:
        thread_local_data = threading.local()
        executor = ThreadPoolExecutor(max_workers=threads)

    with open_input_dicts(recipe, model_dirs, buffer_size_per_file):
        model_config = recipe.model_config
        if strict_weight_space and recipe.merge_space != "weight":
            raise ValueError(f"recipe should be in 'weight' space, not '{recipe.merge_space.identifier}'")

        buffer_size_per_file_per_thread = buffer_size_per_file // max(1, threads)
        recipe_keys = model_config.keys
        with (
            executor,
            tqdm(total=len(recipe_keys), desc="Merging recipe") as progress,
            _get_output_dict(
                output,
                recipe_keys,
                recipe,
                model_dirs,
                buffer_size_per_file_per_thread,
            ) as output_dict,
        ):
            fix_torch_threading(merge_device)
            futures = []
            for key in recipe_keys:
                fn = recipe.accept
                fn = _track_output(fn, output_dict, key, check_finite)
                fn = _track_progress(fn, key, recipe_keys[key].shape, progress)
                fn = _wrap_thread_context(fn, thread_local_data)
                futures.append(executor.submit(fn, KeyMergeVisitor(key)))

            for future in as_completed(futures):
                if future.exception() is not None:
                    for future_to_cancel in futures:
                        future_to_cancel.cancel()
                    raise future.exception()
                future.result()

            if isinstance(output_dict, MutableMapping):
                return output_dict


def fix_torch_threading(device):
    # this greedy loads the torch.linalg module
    # avoids a hard error caused by threads>1 with some torch ops
    # see https://github.com/pytorch/pytorch/issues/90613
    torch.linalg.inv(torch.ones((1, 1), device=device))


@contextlib.contextmanager
def _get_output_dict(
    output: Optional[MutableMapping[str, torch.Tensor]] | pathlib.Path | str,
    merged_header: Mapping[str, TensorMetadata],
    recipe: RecipeNode,
    model_dirs: Iterable[pathlib.Path],
    buffer_size_per_thread: int,
):
    if isinstance(output, (str, pathlib.Path)):
        if not isinstance(output, pathlib.Path):
            output = pathlib.Path(output)
        if not output.is_absolute():
            for model_dir in model_dirs:
                output = model_dir / output
                break
        logging.info(f"Saving to {output}")

        try:
            serialized_recipe = serialization.serialize(recipe)
        except TypeError:
            logging.warning("The recipe graph could not be serialized. The output state dict will not contain the recipe.")
            serialized_recipe = None
        streamed_output = OutSafetensorsDict(
            output,
            merged_header,
            serialized_recipe,
            buffer_size_per_thread,
        )
        try:
            yield streamed_output
        finally:
            streamed_output.close()
    else:
        if output is None:
            output = {}
        yield output


def _track_output(fn, output, key, check_finite: bool):
    @functools.wraps(fn)
    def track_output(*args, **kwargs):
        try:
            res = fn(*args, **kwargs)
            if check_finite and isinstance(res, torch.Tensor) and not res.isfinite().all():
                logging.warning(f"there are non finite values in key '{key}'")
            output[key] = res
        except StateDictKeyError as k:
            logging.debug(f"skipping key {k}")
    return track_output


def _track_progress(fn, key, key_shape, progress):
    @functools.wraps(fn)
    def track_progress(*args, **kwargs):
        progress.set_postfix({"key": key} | ({"shape": list(key_shape)} if key_shape is not None else {}))
        res = fn(*args, **kwargs)
        progress.update()
        return res
    return track_progress


def _wrap_thread_context(fn, ctx):
    @functools.wraps(fn)
    def thread_context(*args, **kwargs):
        if torch.cuda.is_available():
            if not hasattr(ctx, 'cuda_stream'):
                setattr(ctx, "cuda_stream", torch.cuda.Stream())
            with torch.cuda.stream(ctx.cuda_stream):
                return fn(*args, **kwargs)
        else:
            return fn(*args, **kwargs)

    return thread_context


class ThisThreadExecutor(nullcontext):
    def submit(self, fn, /, *args, **kwargs):
        result = Future()
        result.set_result(fn(*args, **kwargs))
        return result


@contextlib.contextmanager
def open_input_dicts(
    recipe: recipe_nodes.RecipeNode,
    model_dirs: Iterable[pathlib.Path] = (),
    buffer_size_per_dict: int = 0,
    empty_cuda_cache: bool = False,
):
    try:
        recipe.accept(LoadInputDictsVisitor(model_dirs, buffer_size_per_dict))
        yield recipe
    finally:
        recipe.accept(CloseInputDictsVisitor())
        if empty_cuda_cache:
            gc.collect()
            torch.cuda.empty_cache()


@dataclasses.dataclass
class LoadInputDictsVisitor(RecipeVisitor):
    model_dirs: Iterable[pathlib.Path]
    buffer_size_per_dict: int
    dicts_cache: MutableMapping[str, Mapping[str, torch.Tensor]] = dataclasses.field(default_factory=dict)
    structural_metadata: MutableMapping[str, TensorMetadata] = dataclasses.field(default_factory=dict)
    param_config: Optional[ModelConfig] = None

    def visit_literal(self, node: LiteralRecipeNode):
        if not isinstance(node.value, Mapping):
            return

        metadata = {}
        for k, v in node.value.items():
            if isinstance(v, RecipeNode):
                v.accept(self)
                metadata[k] = v.model_config.keys.get(k, TensorMetadata(None, None))
            elif isinstance(v, torch.Tensor):
                metadata[k] = TensorMetadata(v.shape, v.dtype)
            elif k not in metadata:
                metadata[k] = TensorMetadata(None, None)

        node.model_config = self.__determine_model_config(metadata, node.model_config)

    def visit_model(self, node: recipe_nodes.ModelRecipeNode):
        node.state_dict, node_path = self.__load_dict(node)
        sd_metadata = dict(node.state_dict.metadata())
        node.model_config = self.__determine_model_config(sd_metadata, node.model_config)

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode):
        input_configs = node.merge_method.get_input_configs().as_dict(len(node.args))
        for k, v in itertools.chain(enumerate(node.args), node.kwargs.items()):
            v.accept(dataclasses.replace(self, param_config=input_configs[k]))

    def __load_dict(self, node: recipe_nodes.ModelRecipeNode):
        if node.state_dict is not None:
            return node.state_dict, node.path

        path = node.path
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        if not path.is_absolute():
            for base_dir in self.model_dirs:
                path_attempt = base_dir / path
                if path_attempt.exists():
                    path = path_attempt
                    break

        cache_key = str(path.resolve())
        if cache_key not in self.dicts_cache:
            matching_formats = []
            for model_format in model_formats.get_all():
                if model_format.matches(path):
                    matching_formats.append(model_format)

            if len(matching_formats) > 1:
                raise RuntimeError(f"ambiguous format ({', '.join(f.identifier for f in matching_formats)}) for model {path}")
            if len(matching_formats) < 1:
                raise RuntimeError(f"no matching format found for model {path}")

            self.dicts_cache[cache_key] = matching_formats[0].get_read_dict(path, self.buffer_size_per_dict)

        return self.dicts_cache[cache_key], path

    def __determine_model_config(self, metadata: MutableMapping[str, TensorMetadata], config_hint: Optional[ModelConfig]) -> ModelConfig:
        config = config_hint
        if config is None or config.identifier == model_configs.INFER.identifier:
            inferred_model_configs = infer_model_configs(metadata)
            if self.param_config in inferred_model_configs:
                config = self.param_config
            elif len(inferred_model_configs) == 1:
                config = next(iter(inferred_model_configs))
            else:
                config = None

        if config is None:
            if not self.structural_metadata:
                self.structural_metadata = metadata
            else:
                for k in list(self.structural_metadata):
                    if k not in metadata:
                        del self.structural_metadata[k]
                    elif (
                        self.structural_metadata[k].shape is None or
                        metadata[k].shape is not None and metadata[k].shape.numel() > self.structural_metadata[k].shape.numel()
                    ):
                        self.structural_metadata[k] = metadata[k]
            config = StructuralModelConfig(self.structural_metadata)

        return config


def is_config_stub(config: Optional[model_configs.ModelConfig]) -> bool:
    return getattr(config, "identifier", None) in (None, model_configs.INFER.identifier)


def infer_model_configs(state_dict: Iterable[str]) -> List[ModelConfig]:
    state_dict_set = set(state_dict)
    configs_affinity = {}
    for model_config in model_configs.get_all():
        matched_keys = state_dict_set.intersection(model_config.keys)
        # heuristic: accept config only if we match more than 90% of the keys of the state dict
        if len(matched_keys) >= len(state_dict_set) * 0.9:
            configs_affinity[model_config] = len(matched_keys)
        # heuristic: break early if we match more than 90% of the keys of a config
        if len(matched_keys) == len(state_dict_set) and len(matched_keys) >= len(model_config.keys) * 0.9:
            break

    best_configs = sorted(configs_affinity, key=configs_affinity.get, reverse=True)
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
        if isinstance(value, NonDictLiteralValue):
            return value
        if isinstance(value, RecipeNode):
            return value.accept(self)
        raise TypeError(f"Unexpected literal node value of type {type(value)}")

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


@dataclasses.dataclass
class CastInputDicts(RecipeVisitor):
    device: str | torch.device
    dtype: torch.dtype

    def visit_literal(self, node: LiteralRecipeNode):
        if (
            isinstance(node.value, torch.Tensor) or
            isinstance(node.value, Mapping) and isinstance(next(iter(node.value.values())), torch.Tensor)
        ):
            return node.to(device=self.device, dtype=self.dtype)
        return node

    def visit_model(self, node: ModelRecipeNode):
        return node.to(device=self.device, dtype=self.dtype)

    def visit_merge(self, node: MergeRecipeNode):
        return MergeRecipeNode(
            node.merge_method,
            tuple(v.accept(self) for v in node.args),
            {k: v.accept(self) for k, v in node.kwargs.items()},
            node.cache,
        )
