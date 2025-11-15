import contextlib
import dataclasses
import functools
import gc
import itertools
import logging
import operator
import os
import pathlib
import sys
import threading
import typing
import torch
from .extensions import model_configs, model_formats
from .extensions.merge_methods import MergeMethod, StateDict, T as MergeMethodT, value_to_node
from .extensions.merge_spaces import MergeSpace
from .extensions.model_configs import ModelConfig, StructuralModelConfig, KeyMetadata
from .recipe_nodes import RecipeVisitor, LiteralRecipeNode, RecipeNode, MergeRecipeNode, ModelRecipeNode, RecipeNodeOrValue, NonDictLiteralValue
from .streaming import OutSafetensorsDict, TensorMetadata, StateDictKeyError
from . import recipe_nodes, serialization
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from contextlib import nullcontext
from tqdm import tqdm as tqdm_original
from types import SimpleNamespace
from typing import Optional, Mapping, MutableMapping, List, Set, Iterable, Tuple, Dict, TypeVar
from .typing_ import is_subclass


def merge(
    recipe: RecipeNodeOrValue,
    *,
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
    omit_extra_keys: bool = ...,
    omit_ema: bool = ...,
    check_mandatory_keys: bool = ...,
    tqdm: type = ...,
    output: Optional[MutableMapping[str, torch.Tensor]] | pathlib.Path | str = ...,
) -> Optional[MutableMapping[str, torch.Tensor]]:
    """
    Materialize a state dict from a recipe graph and optionally save it to a file.

    For each key of the target model config, execute all instructions of the recipe graph
    and store the result into a dictionary using the specified output strategy.

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
            If True, warns if any non-finite values appear in the output model.
        omit_extra_keys (optional):
            If True, warns about and removes unrecognized keys from the output model.
        omit_ema (optional):
            If True, omits ema keys from the output model. Defaults to omit_extra_keys.
        check_mandatory_keys (optional):
            If True and an input model is missing non-optional keys, raises RuntimeError.
        tqdm (optional):
            A custom progress-bar factory. By default, uses `tqdm.tqdm`.
        output (optional):
            Where to store the merged state dict. Can be a filesystem path (string or
            `Path`) ending with `.safetensors`, an in-memory dict-like object, or None.
            If it is None or omitted, an empty dict is created and returned when the merge completes.

    Returns:
        The in-memory dictionary if `output` is either a MutableMapping or None, and nothing if `output` is a file path.
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
    if omit_extra_keys is ...:
        omit_extra_keys = True
    if omit_ema is ...:
        omit_ema = omit_extra_keys
    if check_mandatory_keys is ...:
        check_mandatory_keys = False
    if tqdm is ...:
        tqdm = tqdm_original

    recipe = value_to_node(recipe)
    original_recipe = recipe
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

    with open_input_dicts(recipe, model_dirs, buffer_size_per_file, omit_extra_keys, check_mandatory_keys):
        if strict_weight_space and recipe.merge_space != "weight":
            raise ValueError(f"recipe should be in 'weight' space, not '{recipe.merge_space.identifier}'")

        recipe_metadata = recipe.model_config.metadata()
        if not omit_extra_keys:
            recipe, recipe_metadata = copy_extra_keys(recipe)

        if omit_ema and "ema" in recipe.model_config.components():
            for key in recipe.model_config.components()["ema"].keys():
                if key in recipe_metadata:
                    del recipe_metadata[key]

        buffer_size_per_file_per_thread = buffer_size_per_file // max(1, threads)

        with (
            executor,
            tqdm(total=len(recipe_metadata), desc="Merging recipe") as progress,
            _get_output_dict(
                output,
                recipe_metadata,
                original_recipe,
                model_dirs,
                buffer_size_per_file_per_thread,
            ) as output_dict,
        ):
            fix_torch_threading()
            futures = []
            for key in recipe_metadata:
                fn = recipe.accept
                fn = _track_output(fn, output_dict, key, check_finite)
                fn = _track_progress(fn, key, recipe_metadata[key].shape, progress)
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


def fix_torch_threading():
    if not torch.cuda.is_available():
        return

    # this greedy loads the torch.linalg module
    # avoids a hard error caused by threads>1 with some torch ops
    # see https://github.com/pytorch/pytorch/issues/90613
    torch.linalg.inv(torch.ones((1, 1), device="cuda"))


def copy_extra_keys(recipe: RecipeNode):
    config = recipe.model_config
    merge_space = recipe.merge_space
    metadata = config.metadata()
    aliases = {k_alias: k for k, k_aliases in config.aliases().items() for k_alias in k_aliases}

    forwardable_nodes_visitor = ForwardableNodesVisitor(config, merge_space)
    recipe.accept(forwardable_nodes_visitor)
    forwardable_nodes = forwardable_nodes_visitor.forwardable_nodes

    new_recipe = functools.reduce(operator.or_, [n[0] for n in forwardable_nodes], recipe)
    new_metadata = OrderedDict((aliases.get(k, k), v) for n in forwardable_nodes for k, v in n[1].items()) | metadata
    return new_recipe, new_metadata


@dataclasses.dataclass
class ForwardableNodesVisitor(RecipeVisitor):
    target_config: ModelConfig
    target_merge_space: MergeSpace
    forwardable_nodes: List[Tuple[RecipeNode, Mapping[str, TensorMetadata]]] = dataclasses.field(default_factory=list)

    def visit_literal(self, node: LiteralRecipeNode):
        if not isinstance(node.value, Mapping) or not node.value:
            return

        if isinstance(next(iter(node.value.values())), RecipeNode):
            for v in node.value.values():
                v.accept(self)
        elif (
            isinstance(next(iter(node.value.values())), torch.Tensor) and
            node.model_config == self.target_config and
            node.merge_space == self.target_merge_space and
            not any(node in n[0] for n in self.forwardable_nodes)
        ):
            metadata = OrderedDict(
                (k, TensorMetadata(v.shape, v.dtype))
                for k, v in node.value.items()
            )
            self.forwardable_nodes.append((node, metadata))

    def visit_model(self, node: ModelRecipeNode):
        if node.model_config == self.target_config and not any(node in n[0] for n in self.forwardable_nodes):
            self.forwardable_nodes.append((node, OrderedDict(node.state_dict.metadata())))

    def visit_merge(self, node: MergeRecipeNode):
        for arg in (*node.args, *node.kwargs.values()):
            arg.accept(self)


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
            if check_finite and isinstance(res, torch.Tensor):
                try:
                    all_finite = res.isfinite().all()
                except RuntimeError:  # for example, fp8_e4m3fn doesn't support .isfinite()
                    all_finite = res.to(dtype=torch.bfloat16).isfinite().all()

                if not all_finite:
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
    omit_extra_keys: bool = True,
    check_mandatory_keys: bool = False,
    empty_cuda_cache: bool = False,
):
    try:
        recipe.accept(LoadInputDictsVisitor(list(model_dirs), buffer_size_per_dict, omit_extra_keys, check_mandatory_keys))
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
    strip_extra_keys: bool
    check_mandatory_keys: bool
    dicts_cache: MutableMapping[str, Mapping[str, torch.Tensor]] = dataclasses.field(default_factory=dict)
    structural_metadata: MutableMapping[str, KeyMetadata] = dataclasses.field(default_factory=OrderedDict)
    param_config: Optional[ModelConfig] = None

    def visit_literal(self, node: LiteralRecipeNode):
        if not isinstance(node.value, Mapping):
            return

        metadata = {}
        for k, v in node.value.items():
            if isinstance(v, RecipeNode):
                v.accept(self)
            elif isinstance(v, torch.Tensor):
                metadata[k] = KeyMetadata(v.shape, v.dtype, optional=True)
            elif k not in metadata:
                metadata[k] = KeyMetadata(None, None, optional=True)

        node.model_config = self.__determine_model_config(metadata, node.model_config)
        check_model_config(node.value, node.model_config, self.strip_extra_keys, self.check_mandatory_keys, "<in-memory state dict>")

    def visit_model(self, node: recipe_nodes.ModelRecipeNode):
        node.state_dict, node_path = self.__load_dict(node)
        metadata = OrderedDict(node.state_dict.metadata())
        node.model_config = self.__determine_model_config(metadata, node.model_config)
        check_model_config(node.state_dict, node.model_config, self.strip_extra_keys, self.check_mandatory_keys, str(node.path))

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

    def __determine_model_config(self, metadata: MutableMapping[str, KeyMetadata], config_hint: Optional[ModelConfig]) -> ModelConfig:
        config = config_hint
        if config is None or config == model_configs.INFER:
            inferred_model_configs = infer_model_configs(metadata)
            if any(self.param_config in cfgs for cfgs in inferred_model_configs):
                config = self.param_config
            elif inferred_model_configs and len(inferred_model_configs[0]) == 1:
                config = next(iter(inferred_model_configs[0]))
            elif config is not None:  # config is INFER, structural is not allowed
                raise RuntimeError("could not infer model config")

        if config is None:
            if not self.structural_metadata:
                self.structural_metadata.update(metadata)
            else:
                for k in list(self.structural_metadata):
                    if k not in metadata:
                        del self.structural_metadata[k]
                    elif (
                        self.structural_metadata[k].shape is None or
                        metadata[k].shape is not None and metadata[k].metadata().shape.numel() > self.structural_metadata[k].metadata().shape.numel()
                    ):
                        self.structural_metadata[k] = metadata[k]
            config = StructuralModelConfig(self.structural_metadata)

        return config


def is_config_stub(config: Optional[model_configs.ModelConfig]) -> bool:
    return getattr(config, "identifier", None) in (None, model_configs.INFER.identifier)


def infer_model_configs(state_dict: Iterable[str]) -> List[Set[ModelConfig]]:
    state_dict_set = set(state_dict)
    configs_affinity = {}

    for model_config in model_configs.get_all():
        config_keys = set(model_config.keys()) | set(alias for aliases in model_config.aliases().values() for alias in aliases)
        matched_keys = state_dict_set.intersection(config_keys)

        # heuristic: accept config only if we match more than 10% of the keys of the state dict
        if len(matched_keys) >= len(state_dict_set) * 0.1:
            configs_affinity[model_config] = len(matched_keys)
        # heuristic: break early if we match more than 90% of the keys of a config
        if len(matched_keys) == len(state_dict_set) and len(matched_keys) >= len(model_config.keys()) * 0.9:
            break

    best_configs = sorted((
        {c for c, n in configs_affinity.items() if n == affinity}
        for affinity in set(configs_affinity.values())
    ), key=lambda s: configs_affinity[next(iter(s))], reverse=True)
    return best_configs


def check_model_config(state_dict: Iterable[str], config: ModelConfig, strip_extra_keys: bool, check_mandatory_keys: bool, state_dict_origin: str):
    state_dict_keys = set(state_dict)

    if strip_extra_keys:
        config_keys = set(k_alias for k, v in config.keys().items() for k_alias in [k, *v.aliases])
        extra_keys = state_dict_keys - config_keys
        if extra_keys:
            logging.warning(f"Found extra keys in state dict {state_dict_origin}: {extra_keys}")

    if check_mandatory_keys:
        missing_keys = set()
        for k, v in config.keys().items():
            if v.optional:
                continue

            for k_alias in v.aliases:
                if k_alias in state_dict_keys:
                    k = k_alias
                    break

            if k not in state_dict_keys:
                missing_keys.add(k)

        if missing_keys:
            raise RuntimeError(f"State dict {state_dict_origin} is missing non-optional keys: {missing_keys}")


@dataclasses.dataclass
class CloseInputDictsVisitor(RecipeVisitor):
    def visit_literal(self, node: LiteralRecipeNode):
        if isinstance(node.value, Mapping):
            if not node.value or isinstance(next(iter(node.value.values())), RecipeNode):
                for v in node.value.values():
                    v.accept(self)

    def visit_model(self, node: recipe_nodes.ModelRecipeNode):
        if node.state_dict is not None:
            node.state_dict.close()
        node.state_dict = None

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode):
        for model in (*node.args, *node.kwargs.values()):
            model.accept(self)


@dataclasses.dataclass
class KeyMergeVisitor(RecipeVisitor):
    key: str

    def visit_literal(self, node: LiteralRecipeNode):
        value = node.value
        if isinstance(node.value, Mapping):
            value = self.__visit_sd(node.value, node.model_config)
        if isinstance(value, NonDictLiteralValue):
            return value
        if isinstance(value, RecipeNode):
            return value.accept(self)
        raise TypeError(f"Unexpected literal node value of type {type(value)}")

    def visit_model(self, node: recipe_nodes.ModelRecipeNode) -> torch.Tensor:
        return self.__visit_sd(node.state_dict, node.model_config)

    def __visit_sd(self, state_dict, model_config):
        aliases = model_config.aliases().get(self.key, [])
        keys = [self.key, *aliases]

        for i, key in enumerate(keys):
            try:
                return state_dict[key]
            except KeyError as e:
                if i == len(keys) - 1:
                    raise StateDictKeyError(self.key) from e

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

    def __contains__(self, item):
        return item in self.compute_keys()

    def keys(self) -> Iterable[str]:
        return self.compute_keys().keys()

    @property
    def model_config(self) -> ModelConfig:
        return self.merge_node.model_config

    def compute_keys(self):
        return self.model_config.keys()


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
    if issubclass(expected_type, bool):
        return bool(value)
    if issubclass(expected_type, torch.Tensor):
        return torch.tensor(value, dtype=torch.float32)
    return value


@dataclasses.dataclass
class CastInputDicts(RecipeVisitor):
    device: str | torch.device
    dtype: torch.dtype

    def visit_literal(self, node: LiteralRecipeNode):
        if (
            isinstance(node.value, torch.Tensor | int | float) or
            isinstance(node.value, Mapping) and isinstance(next(iter(node.value.values())), torch.Tensor | int | float)
        ):
            return node.to(device=self.device, dtype=self.dtype)
        if isinstance(node.value, Mapping) and isinstance(next(iter(node.value.values())), RecipeNode):
            return LiteralRecipeNode(
                {k: v.accept(self) for k, v in node.value.items()},
                model_config=node.model_config,
                merge_space=node.merge_space,
            )
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
