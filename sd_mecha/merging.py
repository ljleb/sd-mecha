import contextlib
import dataclasses
import functools
import logging
import os
import pathlib
import sys
import threading
import typing
import torch
import sd_mecha
from .extensions import merge_spaces, model_dirs
from sd_mecha.merge_context import create_merge_method_context, MergeMethodContext
from .extensions.merge_methods import value_to_node, StateDict, T as MergeMethodT
from .extensions.merge_spaces import MergeSpace
from .extensions.model_configs import ModelConfig, KeyMetadata
from .graph_finalization import open_graph
from .keys_map import ActiveKeyMap, RealizedKeyRelation
from .recipe_nodes import (
    ClosedModelRecipeNode, RecipeVisitor, LiteralRecipeNode, RecipeNode, MergeRecipeNode,
    ModelRecipeNode, RecipeNodeOrValue, NonDictLiteralValue, PythonLiteralValue,
)
from .streaming import OutSafetensorsDict, TensorMetadata, StateDictKeyError
from . import recipe_nodes, serialization
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import nullcontext
from tqdm import tqdm as tqdm_original
from types import SimpleNamespace
from typing import Optional, Mapping, MutableMapping, Iterable, Tuple, Dict, TypeVar, Sequence, Any
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

    For each key of the target model config, execute all instructions of the recipe graph
    and store the result into a dictionary using the specified output strategy.

    Args:
        recipe:
            A RecipeNode, python literal, or dictionary describing how to merge or transform multiple models.
        fallback_model (optional):
            A secondary recipe or model to provide values for any keys missing from `recipe`.
        merge_device (optional):
            Torch device to load input tensors onto while merging (e.g., "cpu" or "cuda:0").
        merge_dtype (optional):
            Torch dtype for input merges (e.g., `torch.float32`, `torch.float64`).
        output_device (optional):
            Final output device (e.g., "cpu").
        output_dtype (optional):
            Final dtype for the saved model.
        threads (optional):
            Number of threads to spawn for parallel merges. Defaults to a reasonable guess.
        total_buffer_size (optional):
            Total byte size of the buffers for all safetensors state dicts (input and output).
        strict_merge_space (optional):
            If specified, verifies that the output merge space corresponds to the given merge space.
            If False, merges can happen in any merge space, preferring when many are possible in order:
            "weight", "delta", "param", <user-defined>.
            Defaults to None.
        strict_mandatory_keys (optional):
            If True and an input model is missing non-optional keys, raises RuntimeError. Defaults to False.
        check_extra_keys (optional):
            If True, warns about unrecognized keys from the input models. Defaults to True.
        check_finite_output (optional):
            If True, warns if any non-finite values appear in the output model. Defaults to True.
        omit_non_finite_inputs (optional):
            If True, automatically discards input keys containing non-finite values. Defaults to True.
        memoize_intermediates (optional):
            If True, temporarily memoizes the output of merge nodes that are used by multiple parents to avoid recomputations.
            To minimize extra memory usage, each memoized output is freed as soon as all its consumers have consumed it.
            Defaults to True.
        validate_mm_contract (optional):
            If True, validates that merge methods return the right amount of outputs indicated by `map_keys`
            and do not read other inputs than those reported by `map_keys`. Defaults to True.
        cache (optional):
            Dictionary of caches for any recipe nodes in `recipe`.
            Items should be created like this: `node: node.create_cache()`.
            This can speed up certain merge methods when testing multiple parameter variations with fixed inputs.
        tqdm (optional):
            A custom progress-bar factory. By default, uses `tqdm.tqdm`.
        output (optional):
            Where to store the merged state dict.
            Can be a filesystem path (string or `Path`) ending with `.safetensors`, an in-memory dict-like object, or None.
            If it is None or omitted, a new dict is returned when the merge completes.

    Returns:
        The in-memory dictionary if `output` is either a MutableMapping or None, and nothing if `output` is a file path.
    """
    if fallback_model is ...:
        fallback_model = None
    if merge_device is ...:
        merge_device = "cpu"
    if merge_dtype is ...:
        merge_dtype = torch.float32
    cast_inputs = merge_device is not None and merge_dtype is not None
    if output_device is ...:
        output_device = "cpu"
    if output_dtype is ...:
        output_dtype = torch.float16
    if threads is ...:
        threads = None
    if total_buffer_size is ...:
        total_buffer_size = 2**28
    if strict_merge_space is ...:
        strict_merge_space = None
    if strict_mandatory_keys is ...:
        strict_mandatory_keys = False
    if check_extra_keys is ...:
        check_extra_keys = True
    if check_finite_output is ...:
        check_finite_output = True
    if omit_non_finite_inputs is ...:
        omit_non_finite_inputs = True
    if memoize_intermediates is ...:
        memoize_intermediates = True
    if validate_mm_contract is ...:
        validate_mm_contract = True
    if cache is ...:
        cache = {}
    if tqdm is ...:
        tqdm = tqdm_original
    if output is ...:
        output = None

    if threads is not None and (not isinstance(threads, int) or threads < 0):
        raise RuntimeError("threads should be a non-negative integer or None")

    recipe = value_to_node(recipe)
    original_recipe = recipe

    if fallback_model is not None:
        recipe |= fallback_model

    if output_device is not None or output_dtype is not None:
        recipe = recipe.to(device=output_device, dtype=output_dtype)

    original_to_casted = None
    if cast_inputs or omit_non_finite_inputs:
        cast_visitor = CastInputDicts(merge_device, merge_dtype, omit_non_finite_inputs)
        recipe = recipe.accept(cast_visitor)
        original_to_casted = cast_visitor.converted_nodes

    total_files_open = (
        recipe.accept(recipe_nodes.ModelsCountVisitor()) +
        int(isinstance(output, (str, pathlib.Path)))
    )
    buffer_size_per_file = total_buffer_size // max(1, total_files_open)
    if threads is None:
        threads = min(max(total_files_open, 2), os.cpu_count() or 0, 4)

    if threads == 0:
        thread_local_data = SimpleNamespace()
        executor = ThisThreadExecutor()
    else:
        thread_local_data = threading.local()
        executor = ThreadPoolExecutor(max_workers=threads)

    with open_graph(
        recipe,
        buffer_size_per_file,
    ) as graph:
        finalized_res = graph.finalize_with_keys(
            check_extra_keys=check_extra_keys,
            check_mandatory_keys=strict_mandatory_keys,
            model_config_preference=("singleton-mecha",),
            merge_space=strict_merge_space if strict_merge_space is not None else None,
            merge_space_preference=merge_spaces.get_all() if strict_merge_space is None else None,
        )
        recipe, realized_key_maps = finalized_res.root, finalized_res.realized_key_maps
        realized_root_key_map = realized_key_maps[recipe]
        if original_to_casted is not None:
            original_to_finalized = {original: finalized_res.to_finalized_node[node] for original, node in original_to_casted.items()}
        else:
            original_to_finalized = finalized_res.to_finalized_node

        cache = {original_to_finalized[node]: cache_object for node, cache_object in cache.items()}
        original_recipe = ReplaceSolvedComponents(original_to_finalized).process(original_recipe)

        graph_metadata = {k: v for k, v in recipe.model_config.keys().items() if k in realized_root_key_map}
        buffer_size_per_file_per_thread = buffer_size_per_file // max(1, threads)
        merge_methods_context = create_merge_method_context(
            recipe,
            realized_key_maps,
            finalized_res.node_to_keys,
            memoize_intermediates,
        )

        with (
            executor,
            tqdm(total=len(graph_metadata), desc="Merging recipe") as progress,
            _get_output_dict(
                output,
                graph_metadata,
                original_recipe,
                buffer_size_per_file_per_thread,
            ) as output_dict,
        ):
            _fix_torch_threading()
            futures = []
            for key, key_metadata in graph_metadata.items():
                fn = recipe.accept
                fn = _track_output(fn, output_dict, key, key_metadata, check_finite_output, strict_mandatory_keys)
                fn = _track_progress(fn, key, graph_metadata[key].shape, progress)
                fn = _wrap_thread_context(fn, thread_local_data)
                merge_visitor = KeyMergeVisitor(key, merge_methods_context, validate_mm_contract, cache, realized_key_maps)
                futures.append(executor.submit(fn, merge_visitor))
            _resolve_futures(futures)

            for node, mm_context in merge_methods_context.items():
                unique_refs = {id(ref): ref for ref in mm_context.output_refs.values()}.values()
                num_leaked = sum(not output_ref.was_freed() for output_ref in unique_refs)
                if num_leaked:
                    logging.warning(f"memory leaked during the merge: {node}, number of entries: {num_leaked}")

            if output is None:
                return output_dict


def _fix_torch_threading():
    if torch.cuda.is_available():
        # this greedy loads the torch.linalg module
        # avoids a hard error caused by threads>1 with some torch ops
        # see https://github.com/pytorch/pytorch/issues/90613
        torch.linalg.inv(torch.ones((1, 1), device="cuda"))

    globals()['fix_torch_threading'] = lambda: None


@contextlib.contextmanager
def _get_output_dict(
    output: Optional[MutableMapping[str, torch.Tensor]] | pathlib.Path | str,
    merged_header: Mapping[str, KeyMetadata],
    recipe: RecipeNode,
    buffer_size_per_thread: int,
):
    try:
        serialized_recipe = serialization.serialize(recipe, finalize=False)  # already finalized
    except TypeError:
        logging.warning("The recipe graph could not be serialized. The output state dict will not contain the recipe.")
        serialized_recipe = None

    if isinstance(output, (str, pathlib.Path)):
        if not isinstance(output, pathlib.Path):
            output = pathlib.Path(output)
        if not output.is_absolute():
            for model_dir in model_dirs.get_all():
                output = model_dir / output
                break
        logging.info(f"Saving to {output}")

        streamed_output = OutSafetensorsDict(
            output,
            OrderedDict((k, TensorMetadata(v.shape, v.dtype)) for k, v in merged_header.items()),
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
        if serialized_recipe is not None:
            output["__metadata__"] = {"mecha_recipe": serialized_recipe}
        yield output


def _track_output(fn, output, key: str, key_metadata: KeyMetadata, check_finite: bool, strict_mandatory_keys: bool):
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
                    message = f"There are non finite values in key '{key}': {key_metadata}"
                    if key_metadata.optional:
                        logging.debug(message)
                    else:
                        logging.warning(message)

            output[key] = res
            return res
        except StateDictKeyError as e:
            if key_metadata.optional or not strict_mandatory_keys:
                logging.debug(f"Skipping key: {e}")
            else:
                raise RuntimeError(f"Could not merge mandatory key: {e}") from e
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
        try:
            result.set_result(fn(*args, **kwargs))
        except BaseException as e:
            result.set_exception(e)
        return result


def _resolve_futures(futures: Sequence[Future]):
    from concurrent.futures import wait, FIRST_EXCEPTION
    done, not_done = wait(futures, return_when=FIRST_EXCEPTION)

    first_exc = None
    for future in done:
        exc = future.exception()
        if exc is not None:
            first_exc = exc
            break

    if first_exc is not None:
        for future in not_done:
            future.cancel()
        raise first_exc

    for future in not_done:
        future.result()


@dataclasses.dataclass
class KeyMergeVisitor(RecipeVisitor):
    output_key: str
    merge_methods_context: Mapping[RecipeNode, MergeMethodContext]
    validate_mm_contract: bool
    merge_methods_caches: Mapping[RecipeNode, Any]
    realized_relations: Mapping[RecipeNode, ActiveKeyMap[RealizedKeyRelation]]
    parent_port: Optional[Tuple[RecipeNode, str, int | str]] = None

    def visit_literal(self, node: LiteralRecipeNode):
        value = self.__visit_sd(node.value_dict, node.model_config)
        if isinstance(value, NonDictLiteralValue):
            return value
        if isinstance(value, RecipeNode):
            return value.accept(self)
        raise TypeError(f"Unexpected literal node value of type {type(value)}")

    def visit_model(self, node: recipe_nodes.ModelRecipeNode) -> torch.Tensor:
        return self.__visit_sd(node.state_dict, node.model_config)

    def __visit_sd(self, state_dict, model_config):
        aliases = model_config.aliases().get(self.output_key, [])
        possible_keys = [self.output_key, *aliases]

        for i, possible_key in enumerate(possible_keys):
            try:
                return state_dict[possible_key]
            except KeyError as e:
                if i == len(possible_keys) - 1:
                    raise StateDictKeyError(self.output_key) from e

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode) -> torch.Tensor:
        context = self.merge_methods_context[node]
        with context.output_ref_context(self.output_key) as output_ref:
            if output_ref.cache is not None:
                res = output_ref.use_once(self.parent_port)
            else:
                try:
                    key_map = self.realized_relations[node]
                    if self.output_key not in key_map:
                        raise StateDictKeyError(self.output_key)
                    key_relation = key_map[self.output_key]

                    merged_args, merged_kwargs = self.__visit_deeper_first(node, key_relation)
                    res = node.merge_method.merge_key(
                        merged_args,
                        merged_kwargs,
                        self.output_key,
                        key_relation,
                        self.merge_methods_caches.get(node),
                        context.instance,
                        self.output_key in context.reused_output_keys,
                    )
                finally:
                    if self.parent_port is None:
                        release_visitor = KeyReleaseVisitor(self.output_key, self.merge_methods_context, self.realized_relations, self.parent_port, needs_lock=False)
                        node.accept(release_visitor)
                if isinstance(res, dict):
                    if self.validate_mm_contract:
                        assert set(key_relation.outputs) == set(res.keys()), (
                            f"Merge method {node.merge_method.identifier} returned an unexpected set of keys: {list(res)}",
                        )
                    for key, value in res.items():
                        with context.output_ref_context(key, lock=False) as sibling_output_ref:
                            # output_ref.cache is set here
                            sibling_output_ref.set_cache(value)
                    res = res[self.output_key]
                else:
                    output_ref.set_cache(res)

        return res

    def __visit_deeper_first(
        self,
        node: MergeRecipeNode,
        key_relation: RealizedKeyRelation,
    ) -> Tuple[
        Sequence[NonDictLiteralValue | StateDict[NonDictLiteralValue]],
        Mapping[str, NonDictLiteralValue | StateDict[NonDictLiteralValue]],
    ]:
        def depth_of_value(index) -> int:
            nodes = node.bound_args.args if isinstance(index, int) else node.bound_args.kwargs
            return nodes[index].accept(recipe_nodes.ModelDepthRecipeVisitor())

        error_holder = ErrorHolder()
        merged = {}
        input_types = node.merge_method.get_input_types().as_dict(len(node.bound_args.args))
        input_names = node.merge_method.get_param_names().as_dict(len(node.bound_args.args))
        indices = (*range(len(node.bound_args.args)), *node.bound_args.kwargs.keys())

        for index in sorted(indices, key=depth_of_value, reverse=True):
            parent_port = (node, self.output_key, index)
            input_name = input_names[index]
            input_node = node.bound_args.args[index] if isinstance(index, int) else node.bound_args.kwargs[index]
            input_visitor = dataclasses.replace(self, parent_port=parent_port)
            if is_subclass(input_types[index], StateDict):
                if self.validate_mm_contract:
                    input_keys_constraints = key_relation.inputs.get(input_name, ())
                else:
                    input_keys_constraints = None
                expected_type = next(iter(typing.get_args(input_types[index]) or (MergeMethodT,)))
                merged[index] = error_holder.intercept(MergeNodeWrapperStateDict, input_node, expected_type, input_visitor, input_keys_constraints)
            else:
                merged[index] = cast_node_value(error_holder.intercept(input_node.accept, input_visitor), input_types[index])

        merged_args = tuple(merged.get(index) for index in range(len(node.bound_args.args)))
        merged_kwargs = {k: v for k, v in merged.items() if not isinstance(k, int)}
        error_holder.try_raise()
        return merged_args, merged_kwargs


@dataclasses.dataclass
class KeyReleaseVisitor(RecipeVisitor):
    output_key: str
    merge_methods_context: Mapping[RecipeNode, MergeMethodContext]
    realized_relations: Mapping[RecipeNode, ActiveKeyMap[RealizedKeyRelation]]
    parent_port: Optional[Tuple[RecipeNode, str, int | str]]
    needs_lock: bool

    def visit_literal(self, node: LiteralRecipeNode):
        value = self.__visit_sd(node.value_dict, node.model_config)
        if isinstance(value, RecipeNode):
            value.accept(self)

    def visit_model(self, node: recipe_nodes.ModelRecipeNode):
        pass

    def __visit_sd(self, state_dict, model_config):
        aliases = model_config.aliases().get(self.output_key, [])
        keys = [self.output_key, *aliases]

        for i, key in enumerate(keys):
            try:
                return state_dict[key]
            except KeyError:
                pass

        return None

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode):
        context = self.merge_methods_context[node]
        with context.output_ref_context(self.output_key, lock=self.needs_lock) as output_ref:
            output_ref.use_once(self.parent_port)
            self.__visit_deeper_first(node)

    def __visit_deeper_first(
        self,
        node: MergeRecipeNode,
    ):
        error_holder = ErrorHolder()

        key_map = self.realized_relations[node]
        if self.output_key not in key_map:
            return

        key_relation = key_map[self.output_key]

        for input_idx, input_param in node.merge_method.get_param_names().as_dict(len(node.bound_args.args)).items():
            input_node = node.bound_args.args[input_idx] if isinstance(input_idx, int) else node.bound_args.kwargs[input_idx]
            parent_port = (node, self.output_key, input_idx)
            release_visitors = [
                KeyReleaseVisitor(input_key, self.merge_methods_context, self.realized_relations, parent_port, needs_lock=True)
                for input_key in key_relation.inputs.get(input_param, ())
            ]
            for release_visitor in release_visitors:
                error_holder.intercept(input_node.accept, release_visitor)

        error_holder.try_raise()


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
        node: recipe_nodes.RecipeNode,
        expected_type: type,
        original_merge_visitor: KeyMergeVisitor,
        input_keys_constraints: Optional[Tuple[str, ...]],
    ):
        self.node = node
        self.expected_type = expected_type
        self.original_merge_visitor = original_merge_visitor
        self.input_keys_constraints = input_keys_constraints

    def __getitem__(self, key):
        if self.input_keys_constraints is not None:
            assert key in self.input_keys_constraints, (
                f"node {self.node} tried to fetch key '{key}' "
                f"but only requested these keys through input_keys_for_output(): {self.input_keys_constraints}"
            )

        key_merger = dataclasses.replace(self.original_merge_visitor, output_key=key)
        res_raw = self.node.accept(key_merger)
        res = cast_node_value(res_raw, self.expected_type)
        return res

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
        return self.node.model_config

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
    omit_non_finite: bool
    converted_nodes: Dict[RecipeNode, RecipeNode] = dataclasses.field(default_factory=dict)

    def visit_literal(self, node: LiteralRecipeNode):
        if node in self.converted_nodes:
            return self.converted_nodes[node]

        converted_dict = {}
        can_omit = False
        cast_recipe = False
        for k, v in node.value_dict.items():
            if isinstance(v, RecipeNode):
                v = v.accept(self)
            elif isinstance(v, PythonLiteralValue):
                v = v
            elif isinstance(v, torch.Tensor):
                can_omit = True
                cast_recipe = True
            else:
                raise RuntimeError(f"Cannot cast type {type(v)} to device={self.device}, dtype={self.dtype}")
            converted_dict[k] = v

        res = LiteralRecipeNode(converted_dict, node.model_config, node.merge_space)
        if cast_recipe:
            res = res.to(device=self.device, dtype=self.dtype)
        if can_omit and self.omit_non_finite:
            res = sd_mecha.omit_non_finite(res)
        self.converted_nodes[node] = res
        return res

    def visit_model(self, node: ModelRecipeNode):
        if node in self.converted_nodes:
            return self.converted_nodes[node]

        res = node.to(device=self.device, dtype=self.dtype)
        if self.omit_non_finite:
            res = sd_mecha.omit_non_finite(res)
        self.converted_nodes[node] = res
        return res

    def visit_merge(self, node: MergeRecipeNode):
        if node in self.converted_nodes:
            return self.converted_nodes[node]

        args = tuple(v.accept(self) for v in node.bound_args.args)
        kwargs = {k: v.accept(self) for k, v in node.bound_args.kwargs.items()}
        bound_args = node.merge_method.get_signature().bind(*args, **kwargs)
        res = MergeRecipeNode(
            node.merge_method,
            bound_args,
            node.model_config,
            node.merge_space,
        )
        self.converted_nodes[node] = res
        return res


@dataclasses.dataclass
class ReplaceSolvedComponents(RecipeVisitor):
    to_finalized: Mapping[RecipeNode, RecipeNode]
    node_map: Dict[RecipeNode, RecipeNode] = dataclasses.field(default_factory=defaultdict)

    def process(self, node: RecipeNode) -> RecipeNode:
        res = self.node_map.get(node)
        if res is None:
            if node.model_config is not None and node.merge_space is not None:
                res = node
            else:
                res = node.accept(self)
            self.node_map[node] = res
        return res

    def visit_literal(self, node: LiteralRecipeNode) -> RecipeNode:
        finalized_node = self.to_finalized[node]
        return LiteralRecipeNode(
            {
                k: v.accept(self) if isinstance(v, RecipeNode) else v
                for k, v in node.value_dict.items()
            },
            finalized_node.model_config,
            finalized_node.merge_space,
        )

    def visit_model(self, node: ModelRecipeNode) -> RecipeNode:
        finalized_node = self.to_finalized[node]
        return ClosedModelRecipeNode(
            node.path,
            finalized_node.model_config,
            finalized_node.merge_space,
        )

    def visit_merge(self, node: MergeRecipeNode) -> RecipeNode:
        finalized_node = self.to_finalized[node]
        return MergeRecipeNode(
            node.merge_method,
            node.bound_args.signature.bind(
                *(self.process(v) for v in node.bound_args.args),
                **{k: self.process(v) for k, v in node.bound_args.kwargs.items()}
            ),
            finalized_node.model_config,
            finalized_node.merge_space,
        )
