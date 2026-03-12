import contextlib
import dataclasses
import functools
import logging
import operator
import os
import pathlib
import sys
import threading
import typing
import torch

import sd_mecha
from .extensions import model_dirs
from sd_mecha.merge_context import create_merge_method_context, MergeMethodContext
from .extensions.merge_methods import value_to_node, StateDict, T as MergeMethodT
from .extensions.merge_spaces import MergeSpace
from .extensions.model_configs import ModelConfig, KeyMetadata
from .graph_finalization import open_graph
from .keys_map import KeyRelation
from .recipe_nodes import RecipeVisitor, LiteralRecipeNode, RecipeNode, MergeRecipeNode, ModelRecipeNode, RecipeNodeOrValue, NonDictLiteralValue
from .streaming import OutSafetensorsDict, TensorMetadata, StateDictKeyError
from . import recipe_nodes, serialization
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from contextlib import nullcontext
from tqdm import tqdm as tqdm_original
from types import SimpleNamespace
from typing import Optional, Mapping, MutableMapping, List, Iterable, Tuple, Dict, TypeVar, Sequence, Any
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
    strict_weight_space: bool = ...,
    check_finite: bool = ...,
    check_finite_mandatory_inputs: bool = ...,
    check_finite_optional_inputs: bool = ...,
    omit_extra_keys: bool = ...,
    omit_ema: bool = ...,
    check_mandatory_keys: bool = ...,
    tqdm: type = ...,
    validate_mm_contract: bool = ...,
    cache: Dict[RecipeNode, dict] = ...,
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
        strict_weight_space (optional):
            If True, verifies that merges occur in "weight" space. If False, merges can happen
            in other merge spaces (like "delta" or "param"). Defaults to True.
        check_finite (optional):
            If True, warns if any non-finite values appear in the output model. Defaults to True.
        check_finite_mandatory_inputs (optional):
            If True, automatically discards (raises StateDictKeyError) directly from model nodes for mandatory key containing non-finite values. Defaults to False.
        check_finite_optional_inputs (optional):
            If True, automatically discards (raises StateDictKeyError) directly from model nodes for optional key containing non-finite values. Defaults to True.
        omit_extra_keys (optional):
            If True, warns about and removes unrecognized keys from the output model. Defaults to True.
        omit_ema (optional):
            If True, omits ema keys from the output model. Defaults to omit_extra_keys.
        check_mandatory_keys (optional):
            If True and an input model is missing non-optional keys, raises RuntimeError. Defaults to False.
        tqdm (optional):
            A custom progress-bar factory. By default, uses `tqdm.tqdm`.
        output (optional):
            Where to store the merged state dict. Can be a filesystem path (string or
            `Path`) ending with `.safetensors`, an in-memory dict-like object, or None.
            If it is None or omitted, an empty dict is created and returned when the merge completes.
        validate_mm_contract (optional):
            If True, validates that merge methods return the right amount of outputs indicated by `map_keys`
            and do not read other inputs than those reported by `map_keys`
        cache (optional):
            Dictionary of caches for each recipe node in `recipe`. Each dict should be empty on the first call.
            If set, the dicts are filled by the respective merge methods on the first call to merge(), and then reused for subsequent calls to merge().
            This can speed up certain merge methods when testing multiple parameter variations with fixed inputs.

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
    if strict_weight_space is ...:
        strict_weight_space = True
    if check_finite is ...:
        check_finite = True
    if check_finite_mandatory_inputs is ...:
        check_finite_mandatory_inputs = False
    if check_finite_optional_inputs is ...:
        check_finite_optional_inputs = True
    check_finite_inputs = check_finite_mandatory_inputs or check_finite_optional_inputs
    if omit_extra_keys is ...:
        omit_extra_keys = True
    if omit_ema is ...:
        omit_ema = omit_extra_keys
    if check_mandatory_keys is ...:
        check_mandatory_keys = False
    if tqdm is ...:
        tqdm = tqdm_original
    if validate_mm_contract is ...:
        validate_mm_contract = True
    if cache is ...:
        cache = {}

    if threads is not None and (not isinstance(threads, int) or threads < 0):
        raise RuntimeError("threads should be a non-negative integer or None")

    recipe = value_to_node(recipe)
    original_recipe = recipe
    if cast_inputs or check_finite_inputs:
        cast_visitor = CastInputDicts(merge_device, merge_dtype, check_finite_mandatory_inputs, check_finite_optional_inputs)
        recipe = recipe.accept(cast_visitor)
        cache = {cast_visitor.converted_nodes.get(node, node): cache_dict for node, cache_dict in cache.items()}

    if fallback_model is not None:
        recipe = recipe | fallback_model

    if output_device is not None or output_dtype is not None:
        recipe = recipe.to(device=output_device, dtype=output_dtype)

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
        finalized_res = graph.finalize(
            check_extra_keys=omit_extra_keys,
            check_mandatory_keys=check_mandatory_keys,
            model_config_preference=("singleton-mecha",),
            merge_space="weight" if strict_weight_space else None,
            merge_space_preference=("weight",) if not strict_weight_space else None,
        )
        recipe, node_to_keys = finalized_res.root, finalized_res.node_to_keys
        minimal_keys = node_to_keys[recipe]
        cache = {finalized_res.to_finalized_node[node]: cache_dict for node, cache_dict in cache.items()}

        model_config = recipe.model_config
        graph_metadata = {k: v for k, v in recipe.model_config.keys().items() if k in minimal_keys}
        if not omit_extra_keys:
            recipe, graph_metadata = copy_extra_keys(recipe)

        if omit_ema and "ema" in model_config.components():
            for key in model_config.components()["ema"].keys():
                if key in graph_metadata:
                    # remove ema keys from merge plan
                    del graph_metadata[key]

        buffer_size_per_file_per_thread = buffer_size_per_file // max(1, threads)
        merge_methods_context = create_merge_method_context(recipe, node_to_keys)

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
            fix_torch_threading()
            futures = []
            for key, key_metadata in graph_metadata.items():
                fn = recipe.accept
                fn = _track_output(fn, output_dict, key, key_metadata, check_finite)
                fn = _track_progress(fn, key, graph_metadata[key].shape, progress)
                fn = _wrap_thread_context(fn, thread_local_data)
                futures.append(executor.submit(fn, KeyMergeVisitor(key, merge_methods_context, validate_mm_contract, cache)))

            for future in as_completed(futures):
                if future.exception() is not None:
                    for future_to_cancel in futures:
                        future_to_cancel.cancel()
                future.result()

            for node, mm_context in merge_methods_context.items():
                num_leaked = sum(not output_ref.was_freed() for output_ref in mm_context.output_refs.values())
                if num_leaked:
                    logging.warning(f"memory leaked during the merge: {node}, number of entries: {num_leaked}")

            if isinstance(output_dict, MutableMapping):
                return output_dict


def fix_torch_threading():
    if torch.cuda.is_available():
        # this greedy loads the torch.linalg module
        # avoids a hard error caused by threads>1 with some torch ops
        # see https://github.com/pytorch/pytorch/issues/90613
        torch.linalg.inv(torch.ones((1, 1), device="cuda"))

    globals()['fix_torch_threading'] = lambda: None


@dataclasses.dataclass
class MinimalMetadataVisitor(RecipeVisitor):
    def visit_literal(self, node: LiteralRecipeNode) -> Mapping[str, KeyMetadata]:
        meta = node.model_config.keys()
        res = {}

        child_keys = defaultdict(set)
        for k, child_node in node.value_dict.items():
            if isinstance(child_node, RecipeNode):
                child_keys[child_node].add(k)

        for k, child_node in node.value_dict.items():
            if isinstance(child_node, RecipeNode):
                child_res = child_node.accept(dataclasses.replace(self))
                for k_res in child_keys[child_node]:
                    child_keys[k_res] = child_res[k_res]
            else:
                res[k] = meta[k]

        return res

    def visit_model(self, node: ModelRecipeNode) -> Mapping[str, KeyMetadata]:
        return {k: v for k, v in node.state_dict.metadata()}

    def visit_merge(self, node: MergeRecipeNode) -> Mapping[str, KeyMetadata]:
        res = {}
        meta = node.model_config.keys()
        key_map = node.key_map()

        for param, args in node.all_args().items():
            for arg in (args if isinstance(args, tuple) else (args,)):
                arg_res = arg.accept(self)
                for input_key in arg_res:
                    output_keys = key_map.in_to_out.get(input_key)
                    output_keys = output_keys.get(param) if output_keys is not None else None
                    if output_keys is None:
                        continue

                    for output_key in output_keys:
                        res[output_key] = meta[output_key]

        return res


def copy_extra_keys(recipe: RecipeNode):
    config = recipe.model_config
    merge_space = recipe.merge_space
    metadata = config.keys()
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
    forwardable_nodes: List[Tuple[RecipeNode, Mapping[str, KeyMetadata]]] = dataclasses.field(default_factory=list)

    def visit_literal(self, node: LiteralRecipeNode):
        if not node.value_dict:
            return

        can_forward = (
            node.model_config == self.target_config and
            node.merge_space == self.target_merge_space and
            not any(node in n[0] for n in self.forwardable_nodes)
        )
        metadata = {}
        for k, v in node.value_dict.items():
            if isinstance(v, RecipeNode):
                v.accept(self)
            elif can_forward and isinstance(v, torch.Tensor):
                metadata[k] = KeyMetadata(v.shape, v.dtype)

        if can_forward and metadata:
            self.forwardable_nodes.append((node, metadata))

    def visit_model(self, node: ModelRecipeNode):
        if node.model_config == self.target_config and not any(node in n[0] for n in self.forwardable_nodes):
            self.forwardable_nodes.append((
                node,
                OrderedDict(
                    (k, KeyMetadata(v.shape, v.dtype))
                    for k, v in node.state_dict.metadata()
                )
            ))

    def visit_merge(self, node: MergeRecipeNode):
        for arg in node.all_args():
            arg.accept(self)


@contextlib.contextmanager
def _get_output_dict(
    output: Optional[MutableMapping[str, torch.Tensor]] | pathlib.Path | str,
    merged_header: Mapping[str, KeyMetadata],
    recipe: RecipeNode,
    buffer_size_per_thread: int,
):
    if isinstance(output, (str, pathlib.Path)):
        if not isinstance(output, pathlib.Path):
            output = pathlib.Path(output)
        if not output.is_absolute():
            for model_dir in model_dirs.get_all():
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
        yield output


def _track_output(fn, output, key: str, key_metadata: KeyMetadata, check_finite: bool):
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
                    logging.warning(f"there are non finite values in key '{key}': {key_metadata}")

            output[key] = res
        except StateDictKeyError as e:
            if key_metadata.optional:
                logging.debug(f"skipping key {e}")
            else:
                raise RuntimeError(f"could not merge mandatory key: {e}") from e
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


@dataclasses.dataclass
class KeyMergeVisitor(RecipeVisitor):
    output_key: str
    merge_methods_context: Dict[RecipeNode, MergeMethodContext]
    validate_mm_contract: bool
    merge_methods_caches: Dict[RecipeNode, Any]
    parent_id: Optional[Tuple[RecipeNode, str]] = None

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
                res = output_ref.use_once(self.parent_id)
            else:
                try:
                    key_map = node.key_map()
                    assert self.output_key in key_map, f"Merge method {node.merge_method} does not produce key {self.output_key}."
                    key_relation = key_map[self.output_key]

                    merged_args, merged_kwargs = self.__visit_deeper_first(node, key_relation)
                    res = node.merge_method.merge_key(
                        merged_args,
                        merged_kwargs,
                        self.output_key,
                        key_relation,
                        self.merge_methods_caches.get(node),
                        context.instance,
                    )
                finally:
                    if self.parent_id is None:
                        release_visitor = KeyReleaseVisitor(self.output_key, self.merge_methods_context, self.parent_id, needs_lock=False)
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
        key_relation: KeyRelation,
    ) -> Tuple[
        Sequence[NonDictLiteralValue | StateDict[NonDictLiteralValue]],
        Mapping[str, NonDictLiteralValue | StateDict[NonDictLiteralValue]],
    ]:
        def depth_of_value(index) -> int:
            nodes = node.bound_args.args if isinstance(index, int) else node.bound_args.kwargs
            return nodes[index].accept(recipe_nodes.ModelDepthRecipeVisitor())

        new_parent_id = (node, self.output_key)
        error_holder = ErrorHolder()
        merged = {}
        input_types = node.merge_method.get_input_types().as_dict(len(node.bound_args.args))
        input_names = node.merge_method.get_param_names().as_dict(len(node.bound_args.args))
        indices = (*range(len(node.bound_args.args)), *node.bound_args.kwargs.keys())

        for index in sorted(indices, key=depth_of_value, reverse=True):
            input_name = input_names[index]
            input_node = node.bound_args.args[index] if isinstance(index, int) else node.bound_args.kwargs[index]
            input_visitor = dataclasses.replace(self, parent_id=new_parent_id)
            if is_subclass(input_types[index], StateDict):
                if self.validate_mm_contract:
                    input_keys_constraints = key_relation.inputs[input_name]
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
    merge_methods_context: Dict[RecipeNode, MergeMethodContext]
    parent_id: Optional[Tuple[RecipeNode, str]]
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
            output_ref.use_once(self.parent_id)
            self.__visit_deeper_first(node)

    def __visit_deeper_first(
        self,
        node: MergeRecipeNode,
    ):
        new_parent_id = (node, self.output_key)
        error_holder = ErrorHolder()

        key_map = node.key_map()
        if self.output_key not in key_map:
            return

        key_relation = key_map[self.output_key]

        for input_idx, input_param in node.merge_method.get_param_names().as_dict(len(node.bound_args.args)).items():
            input_node = node.bound_args.args[input_idx] if isinstance(input_idx, int) else node.bound_args.kwargs[input_idx]
            release_visitors = [
                KeyReleaseVisitor(input_key, self.merge_methods_context, new_parent_id, needs_lock=True)
                for input_key in key_relation.inputs[input_param]
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
    check_finite_mandatory: bool
    check_finite_optional: bool
    converted_nodes: Dict[RecipeNode, RecipeNode] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.check_finite = self.check_finite_mandatory or self.check_finite_optional

    def visit_literal(self, node: LiteralRecipeNode):
        if node in self.converted_nodes:
            return self.converted_nodes[node]

        converted_dict = {}
        for k, v in node.value_dict.items():
            if isinstance(v, RecipeNode):
                converted_dict[k] = v.accept(self)
            elif isinstance(v, (int, float, bool)):
                v = torch.tensor(v, device=self.device, dtype=self.dtype)
                converted_dict[k] = v
            elif isinstance(v, torch.Tensor):
                v = v.to(device=self.device, dtype=self.dtype)
                converted_dict[k] = v
            elif isinstance(v, str):
                converted_dict[k] = v
            else:
                raise RuntimeError(f"Cannot cast type {type(v)} to device={self.device}, dtype={self.dtype}")

        res = LiteralRecipeNode(converted_dict, node.model_config, node.merge_space)
        if self.check_finite:
            res = sd_mecha.omit_non_finite(res, self.check_finite_mandatory, self.check_finite_optional)
        self.converted_nodes[node] = res
        return res

    def visit_model(self, node: ModelRecipeNode):
        if node in self.converted_nodes:
            return self.converted_nodes[node]

        res = node.to(device=self.device, dtype=self.dtype)
        if self.check_finite:
            res = sd_mecha.omit_non_finite(res, self.check_finite_mandatory, self.check_finite_optional)
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
        # if self.check_finite and node.merge_method == sd_mecha.extensions.merge_methods.resolve("convert_singleton"):
        #     res = sd_mecha.omit_non_finite(res, self.check_finite_mandatory, self.check_finite_optional)
        self.converted_nodes[node] = res
        return res
