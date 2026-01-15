import contextlib
import dataclasses
import threading
from collections import defaultdict
from sd_mecha.recipe_nodes import LiteralRecipeNode, MergeRecipeNode, ModelRecipeNode, RecipeNode, RecipeVisitor
from typing import Any, Dict, Mapping, Optional, Set, Tuple, Union


# todo: test this function
def create_merge_method_context(recipe: RecipeNode, active_keys: Mapping[RecipeNode, Set[str]]) -> Dict[RecipeNode, "MergeMethodContext"]:
    ports_visitor = GetOutputPortsVisitor(active_keys)
    recipe.accept(ports_visitor)
    create_context_visitor = CreateMergeMethodContextVisitor(ports_visitor.parent_ids)
    res = recipe.accept(create_context_visitor)
    return res


@dataclasses.dataclass
class GetOutputPortsVisitor(RecipeVisitor):
    active_keys: Mapping[RecipeNode, Set[str]]
    parent_ids: Dict[RecipeNode, Dict[str, Set[Tuple[RecipeNode, str]]]] = dataclasses.field(default_factory=lambda: defaultdict(lambda: defaultdict(set)))

    def visit_literal(self, node: LiteralRecipeNode):
        for child_node in set(node.value_dict.values()):
            if isinstance(child_node, RecipeNode):
                child_node.accept(self)

    def visit_model(self, node: ModelRecipeNode):
        pass

    def visit_merge(self, node: MergeRecipeNode):
        param_names = node.merge_method.get_param_names().as_dict(len(node.bound_args.args))
        key_map = node.key_map()

        for output_key in self.active_keys[node]:
            for input_idx, param_name in param_names.items():
                child_node = node.bound_args.args[input_idx] if isinstance(input_idx, int) else node.bound_args.kwargs[input_idx]
                if output_key not in key_map:
                    continue

                child_output_keys = self.active_keys[child_node].intersection(key_map[output_key].inputs[param_name])
                for child_output_key in child_output_keys:
                    self.parent_ids[child_node][child_output_key].add((node, output_key))

        for input_node in (*node.bound_args.args, *node.bound_args.kwargs.values()):
            input_node.accept(self)


@dataclasses.dataclass
class CreateMergeMethodContextVisitor(RecipeVisitor):
    parent_ids: Mapping[RecipeNode, Mapping[str, Set[Tuple[RecipeNode, str]]]] = dataclasses.field(default_factory=dict)

    def visit_literal(self, node: LiteralRecipeNode) -> Dict[RecipeNode, "MergeMethodContext"]:
        res = {}
        for key, nested_node in node.value_dict.items():
            if isinstance(nested_node, RecipeNode):
                res |= nested_node.accept(self)
        return res

    def visit_model(self, node: ModelRecipeNode) -> Dict[RecipeNode, "MergeMethodContext"]:
        return {}

    def visit_merge(self, node: MergeRecipeNode) -> Dict[RecipeNode, "MergeMethodContext"]:
        res = {}
        for input_node in (*node.bound_args.args, *node.bound_args.kwargs.values()):
            res |= input_node.accept(self)

        key_map = node.key_map()

        node_parent_ids = self.parent_ids[node]
        locks = {
            k: l
            for t, l in ((t, threading.Lock()) for t in key_map.n_to_n_map)
            for k in t if k in node_parent_ids
        }

        res[node] = MergeMethodContext(
            {
                output_key: MergeMethodOutputRef(node_parent_ids[output_key], None, locks[output_key])
                for output_key in locks
                if len(node_parent_ids[output_key]) >= 2 or len(key_map[output_key].outputs) >= 2
            },
            node.merge_method.instantiate(),
        )

        return res


@dataclasses.dataclass
class MergeMethodContext:
    output_refs: Dict[str, "MergeMethodOutputRef"]
    instance: Any

    @contextlib.contextmanager
    def output_ref_context(self, key: str, lock: bool = True) -> "MergeMethodOutputRef":
        output_ref = self.output_refs.get(key)
        if output_ref is None:
            yield MergeMethodOutputRef(set(), None, contextlib.nullcontext(), locked=True)
        elif lock:
            with output_ref:
                yield output_ref
            if output_ref.was_freed():
                self.output_refs.pop(key, None)
        else:
            with output_ref.assumed_locked():
                yield output_ref


@dataclasses.dataclass
class MergeMethodOutputRef:
    remaining_ports: Set[Tuple[RecipeNode, str]]
    cache: Any
    lock: Union[threading.Lock, contextlib.AbstractContextManager]
    locked: bool = False

    def use_once(self, port: Optional[Tuple[RecipeNode, str]]) -> Any:
        assert self.locked
        res = self.cache
        self.remaining_ports.discard(port)
        if self.was_freed():
            self.cache = None
        return res

    def was_freed(self) -> bool:
        return len(self.remaining_ports) <= 0

    def set_cache(self, cache: Any):
        assert self.locked
        if self.was_freed():
            return
        assert self.cache is None, f"cache was already set: {self.cache}, trying to assign {cache}"
        self.cache = cache

    @contextlib.contextmanager
    def assumed_locked(self):
        old = self.locked
        self.locked = True
        try:
            yield self
        finally:
            self.locked = old

    def __enter__(self):
        self.lock.__enter__()
        self.locked = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.locked = False
        self.lock.__exit__(exc_type, exc_val, exc_tb)
