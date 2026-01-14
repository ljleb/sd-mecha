import contextlib
import dataclasses
import threading
from collections import defaultdict
from sd_mecha.recipe_nodes import LiteralRecipeNode, MergeRecipeNode, ModelRecipeNode, RecipeNode, RecipeVisitor
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple, Union


# todo: test this function
def create_merge_method_context(recipe: RecipeNode, root_keys: Iterable[str]) -> Dict[RecipeNode, "MergeMethodContext"]:
    ports_visitor = GetOutputPortsVisitor(keys_constraint=list(root_keys))
    recipe.accept(ports_visitor)
    create_context_visitor = CreateMergeMethodContextVisitor(ports_visitor.parent_ids)
    res = recipe.accept(create_context_visitor)
    return res


@dataclasses.dataclass
class GetOutputPortsVisitor(RecipeVisitor):
    parent_ids: Dict[RecipeNode, Dict[str, Set[Tuple[RecipeNode, str]]]] = dataclasses.field(default_factory=lambda: defaultdict(lambda: defaultdict(set)))
    keys_constraint: Sequence[str] = dataclasses.field(default_factory=list)

    def visit_literal(self, node: LiteralRecipeNode):
        keys_constraints = defaultdict(list)
        for key, nested_node in node.value_dict.items():
            if key in self.keys_constraint and isinstance(nested_node, RecipeNode):
                keys_constraints[nested_node].append(key)
        for nested_node, keys_constraint in keys_constraints.items():
            nested_node.accept(dataclasses.replace(self, keys_constraint=keys_constraint))

    def visit_model(self, node: ModelRecipeNode):
        pass

    def visit_merge(self, node: MergeRecipeNode):
        param_names = node.merge_method.get_param_names()
        key_map = node.key_map()

        for output_key in self.keys_constraint:
            for input_idx, input_name in param_names.as_dict(len(node.bound_args.args)).items():
                input_node = node.bound_args.args[input_idx] if isinstance(input_idx, int) else node.bound_args.kwargs[input_idx]
                if output_key not in key_map:
                    continue

                for input_key in key_map[output_key].inputs[input_name]:
                    self.parent_ids[input_node][input_key].add((node, output_key))

        for input_node in (*node.bound_args.args, *node.bound_args.kwargs.values()):
            input_config = input_node.model_config or node.model_config
            input_visitor = dataclasses.replace(self, keys_constraint=list(input_config.keys())) if input_config is not None else self
            input_node.accept(input_visitor)


@dataclasses.dataclass
class CreateMergeMethodContextVisitor(RecipeVisitor):
    parent_ids: Dict[RecipeNode, Dict[str, Set[Tuple[RecipeNode, str]]]] = dataclasses.field(default_factory=dict)

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

        locks = defaultdict(lambda: threading.Lock())
        for output_key in self.parent_ids[node].keys():
            if output_key not in key_map:
                continue

            output_key_group = key_map[output_key].outputs
            lock = threading.Lock()
            for output_key in output_key_group:
                locks[output_key] = lock

        res[node] = MergeMethodContext(
            {
                output_key: MergeMethodOutputRef(output_parent_ids, None, locks[output_key])
                for output_key, output_parent_ids in self.parent_ids[node].items()
                if len(output_parent_ids) >= 2 or (output_key in key_map and len(key_map[output_key].outputs) >= 2)
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
