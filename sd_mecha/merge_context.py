import contextlib
import dataclasses
import threading
from collections import defaultdict
from sd_mecha.recipe_nodes import LiteralRecipeNode, MergeRecipeNode, ModelRecipeNode, RecipeNode, RecipeVisitor
from typing import Any, Dict, Mapping, Optional, Set, Tuple


def create_merge_method_context(recipe: RecipeNode, active_keys: Mapping[RecipeNode, Set[str]], enable_outputs_reuse: bool) -> Dict[RecipeNode, "MergeMethodContext"]:
    ports_visitor = GetOutputPortsVisitor(active_keys)
    ports_visitor.visit_root(recipe)
    parent_ports = ports_visitor.parent_ports
    create_context_visitor = CreateMergeMethodContextVisitor(active_keys, parent_ports, enable_outputs_reuse)
    res = recipe.accept(create_context_visitor)
    return res


@dataclasses.dataclass
class GetOutputPortsVisitor(RecipeVisitor):
    active_keys: Mapping[RecipeNode, Set[str]]
    parent_ports: Dict[RecipeNode, Dict[str, Set[Optional[Tuple[RecipeNode, str]]]]] = dataclasses.field(default_factory=lambda: defaultdict(lambda: defaultdict(set)))

    def visit_root(self, root: RecipeNode):
        for k in self.active_keys[root]:
            self.parent_ports[root][k].add(None)
        root.accept(self)

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
            if output_key not in key_map:
                continue

            for input_idx, param_name in param_names.items():
                child_node = node.bound_args.args[input_idx] if isinstance(input_idx, int) else node.bound_args.kwargs[input_idx]

                child_output_keys = self.active_keys[child_node].intersection(key_map[output_key].inputs[param_name])
                for child_output_key in child_output_keys:
                    self.parent_ports[child_node][child_output_key].add((node, output_key))

        for input_node in (*node.bound_args.args, *node.bound_args.kwargs.values()):
            input_node.accept(self)


@dataclasses.dataclass
class CreateMergeMethodContextVisitor(RecipeVisitor):
    active_keys: Mapping[RecipeNode, Set[str]]
    parent_ports: Mapping[RecipeNode, Mapping[str, Set[Optional[Tuple[RecipeNode, str]]]]]
    enable_outputs_reuse: bool

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

        if not node.merge_method.reuse_outputs:
            res[node] = MergeMethodContext({}, node.merge_method.instantiate(), reused_output_keys=self.active_keys[node])
            return res

        key_map = node.key_map()

        node_parent_ports = self.parent_ports[node]
        locks_map = defaultdict(lambda: defaultdict(threading.Lock if self.enable_outputs_reuse else contextlib.nullcontext))

        refs = {
            output_key: MergeMethodOutputRef(parent_ports, None, locks_map[node][key_map[output_key].outputs])
            for output_key, parent_ports in node_parent_ports.items()
            if (
                len(parent_ports) > 1 or
                len(parent_ports) > 0 and len(key_map[output_key].outputs) > 1
            )
        }
        res[node] = MergeMethodContext(
            refs if self.enable_outputs_reuse else {},
            node.merge_method.instantiate(),
            self.active_keys[node] if self.enable_outputs_reuse else (self.active_keys[node] - set(refs)),
        )

        return res


@dataclasses.dataclass
class MergeMethodContext:
    output_refs: Dict[str, "MergeMethodOutputRef"]
    instance: Any
    reused_output_keys: Set[str]

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
    remaining_ports: Set[Optional[Tuple[RecipeNode, str]]]
    cache: Any
    lock: contextlib.AbstractContextManager
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
