import contextlib
import dataclasses
import threading
from collections import defaultdict
from sd_mecha.keys_map import ActiveKeyMap, RealizedKeyRelation
from sd_mecha.recipe_nodes import LiteralRecipeNode, MergeRecipeNode, ModelRecipeNode, RecipeNode, RecipeVisitor
from typing import Any, Dict, Mapping, Optional, Set, Tuple


def create_merge_method_context(
    recipe: RecipeNode,
    realized_key_maps: Mapping[RecipeNode, ActiveKeyMap[RealizedKeyRelation]],
    active_keys: Mapping[RecipeNode, Set[str]],
    enable_outputs_reuse: bool,
) -> Dict[RecipeNode, "MergeMethodContext"]:
    ports_visitor = GetOutputPortsVisitor(realized_key_maps, active_keys)
    ports_visitor.visit_root(recipe)
    parent_ports = ports_visitor.parent_ports
    create_context_visitor = CreateMergeMethodContextVisitor(realized_key_maps, active_keys, parent_ports, enable_outputs_reuse)
    return recipe.accept(create_context_visitor)


Port = Optional[Tuple[RecipeNode, str, int | str]]


@dataclasses.dataclass
class GetOutputPortsVisitor(RecipeVisitor):
    realized_key_maps: Mapping[RecipeNode, ActiveKeyMap[RealizedKeyRelation]]
    active_keys: Mapping[RecipeNode, Set[str]]
    parent_ports: Dict[RecipeNode, Dict[str, Set[Port]]] = dataclasses.field(default_factory=lambda: defaultdict(lambda: defaultdict(set)))

    def visit_root(self, root: RecipeNode):
        for k in self._node_active_keys(root):
            self.parent_ports[root][k].add(None)
        root.accept(self)

    def visit_literal(self, node: LiteralRecipeNode):
        for child_node in set(node.value_dict.values()):
            if isinstance(child_node, RecipeNode):
                child_node.accept(self)

    def visit_model(self, node: ModelRecipeNode):
        pass

    def visit_merge(self, node: MergeRecipeNode):
        key_map = self.realized_key_maps.get(node)
        if key_map is None:
            return

        param_names = node.merge_method.get_param_names().as_dict(len(node.bound_args.args))
        key_map = self.realized_key_maps[node]

        for output_key, relation in key_map.simple_map.items():
            for input_idx, param_name in param_names.items():
                child_node = node.bound_args.args[input_idx] if isinstance(input_idx, int) else node.bound_args.kwargs[input_idx]
                child_output_keys = self._node_active_keys(child_node).intersection(relation.inputs.get(param_name, ()))
                for child_output_key in child_output_keys:
                    self.parent_ports[child_node][child_output_key].add((node, output_key, input_idx))

        for input_node in (*node.bound_args.args, *node.bound_args.kwargs.values()):
            input_node.accept(self)

    def _node_active_keys(self, node: RecipeNode) -> Set[str]:
        return self.active_keys.get(node, set())


@dataclasses.dataclass
class CreateMergeMethodContextVisitor(RecipeVisitor):
    realized_key_maps: Mapping[RecipeNode, ActiveKeyMap[RealizedKeyRelation]]
    active_keys: Mapping[RecipeNode, Set[str]]
    parent_ports: Mapping[RecipeNode, Mapping[str, Set[Port]]]
    enable_outputs_reuse: bool

    def visit_literal(self, node: LiteralRecipeNode) -> Dict[RecipeNode, "MergeMethodContext"]:
        res = {}
        for nested_node in node.value_dict.values():
            if isinstance(nested_node, RecipeNode):
                res |= nested_node.accept(self)
        return res

    def visit_model(self, node: ModelRecipeNode) -> Dict[RecipeNode, "MergeMethodContext"]:
        return {}

    def visit_merge(self, node: MergeRecipeNode) -> Dict[RecipeNode, "MergeMethodContext"]:
        res = {}
        for input_node in (*node.bound_args.args, *node.bound_args.kwargs.values()):
            res |= input_node.accept(self)

        node_active_keys = self._node_active_keys(node)
        if not node_active_keys:
            return res

        if not node.merge_method.reuse_outputs:
            res[node] = MergeMethodContext({}, node.merge_method.instantiate(), node_active_keys)
            return res

        key_map = self.realized_key_maps[node]
        node_parent_ports = self.parent_ports[node]

        lock_factory = threading.Lock if self.enable_outputs_reuse else contextlib.nullcontext
        output_refs: Dict[str, MergeMethodOutputRef] = {}

        relation_to_ports: Dict[tuple, Tuple[RealizedKeyRelation, Set[Port]]] = {}
        for output_key, parent_ports in node_parent_ports.items():
            relation = key_map[output_key]
            relation_key = relation.outputs

            existing = relation_to_ports.get(relation_key)
            if existing is None:
                relation_to_ports[relation_key] = (relation, set(parent_ports))
            else:
                existing_relation, existing_ports = existing
                existing_ports.update(parent_ports)

        for relation, all_ports in relation_to_ports.values():
            needs_ref = len(all_ports) > 1 or (len(all_ports) > 0 and len(relation.outputs) > 1)
            if not needs_ref:
                continue

            lock = lock_factory()
            for sibling_key in relation.outputs:
                if sibling_key not in node_active_keys:
                    continue

                sibling_ports = set(node_parent_ports.get(sibling_key, ()))
                output_refs[sibling_key] = MergeMethodOutputRef(
                    remaining_ports=sibling_ports,
                    cache=None,
                    lock=lock,
                )

        res[node] = MergeMethodContext(
            output_refs if self.enable_outputs_reuse else {},
            node.merge_method.instantiate(),
            node_active_keys if self.enable_outputs_reuse else (node_active_keys - set(output_refs)),
        )
        return res

    def _node_active_keys(self, node: RecipeNode) -> Set[str]:
        return self.active_keys.get(node, set())


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
    remaining_ports: Set[Port]
    cache: Any
    lock: contextlib.AbstractContextManager
    locked: bool = False

    def use_once(self, port: Port) -> Any:
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
