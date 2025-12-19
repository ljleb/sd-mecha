import contextlib
import dataclasses
import threading
from collections import defaultdict

from sd_mecha.extensions.model_configs import ModelConfig
from sd_mecha.recipe_nodes import LiteralRecipeNode, MergeRecipeNode, ModelRecipeNode, RecipeNode, RecipeVisitor
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple, Union


# todo: test this function
def create_merge_method_context(recipe: RecipeNode, root_keys: Iterable[str]) -> Dict[RecipeNode, "MergeMethodContext"]:
    ports_visitor = GetOutputPortsVisitor(keys_constraint=list(root_keys))
    recipe.accept(ports_visitor)
    create_context_visitor = CreateMergeMethodContextVisitor(recipe.model_config, ports_visitor.inputs_ports)
    res = recipe.accept(create_context_visitor)
    return res


@dataclasses.dataclass
class GetOutputPortsVisitor(RecipeVisitor):
    inputs_ports: Dict[RecipeNode, Dict[str, Set[Tuple[RecipeNode, str]]]] = dataclasses.field(default_factory=lambda: defaultdict(lambda: defaultdict(set)))
    keys_constraint: Sequence[str] = dataclasses.field(default_factory=list)

    def visit_literal(self, node: LiteralRecipeNode):
        if isinstance(node.value, dict) and node.value and isinstance(next(iter(node.value.values())), RecipeNode):
            keys_constraints = defaultdict(list)
            for key, nested_node in node.value.items():
                if key in self.keys_constraint:
                    keys_constraints[nested_node].append(key)
            for nested_node, keys_constraint in keys_constraints.items():
                nested_node.accept(dataclasses.replace(self, keys_constraint=keys_constraint))

    def visit_model(self, node: ModelRecipeNode):
        pass

    def visit_merge(self, node: MergeRecipeNode):
        param_names = node.merge_method.get_param_names()
        output_config = node.model_config
        for input_idx, input_name in param_names.as_dict(len(node.args)).items():
            input_node = node.args[input_idx] if isinstance(input_idx, int) else node.kwargs[input_idx]
            for output_key in self.keys_constraint:

                input_config = input_node.model_config or node.model_config
                for input_key in node.merge_method.input_keys_for_output(input_name, output_key):
                    assert (input_config is None or input_key in input_config.keys()) or (output_config is None or output_config.keys()[output_key].optional), (
                        f"The configuration of merge method {node.merge_method.identifier} is invalid.\n"
                        f"The input key '{input_key}' does not exist for parameter {input_name}. (the effective config is {input_config.identifier})\n"
                        f"The output key that causes this problem is '{output_key}'. (the effective output config is {output_config.identifier})"
                    )
                    self.inputs_ports[input_node][input_key].add((node, output_key))

        for input_node in (*node.args, *node.kwargs.values()):
            input_config = input_node.model_config or node.model_config
            if input_config is not None:
                keys = list(input_config.keys())
            else:
                keys = self.keys_constraint
            input_node.accept(dataclasses.replace(self, keys_constraint=keys))


@dataclasses.dataclass
class CreateMergeMethodContextVisitor(RecipeVisitor):
    parent_config: Optional[ModelConfig]
    inputs_ports: Dict[RecipeNode, Dict[str, Set[Tuple[RecipeNode, str]]]] = dataclasses.field(default_factory=dict)

    def visit_literal(self, node: LiteralRecipeNode):
        res = {}
        if isinstance(node.value, dict) and node.value and isinstance(next(iter(node.value.values())), RecipeNode):
            for nested_node in node.value.values():
                nested_node_config = nested_node.model_config or self.parent_config
                res |= nested_node.accept(dataclasses.replace(self, parent_config=nested_node_config))
        return res

    def visit_model(self, node: ModelRecipeNode):
        return {}

    def visit_merge(self, node: MergeRecipeNode):
        res = {}
        for input_node in (*node.args, *node.kwargs.values()):
            input_node_config = input_node.model_config or self.parent_config
            res |= input_node.accept(dataclasses.replace(self, parent_config=input_node_config))

        locks = {}
        groups_by_key = {}
        node_config = node.model_config or self.parent_config
        for output_key_group in node.merge_method.output_groups(node_config):
            lock = threading.Lock()
            for output_key in output_key_group:
                groups_by_key[output_key] = output_key_group
                locks[output_key] = lock

        res[node] = MergeMethodContext(
            {
                output_key: MergeMethodOutputRef(input_ports, None, locks[output_key])
                for output_key, input_ports in self.inputs_ports.get(node, {}).items()
                if len(input_ports) >= 2 or len(groups_by_key.get(output_key, ())) >= 2
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

    def held_by(self, port: Tuple[RecipeNode, str]) -> bool:
        assert self.locked
        return port in self.remaining_ports

    def set_cache(self, cache: Any):
        assert self.locked
        if self.was_freed():
            return
        assert self.cache is None, f"cache was already set: {self.cache}, trying to assign {cache}"
        self.cache = cache

    def was_freed(self) -> bool:
        return len(self.remaining_ports) <= 0

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
