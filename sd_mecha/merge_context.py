import contextlib
import dataclasses
import threading
from sd_mecha.extensions.model_configs import ModelConfig
from sd_mecha.recipe_nodes import LiteralRecipeNode, MergeRecipeNode, ModelRecipeNode, RecipeNode, RecipeVisitor
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple, Union


# todo: test this function
def create_merge_method_context(recipe: RecipeNode, root_keys: Iterable[str]) -> Dict[RecipeNode, "MergeMethodContext"]:
    ref_ids_visitor = GetOutputRefIdsVisitor(keys_constraint=list(root_keys))
    recipe.accept(ref_ids_visitor)
    nodes_ref_ids = ref_ids_visitor.nodes_ref_ids
    create_context_visitor = CreateMergeMethodContextVisitor(recipe.model_config, nodes_ref_ids)
    return recipe.accept(create_context_visitor)


@dataclasses.dataclass
class CreateMergeMethodContextVisitor(RecipeVisitor):
    parent_config: ModelConfig
    nodes_ref_ids: Dict[RecipeNode, Dict[str, Set[Tuple[RecipeNode, str]]]] = dataclasses.field(default_factory=dict)

    def visit_literal(self, node: LiteralRecipeNode):
        res = {}
        if isinstance(node.value, dict) and isinstance(next(iter(node.value)), RecipeNode):
            for nested_node in node.value.values():
                nested_node_config = nested_node.model_config or self.parent_config
                res |= nested_node.accept(dataclasses.replace(self, parent_config=nested_node_config))
        return res

    def visit_model(self, node: ModelRecipeNode):
        return {}

    def visit_merge(self, node: MergeRecipeNode):
        res = {}
        for arg_node in (*node.args, *node.kwargs.values()):
            arg_node_config = arg_node.model_config or self.parent_config
            res |= arg_node.accept(dataclasses.replace(self, parent_config=arg_node_config))

        locks = {}
        groups_by_key = {}
        node_config = node.model_config or self.parent_config
        for output_key_group in node.merge_method.get_output_key_groups(node_config):
            lock = threading.Lock()
            for output_key in output_key_group:
                groups_by_key[output_key] = output_key_group
                locks[output_key] = lock

        res[node] = MergeMethodContext(
            {
                output_key: MergeMethodOutputRef(ref_ids, None, locks[output_key])
                for output_key, ref_ids in self.nodes_ref_ids.get(node, {}).items()
                if len(ref_ids) >= 2 or len(groups_by_key[output_key]) >= 2
            },
            node.merge_method.instantiate(),
        )

        return res


@dataclasses.dataclass
class GetOutputRefIdsVisitor(RecipeVisitor):
    nodes_ref_ids: Dict[RecipeNode, Dict[str, Set[Tuple[RecipeNode, str]]]] = dataclasses.field(default_factory=dict)
    keys_constraint: Sequence[str] = dataclasses.field(default_factory=list)

    def visit_literal(self, node: LiteralRecipeNode):
        if isinstance(node.value, dict) and isinstance(next(iter(node.value)), RecipeNode):
            keys_constraints = {}
            for key, nested_node in node.value.items():
                if key in self.keys_constraint:
                    keys_constraints.setdefault(nested_node, []).append(key)
            for nested_node, keys_constraint in keys_constraints.items():
                nested_node.accept(dataclasses.replace(self, keys_constraint=keys_constraint))

    def visit_model(self, node: ModelRecipeNode):
        pass

    def visit_merge(self, node: MergeRecipeNode):
        param_names = node.merge_method.get_param_names()
        output_config = node.model_config
        for output_key in self.keys_constraint:
            for arg_idx, arg_name in param_names.as_dict(len(node.args)).items():
                arg_node = node.args[arg_idx] if isinstance(arg_idx, int) else node.kwargs[arg_idx]
                if arg_node not in self.nodes_ref_ids:
                    self.nodes_ref_ids[arg_node] = {}

                arg_config = arg_node.model_config or node.model_config
                for key_read in node.merge_method.get_key_reads(arg_name, output_key):
                    assert arg_config is None or key_read in arg_config.keys() or output_config.keys()[output_key].optional, (
                        f"The configuration of merge method {node.merge_method.identifier} is invalid.\n"
                        f"The input key '{key_read}' does not exist for parameter {arg_name}. (the effective config is {arg_config.identifier})\n"
                        f"The output key that causes this problem is '{output_key}'. (the effective output config is {output_config.identifier})"
                    )
                    arg_ref_ids = self.nodes_ref_ids[arg_node]
                    arg_ref_ids.setdefault(key_read, set())
                    arg_ref_ids[key_read].add((node, output_key))

        for arg_node in (*node.args, *node.kwargs.values()):
            arg_config = arg_node.model_config or node.model_config
            if arg_config is not None:
                keys = list(arg_config.keys())
            else:
                keys = self.keys_constraint
            arg_node.accept(dataclasses.replace(self, keys_constraint=keys))


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
            yield output_ref


@dataclasses.dataclass
class MergeMethodOutputRef:
    ref_ids: Set[Tuple[RecipeNode, str]]
    cache: Any
    lock: Union[threading.Lock, contextlib.AbstractContextManager]
    locked: bool = False

    def use_once(self, ref_id: Optional[Tuple[RecipeNode, str]]) -> Any:
        assert self.locked
        res = self.cache
        self.ref_ids.difference_update((ref_id,))
        if self.was_freed():
            self.cache = None
            self.lock = contextlib.nullcontext()
        return res

    def held_by(self, ref_id: Tuple[RecipeNode, str]) -> bool:
        assert self.locked
        return ref_id in self.ref_ids

    def set_cache(self, cache: Any):
        assert self.locked
        if self.was_freed():
            return
        assert self.cache is None, f"cache was already set: {self.cache}, trying to assign {cache}"
        self.cache = cache

    def was_freed(self) -> bool:
        return len(self.ref_ids) <= 0

    def __enter__(self):
        self.lock.__enter__()
        self.locked = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.locked = False
        self.lock.__exit__(exc_type, exc_val, exc_tb)
