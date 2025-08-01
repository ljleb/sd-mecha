import dataclasses
import typing
import torch
from typing import List, Tuple, Dict, Iterable, Any, Set
from sd_mecha import StateDictKeyError
from sd_mecha.extensions.merge_methods import MergeMethod, StateDict, T as MergeMethodT, merge_method, Parameter, Return
from sd_mecha.extensions.model_configs import ModelConfig
from sd_mecha.recipe_nodes import RecipeVisitor, MergeRecipeNode, ModelRecipeNode, LiteralRecipeNode, \
    NonDictLiteralValue, RecipeNode, ModelDepthRecipeVisitor
from sd_mecha.typing_ import is_subclass


@dataclasses.dataclass
class ForwardingStateDict(StateDict):
    state_dict: Dict[str, Any]
    model_config_: ModelConfig
    expected_type: type = None

    @property
    def model_config(self) -> ModelConfig:
        return self.model_config_

    def keys(self) -> Iterable[str]:
        return self.model_config_.keys()

    def __getitem__(self, key):
        aliases = self.model_config.aliases().get(key, [])
        keys = [key, *aliases]

        error = None
        for i, alias_key in enumerate(keys):
            try:
                return cast_node_value(self.state_dict[alias_key], self.expected_type)
            except KeyError as error:
                pass

        raise StateDictKeyError(key) from error

    def __len__(self):
        return len(self.model_config_.keys())

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item):
        return item in self.keys()


@dataclasses.dataclass
class LazyMergeStateDict(StateDict):
    merge_method: MergeMethod
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    mm_cache: Dict
    model_config_: ModelConfig
    expected_type: type = None
    merged_keys: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def model_config(self) -> ModelConfig:
        return self.model_config_

    def keys(self) -> Iterable[str]:
        return self.model_config.keys()

    def __getitem__(self, key: str) -> Any:
        if key not in self.merged_keys:
            result = self.merge_method.merge_key(self.args, self.kwargs, key, self.mm_cache)
            if result is None:
                raise StateDictKeyError(key)

            if isinstance(result, dict):
                self.merged_keys.update(result)
            else:
                self.merged_keys[key] = result

        return cast_node_value(self.merged_keys[key], self.expected_type)

    def __len__(self):
        return len(self.model_config.keys())

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item):
        return item in self.keys()


StateDictImpl = ForwardingStateDict | LazyMergeStateDict


# placeholder merge method that represents the output of the merge graph
@merge_method(register=False)
def executor_root_mm(a: Parameter(MergeMethodT)) -> Return(MergeMethodT):
    return a


@dataclasses.dataclass
class CreateRecipeExecutorVisitor(RecipeVisitor):
    sorted_recipe_nodes: List[MergeRecipeNode]
    sorted_model_configs: List[ModelConfig]
    output_edges: Dict[int, Dict[int, List[str | int]]] = dataclasses.field(default_factory=dict)
    key_whitelist: Dict[int, Set[str]] = dataclasses.field(default_factory=dict)
    init_args: Dict[int, Dict[int, StateDictImpl]] = dataclasses.field(default_factory=dict)
    init_kwargs: Dict[int, Dict[str, StateDictImpl]] = dataclasses.field(default_factory=dict)
    caches: Dict = dataclasses.field(default_factory=dict)
    parent_idx_port: Tuple[int, str | int] = (0, 0)

    def __post_init__(self):
        assert len(self.sorted_recipe_nodes) == len(self.sorted_model_configs)
        assert self.sorted_recipe_nodes[-1].merge_method is executor_root_mm

    def visit_literal(self, node: LiteralRecipeNode):
        state_dict = node.value
        if isinstance(state_dict, dict):
            if state_dict and isinstance(next(iter(state_dict.values())), RecipeNode):
                # substitute this literal node with each underlying node restricted to their respective key(s)
                for k, v in state_dict.items():
                    v.accept(self)
                    try:
                        v_idx = self.sorted_recipe_nodes.index(v)
                        self.key_whitelist.setdefault(v_idx, set()).add(k)
                    except ValueError:
                        pass
                return
        elif isinstance(state_dict, NonDictLiteralValue):
            state_dict = EveryKeyDict(state_dict)
        elif isinstance(state_dict, RecipeNode):
            # substitute this literal node with the underlying node
            state_dict.accept(self)
            return
        else:
            raise TypeError(f"Unexpected literal node value of type {type(node.value)}")

        parent_idx, parent_port = self.parent_idx_port
        if isinstance(parent_port, int):
            init_dict = self.init_args
        else:  # isinstance(port, str)
            init_dict = self.init_kwargs
        init_dict.setdefault(parent_idx, {}).setdefault(parent_port, ForwardingStateDict(state_dict, node.model_config))

    def visit_model(self, node: ModelRecipeNode):
        parent_idx, parent_port = self.parent_idx_port
        if isinstance(parent_port, int):
            init_dict = self.init_args
        else:  # isinstance(port, str)
            init_dict = self.init_kwargs
        init_dict.setdefault(parent_idx, {}).setdefault(parent_port, ForwardingStateDict(node.state_dict, node.model_config))

    def visit_merge(self, node: MergeRecipeNode):
        node_idx = self.sorted_recipe_nodes.index(node)
        parent_idx, parent_port = self.parent_idx_port
        self.output_edges.setdefault(node_idx, {}).setdefault(parent_idx, []).append(parent_port)

        def depth_of_value(index) -> int:
            nodes = node.args if isinstance(index, int) else node.kwargs
            return nodes[index].accept(ModelDepthRecipeVisitor())

        indices = (*range(len(node.args)), *node.kwargs.items())
        for port in sorted(indices, key=depth_of_value, reverse=True):
            args = node.args if isinstance(port, int) else node.kwargs
            nested_self = dataclasses.replace(self, parent_idx_port=(node_idx, port))
            args[port].accept(nested_self)

        if node.cache is not None:
            self.caches[node_idx] = node.cache


@dataclasses.dataclass
class InsertMergeNodesVisitor(RecipeVisitor):
    sorted_recipe_nodes: List[MergeRecipeNode]
    sorted_model_configs: List[ModelConfig]

    def visit_literal(self, node: LiteralRecipeNode):
        if isinstance(node.value, dict):
            if node.value and isinstance(next(iter(node.value.values())), RecipeNode):
                for v in node.value.values():
                    v.accept(self)

    def visit_model(self, node: ModelRecipeNode):
        return

    def visit_merge(self, node: MergeRecipeNode):
        if node in self.sorted_recipe_nodes:
            return

        self.sorted_recipe_nodes.insert(0, node)

        model_config = node.model_config
        if model_config is None:
            raise RuntimeError(f"unable to infer the target model config for merge method '{node.merge_method.identifier}'")
        self.sorted_model_configs.insert(0, model_config)

        for k, v in (*enumerate(node.args), node.kwargs.items()):
            v.accept(self)


@dataclasses.dataclass
class EveryKeyDict:
    value: Any

    def __getitem__(self, item):
        return self.value


@dataclasses.dataclass
class RecipeExecutor:
    sorted_merge_methods: List[MergeMethod]
    model_configs: List[ModelConfig]
    output_edges: Dict[int, Dict[int, Iterable[str | int]]]  # mm_id -> {target_mm_id: [port, port, port, ...]}
    key_whitelist: Dict[int, Set[str]]
    input_args: Dict[int, Dict[int, StateDictImpl]]
    input_kwargs: Dict[int, Dict[str, StateDictImpl]]
    caches: Dict[int, Dict]

    def merge_key(self, key: str) -> Any:
        result = None

        for mm_idx, mm in enumerate(self.sorted_merge_methods):
            key_whitelist = self.key_whitelist.get(mm_idx)
            if key_whitelist is not None and key not in key_whitelist:
                continue

            sd_args = self.input_args.pop(mm_idx, {})
            input_types = mm.get_input_types().as_dict(len(sd_args))
            args = tuple(
                evaluate_state_dict_input(sd_args[port], key, input_types[port])
                if port in sd_args else raise_expr(RuntimeError(
                    f"node '{mm.identifier}' executed before its positional argument {port} (0-based position)"
                ))
                for port in range(len(sd_args))
            )
            kwargs = {
                port: evaluate_state_dict_input(sd, key, input_types[port])
                for port, sd in self.input_kwargs.pop(mm_idx, {}).items()
            }
            cache = self.caches.get(mm_idx)

            result = LazyMergeStateDict(mm, args, kwargs, cache, self.model_configs[mm_idx])

            output_edges = self.output_edges.get(mm_idx)
            if output_edges is None:
                continue

            for target_mm_idx, ports in output_edges.items():
                for port in ports:
                    if isinstance(port, int):
                        self.input_args.setdefault(target_mm_idx, {})[port] = result
                    else:  # isinstance(port, str)
                        self.input_kwargs.setdefault(target_mm_idx, {})[port] = result

        if result is not None:
            return result[key]


def raise_expr(exc: Exception) -> None:
    raise exc


def create_recipe_executor(
    recipe: RecipeNode,
) -> RecipeExecutor:
    recipe = executor_root_mm(recipe)
    sorted_recipe_nodes = []
    sorted_model_configs = []
    recipe.accept(InsertMergeNodesVisitor(sorted_recipe_nodes, sorted_model_configs))

    visitor = CreateRecipeExecutorVisitor(sorted_recipe_nodes, sorted_model_configs)
    recipe.accept(visitor)

    sorted_merge_methods = [merge_node.merge_method for merge_node in visitor.sorted_recipe_nodes]
    executor = RecipeExecutor(
        sorted_merge_methods,
        visitor.sorted_model_configs,
        visitor.output_edges,
        visitor.key_whitelist,
        visitor.init_args,
        visitor.init_kwargs,
        visitor.caches,
    )
    return executor


def evaluate_state_dict_input(
    sd: StateDictImpl,
    key: str,
    expected_type: type,
):
    if is_subclass(expected_type, StateDict):
        return dataclasses.replace(sd, expected_type=expected_type)
    else:
        return cast_node_value(sd[key], expected_type)


def cast_node_value(value, expected_type):
    assert expected_type is not None, "expected type should never be None"
    if value is None:
        return value

    try:
        if issubclass(typing.get_origin(expected_type) or expected_type, StateDict):
            expected_type = (typing.get_args(expected_type) + (MergeMethodT,))[0]
    except TypeError:
        pass

    if isinstance(expected_type, typing.TypeVar) or isinstance(value, expected_type):
        return value
    if issubclass(expected_type, str):
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
