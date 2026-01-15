import abc
import contextlib
import dataclasses
import logging
import pathlib
import torch
from collections import deque, OrderedDict
from typing import ContextManager, Dict, Generic, Iterable, List, Mapping, Optional, Set, Tuple, TypeVar
from sd_mecha.extensions import merge_spaces, model_configs, model_dirs, model_formats
from sd_mecha.extensions.merge_methods import value_to_node
from sd_mecha.extensions.merge_spaces import MergeSpace, MergeSpaceSymbol
from sd_mecha.extensions.model_configs import ModelConfig, StructuralModelConfig
from sd_mecha.recipe_nodes import ClosedModelRecipeNode, LiteralRecipeNode, MergeRecipeNode, ModelRecipeNode, OpenModelRecipeNode, RecipeNode, RecipeNodeOrValue, RecipeVisitor
from sd_mecha.streaming import SafetensorsMapping, TensorMetadata


T = TypeVar("T")


@contextlib.contextmanager
def open_graph(
    root: RecipeNodeOrValue,
    buffer_size_per_dict: int = 0,
    check_extra_keys: bool = False,
    check_mandatory_keys: bool = False,
    root_only: bool = False,
    solve_model_config: bool = True,
    solve_merge_space: bool = True,
    model_config: Optional[str | ModelConfig] = None,
    merge_space: Optional[str | MergeSpace] = None,
    model_config_preference: Optional[Iterable[str | ModelConfig]] = None,
    merge_space_preference: Optional[Iterable[str | MergeSpace]] = None,
) -> ContextManager[RecipeNode]:
    model_config = model_configs.resolve(model_config) if isinstance(model_config, str) else model_config
    merge_space = merge_spaces.resolve(merge_space) if isinstance(merge_space, str) else merge_space
    if model_config_preference is not None:
        model_config_preference = [
            model_configs.resolve(mc) if isinstance(mc, str) else mc
            for mc in model_config_preference
        ]
    if merge_space_preference is not None:
        merge_space_preference = [
            merge_spaces.resolve(ms) if isinstance(ms, str) else ms
            for ms in merge_space_preference
        ]

    root, dicts_cache = _open_graph_impl(
        root=root,
        buffer_size_per_dict=buffer_size_per_dict,
        check_extra_keys=check_extra_keys,
        check_mandatory_keys=check_mandatory_keys,
        root_only=root_only,
        solve_model_config=solve_model_config,
        solve_merge_space=solve_merge_space,
        model_config=model_config,
        merge_space=merge_space,
        model_config_preference=model_config_preference,
        merge_space_preference=merge_space_preference,
    )

    try:
        yield root
    finally:
        for v in dicts_cache.values():
            v.close()


def _open_graph_impl(
    *,
    root: RecipeNodeOrValue,
    buffer_size_per_dict: int,
    check_extra_keys: bool,
    check_mandatory_keys: bool,
    root_only: bool,
    solve_model_config: bool,
    solve_merge_space: bool,
    model_config: Optional[str | ModelConfig],
    merge_space: Optional[str | MergeSpace],
    model_config_preference: Optional[Iterable[str | ModelConfig]],
    merge_space_preference: Optional[Iterable[str | MergeSpace]],
) -> Tuple[RecipeNode, Dict[pathlib.Path, SafetensorsMapping]]:
    component_types = {}
    if solve_model_config:
        component_types[ModelConfigComponentType] = (model_config, model_config_preference)
    if solve_merge_space:
        component_types[MergeSpaceComponentType] = (merge_space, merge_space_preference)

    root = value_to_node(root).accept(ResolvePathsVisitor())

    node_to_indices, processed_nodes = discover_components(root, root_only, set(component_types))
    active_nodes = processed_nodes if root_only else set(node_to_indices.keys())

    opener = OpenActiveStateDictsVisitor(buffer_size_per_dict, active_nodes)
    root = opener.process(root)

    candidates: Dict[type[ComponentType], Dict[ComponentIndex, ComponentCandidates]] = {}
    for t, (return_hint, return_preference) in component_types.items():
        t_candidates = t.build_candidates(node_to_indices, processed_nodes)
        root_t_candidates = t_candidates[node_to_indices[root].ids[t]]

        if return_hint is not None:
            root_t_candidates.apply_return_hint(return_hint, reason=f"return hint {return_hint}")
        if return_preference is not None:
            root_t_candidates.apply_preference(return_preference)

        candidates[t] = t_candidates

    solutions = {t: {idx: c.finalize() for idx, c in candidates[t].items()} for t in candidates}

    finalizer = FinalizeVisitor(
        node_to_indices=node_to_indices,
        solved_cfg=solutions.get(ModelConfigComponentType),
        solved_ms=solutions.get(MergeSpaceComponentType),
        check_extra_keys=check_extra_keys,
        check_mandatory_keys=check_mandatory_keys,
    )
    root = root.accept(finalizer)

    return root, opener.dicts_cache


@dataclasses.dataclass
class ResolvePathsVisitor(RecipeVisitor):
    def visit_literal(self, node: LiteralRecipeNode):
        value_dict = {
            k: v.accept(self) if isinstance(v, RecipeNode) else v
            for k, v in node.value_dict.items()
        }
        return LiteralRecipeNode(value_dict, node.model_config, node.merge_space)

    def visit_model(self, node: ModelRecipeNode):
        if node.is_open:
            return node

        path = model_dirs.normalize_path(node.path)
        if not path.exists():
            searched = [str(d) for d in model_dirs.get_all()] if not path.is_absolute() else []
            raise FileNotFoundError(
                "Model file not found.\n"
                f"  Requested: {node.path}\n"
                f"  Resolved:  {path}\n"
                + (f"  Searched base dirs: {searched}\n" if searched else "")
                + "Fix: provide an absolute path, or add the correct base directory to model_dirs."
            )

        return ClosedModelRecipeNode(path, node.model_config, node.merge_space)

    def visit_merge(self, node: MergeRecipeNode):
        args = tuple(v.accept(self) for v in node.bound_args.args)
        kwargs = {k: v.accept(self) for k, v in node.bound_args.kwargs.items()}
        bound_args = node.merge_method.get_signature().bind(*args, **kwargs)
        return MergeRecipeNode(node.merge_method, bound_args, node.model_config, node.merge_space)


@dataclasses.dataclass
class OpenActiveStateDictsVisitor(RecipeVisitor):
    buffer_size_per_dict: int
    active_nodes: Set[RecipeNode]

    dicts_cache: Dict[pathlib.Path, SafetensorsMapping] = dataclasses.field(default_factory=dict)
    transform_cache: Dict[RecipeNode, RecipeNode] = dataclasses.field(default_factory=dict)

    def process(self, node: RecipeNode) -> RecipeNode:
        if node not in self.active_nodes:
            return node
        cached = self.transform_cache.get(node)
        if cached is not None:
            return cached
        out = node.accept(self)
        self.transform_cache[node] = out
        return out

    def visit_literal(self, node: LiteralRecipeNode) -> RecipeNode:
        changed = False
        value_dict = {}
        for k, old_v in node.value_dict.items():
            if isinstance(old_v, RecipeNode):
                new_v = self.process(old_v)
                changed |= (new_v is not old_v)
                value_dict[k] = new_v
            else:
                value_dict[k] = old_v

        if not changed:
            return node

        return LiteralRecipeNode(value_dict, node.model_config, node.merge_space)

    def visit_model(self, node: ModelRecipeNode) -> RecipeNode:
        if node.is_open:
            self.dicts_cache[node.path] = node.state_dict
            return node

        sd = self.dicts_cache.get(node.path)
        if sd is None:
            matching_formats = [fmt for fmt in model_formats.get_all() if fmt.matches(node.path)]
            if len(matching_formats) > 1:
                ids = [f.identifier for f in matching_formats]
                raise TypeError(
                    "Model format detection is ambiguous.\n"
                    f"  Model path: {node.path}\n"
                    f"  Matched formats: {ids}\n"
                    "Fix: rename the file to use an unambiguous extension, or restrict enabled model formats."
                )
            if len(matching_formats) < 1:
                known = [f.identifier for f in model_formats.get_all()]
                raise TypeError(
                    "Model format detection failed: no registered format matches this file.\n"
                    f"  Model path: {node.path}\n"
                    f"  Known formats: {known}\n"
                    "Fix: install/enable the format plugin for this file type, or convert the model to a supported format."
                )

            sd = matching_formats[0].get_read_dict(node.path, self.buffer_size_per_dict)
            self.dicts_cache[node.path] = sd

        return OpenModelRecipeNode(sd, node.path, node.model_config, node.merge_space)

    def visit_merge(self, node: MergeRecipeNode) -> RecipeNode:
        changed = False

        new_args = []
        for old_v in node.bound_args.args:
            new_v = self.process(old_v)
            changed |= (new_v is not old_v)
            new_args.append(new_v)

        new_kwargs = {}
        for k, old_v in node.bound_args.kwargs.items():
            new_v = self.process(old_v)
            changed |= (new_v is not old_v)
            new_kwargs[k] = new_v

        if not changed:
            out = node
        else:
            bound_args = node.merge_method.get_signature().bind(*tuple(new_args), **new_kwargs)
            out = MergeRecipeNode(node.merge_method, bound_args, node.model_config, node.merge_space)

        return out


@dataclasses.dataclass
class ComponentCandidates(RecipeVisitor, abc.ABC, Generic[T]):
    def apply_return_hint(self, cfg: T, *, reason: str) -> None:
        ...

    def apply_preference(self, prefs: Iterable[T]) -> None:
        ...

    @abc.abstractmethod
    def finalize(self) -> T:
        ...


class ComponentType(abc.ABC, Generic[T]):
    all_components: Dict[str, type["ComponentType"]] = {}

    name: str
    Candidates: type[ComponentCandidates[T]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.name = kwargs.get("component_name", cls.__name__)
        if cls.name in cls.all_components:
            raise TypeError(f"Component {cls.name} already exists.")

        cls.all_components[cls.name] = cls

    @classmethod
    def build_candidates(
        cls,
        node_to_indices: Dict[RecipeNode, "ComponentIndices"],
        processed_nodes: Set[RecipeNode],
    ) -> Dict["ComponentIndex", ComponentCandidates[T]]:
        components: Set[ComponentIndex] = set()
        for n in processed_nodes:
            idxs = node_to_indices.get(n)
            if idxs is None:
                continue
            components.add(idxs.ids[cls])

        candidates: Dict[ComponentIndex, ComponentCandidates[T]] = {rep: cls.Candidates() for rep in components}
        for n in processed_nodes:
            idxs = node_to_indices.get(n)
            if idxs is None:
                continue
            rep = idxs.ids[cls]
            cc = candidates.get(rep)
            if cc is None:
                continue

            n.accept(
                cc,
                node_to_indices=node_to_indices,
                candidates=candidates,
            )

        return candidates

    @staticmethod
    @abc.abstractmethod
    def literal_delineates(node: LiteralRecipeNode) -> bool:
        ...

    @staticmethod
    @abc.abstractmethod
    def merge_delineate(
        node: MergeRecipeNode,
        parent_component: "ComponentIndex",
    ) -> Dict[RecipeNode, "ComponentIndex"]:
        ...


@dataclasses.dataclass
class ModelConfigCandidates(ComponentCandidates[ModelConfig]):
    stats: Optional[Dict[str, "ConfigMatchStats"]] = None
    requires_known_config: bool = False
    explicit_ids: Set[str] = dataclasses.field(default_factory=set)
    common_keys: Optional[Dict[str, TensorMetadata]] = None

    def apply_return_hint(self, cfg: ModelConfig, *, reason: str) -> None:
        self.requires_known_config = True
        self.explicit_ids.add(cfg.identifier)
        self._constrain_to_id(cfg, reason=reason)
        self._update_with_metadata(cfg.metadata(), cfg)

    def apply_preference(self, prefs: Iterable[ModelConfig]) -> None:
        if not self.stats:
            return
        for mc in prefs:
            if mc.identifier in self.stats:
                self.apply_return_hint(mc, reason=f"return preference {mc.identifier}")
                break

    def _update_common_keys(self, metadata: Mapping[str, TensorMetadata]) -> None:
        if self.common_keys is None:
            self.common_keys = dict(metadata)
            return

        out = {}
        for k, m1 in self.common_keys.items():
            m2 = metadata.get(k)
            if m2 is None:
                continue
            out[k] = m1 if _meta_score(m1) < _meta_score(m2) else m2
        self.common_keys = out

    def _constrain_to_id(self, config: ModelConfig, *, reason: str) -> None:
        if self.stats is None:
            self.stats = {config.identifier: ConfigMatchStats(set(config.keys()))}
        else:
            common_ids = set(self.stats.keys()).intersection({config.identifier})
            if not common_ids and self.requires_known_config:
                raise TypeError(
                    "Incompatible model config constraints: candidate set intersection is empty.\n"
                    f"  Existing: {sorted(self.stats.keys())}\n"
                    f"  New:      {config.identifier}\n"
                    f"  Reason:   {reason}\n"
                )
            self.stats = {cfg_id: self.stats[cfg_id] for cfg_id in common_ids}

    def _update_with_metadata(self, metadata: Mapping[str, TensorMetadata], hint: Optional[ModelConfig]) -> None:
        self._update_common_keys(metadata)

        if hint is not None:
            self.requires_known_config = True

        if is_config_stub(hint):
            new_stats = generate_configs_stats(metadata)
            if self.stats is None:
                self.stats = new_stats
            else:
                common = set(self.stats.keys()).intersection(new_stats.keys())
                if not common and self.requires_known_config:
                    raise TypeError(
                        "Incompatible model config constraints: candidate set intersection is empty.\n"
                        f"  Existing: {sorted(self.stats.keys())}\n"
                        f"  New:      {sorted(new_stats.keys())}\n"
                        "Fix: ensure inputs correspond to a known config, or use the structural model config.\n"
                    )
                self.stats = {cfg_id: self.stats[cfg_id] | new_stats[cfg_id] for cfg_id in common}
        else:
            self.explicit_ids.add(hint.identifier)
            self.requires_known_config = True

            self._constrain_to_id(hint, reason=f"explicit hint {hint.identifier}")

            cfg_stat = create_config_stats(hint, set(metadata))
            if self.stats is None:
                self.stats = {hint.identifier: cfg_stat}
            else:
                prev = self.stats.get(hint.identifier, ConfigMatchStats(set(hint.keys())))
                self.stats[hint.identifier] = prev | cfg_stat

    def visit_literal(self, node: LiteralRecipeNode, **_kwargs):
        metadata: Dict[str, TensorMetadata] = {}
        for k, v in node.value_dict.items():
            if isinstance(v, RecipeNode):
                continue
            if isinstance(v, torch.Tensor):
                metadata[k] = TensorMetadata(v.shape, v.dtype)
            else:
                metadata[k] = TensorMetadata(None, None)

        self._update_with_metadata(metadata, node.model_config)

    def visit_model(self, node: ModelRecipeNode, **_kwargs):
        metadata = OrderedDict(node.state_dict.metadata())
        self._update_with_metadata(metadata, node.model_config)

    def visit_merge(self, node: MergeRecipeNode, node_to_indices: Dict[RecipeNode, "ComponentIndices"], candidates: Dict["ComponentIndex", "ModelConfigCandidates"]):
        return_cfg = node.merge_method.get_signature().return_annotation.data.model_config
        if return_cfg is not None:
            self.requires_known_config = True
            self.explicit_ids.add(return_cfg.identifier)
            self._constrain_to_id(return_cfg, reason=f"merge return {return_cfg.identifier}")
            self._update_with_metadata(return_cfg.metadata(), return_cfg)

        arg_count = len(node.bound_args.args)
        input_cfgs = node.merge_method.get_input_configs().as_dict(arg_count)

        for k, child in (*enumerate(node.bound_args.args), *node.bound_args.kwargs.items()):
            cfg = input_cfgs[k]
            if is_config_stub(cfg):
                continue

            child_candidates = candidates.get(node_to_indices[child].ids[ModelConfigComponentType])
            if child_candidates is None:
                continue

            child_candidates.apply_return_hint(cfg, reason=f"merge input {cfg.identifier}")

    def finalize(self) -> ModelConfig:
        if not self.stats:
            if self.requires_known_config:
                if not self.explicit_ids:
                    sample = sorted((self.common_keys or {}).keys())
                    raise TypeError(
                        "Model config inference failed: INFER was requested but no known config matches.\n"
                        f"  Keys: {sample}\n"
                        "Fix: ensure inputs correspond to a known config, or use the structural model config."
                    )
                if len(self.explicit_ids) == 1:
                    cfg = next(iter(self.explicit_ids))
                    raise TypeError(
                        "Model config constraint violation: the model config did not match the config requirement.\n"
                        f"  Required: {cfg}\n"
                        "Fix: provide checkpoints/inputs matching the requested config, or let config auto-detection determine an appropriate config type."
                    )
                raise TypeError(
                    "Model config constraint violation: incompatible configs were mixed together.\n"
                    f"  Requested configs: {sorted(self.explicit_ids)}\n"
                    "Fix: provide checkpoints/inputs matching the requested config, or let config auto-detection determine an appropriate shared config type."
                )

            keys = self.common_keys if self.common_keys is not None else {}
            return StructuralModelConfig(keys)

        best_id = min(self.stats.items(), key=lambda kv: kv[1].state_dict_misses)[0]
        return model_configs.resolve(best_id)


@dataclasses.dataclass
class MergeSpaceCandidates(ComponentCandidates[MergeSpace]):
    allowed: Optional[Set[MergeSpace]] = None

    def apply_return_hint(self, ms: MergeSpace, *, reason: str) -> None:
        self.constrain({ms}, reason=reason)

    def apply_preference(self, prefs: Iterable[MergeSpace]) -> None:
        for ms in prefs:
            if self.allowed is None or ms in self.allowed:
                self.constrain({ms}, reason=f"return preference {ms.identifier}")
                break

    def constrain(self, constraint: Set[MergeSpace], *, reason: str = "") -> None:
        if self.allowed is None:
            self.allowed = set(constraint)
        else:
            self.allowed.intersection_update(constraint)

        if self.allowed is not None and len(self.allowed) == 0:
            prefix = f"{reason}\n" if reason else ""
            raise TypeError(prefix + "Merge-space constraints are unsatisfiable (empty intersection).")

    def visit_literal(self, node: LiteralRecipeNode, **_kwargs):
        if node.merge_space is not None:
            self.constrain({node.merge_space}, reason="literal merge_space hint")

    def visit_model(self, node: ModelRecipeNode, **_kwargs):
        if node.merge_space is not None:
            self.constrain({node.merge_space}, reason="model merge_space hint")

    def visit_merge(self, node: MergeRecipeNode, node_to_indices: Dict[RecipeNode, "ComponentIndices"], candidates: Dict["ComponentIndex", "MergeSpaceCandidates"]):
        return_ms = node.merge_method.get_signature().return_annotation.data.merge_space
        if return_ms is None:
            return_ms = node.merge_method.default_merge_space

        if isinstance(return_ms, MergeSpaceSymbol):
            self.constrain(return_ms.merge_spaces, reason="merge return symbol constraint")
        else:
            self.constrain({return_ms}, reason="merge return fixed constraint")

        arg_count = len(node.bound_args.args)
        input_ms = node.merge_method.get_input_merge_spaces().as_dict(arg_count)

        for k, child in (*enumerate(node.bound_args.args), *node.bound_args.kwargs.items()):
            param_ms = input_ms[k]
            child_rep = node_to_indices[child].ids[MergeSpaceComponentType]
            child_c = candidates.get(child_rep)
            if child_c is None:
                continue

            if isinstance(param_ms, MergeSpaceSymbol):
                child_c.constrain(set(param_ms.merge_spaces), reason="merge input symbol constraint")
            else:
                assert isinstance(param_ms, set), f"Unexpected merge space type: {type(param_ms)}"
                child_c.constrain(set(param_ms), reason="merge input fixed-set constraint")

    def finalize(self) -> MergeSpace:
        if self.allowed is None:
            all_spaces = tuple(sorted(ms.identifier for ms in merge_spaces.get_all()))
            raise TypeError(
                "Merge-space inference failed: no constraints were provided.\n"
                f"  Known merge spaces: {all_spaces}\n"
                "Fix: provide a merge_space_hint, or use a merge method that specifies/propagates merge spaces."
            )

        if len(self.allowed) == 0:
            raise TypeError("Merge-space constraints are unsatisfiable (empty intersection).")

        if len(self.allowed) > 1:
            opts = tuple(sorted(ms.identifier for ms in self.allowed))
            raise TypeError(
                "Merge-space inference is ambiguous: multiple candidates remain.\n"
                f"  Candidates: {opts}\n"
                "Fix: add a merge_space_hint to disambiguate."
            )

        return next(iter(self.allowed))


class ModelConfigComponentType(ComponentType, component_name="model_config"):
    Candidates = ModelConfigCandidates

    @staticmethod
    def literal_delineates(node: LiteralRecipeNode) -> bool:
        return node.model_config is not None

    @staticmethod
    def merge_delineate(
        node: MergeRecipeNode,
        parent_component: "ComponentIndex",
    ) -> Dict[RecipeNode, "ComponentIndex"]:
        arg_count = len(node.bound_args.args)
        input_configs = node.merge_method.get_input_configs().as_dict(arg_count)

        out: Dict[RecipeNode, ComponentIndex] = {}
        for k, child in (*enumerate(node.bound_args.args), *node.bound_args.kwargs.items()):
            cfg = input_configs[k]
            if is_config_stub(cfg):
                out[child] = parent_component
            else:
                out[child] = ComponentIndex(ModelConfigComponentType)
        return out


class MergeSpaceComponentType(ComponentType, component_name="merge_space"):
    Candidates = MergeSpaceCandidates

    @staticmethod
    def literal_delineates(node: LiteralRecipeNode) -> bool:
        return node.merge_space is not None

    @staticmethod
    def merge_delineate(
        node: MergeRecipeNode,
        parent_component: "ComponentIndex",
    ) -> Dict[RecipeNode, "ComponentIndex"]:
        arg_count = len(node.bound_args.args)
        input_merge_spaces = node.merge_method.get_input_merge_spaces().as_dict(arg_count)

        return_ms = node.merge_method.get_signature().return_annotation.data.merge_space
        if return_ms is None:
            return_ms = node.merge_method.default_merge_space

        symbol_to_idx: Dict[MergeSpaceSymbol, ComponentIndex] = {}
        if isinstance(return_ms, MergeSpaceSymbol):
            symbol_to_idx[return_ms] = parent_component

        out: Dict[RecipeNode, ComponentIndex] = {}
        for k, child in (*enumerate(node.bound_args.args), *node.bound_args.kwargs.items()):
            param_ms = input_merge_spaces[k]
            if isinstance(param_ms, MergeSpaceSymbol):
                idx = symbol_to_idx.get(param_ms)
                if idx is None:
                    idx = ComponentIndex(MergeSpaceComponentType)
                    symbol_to_idx[param_ms] = idx
                out[child] = idx
            else:
                out[child] = ComponentIndex(MergeSpaceComponentType)
        return out


@dataclasses.dataclass(eq=False)
class ComponentIndex:
    component_type: type[ComponentType]
    _parent: Optional["ComponentIndex"] = None
    _rank: int = 0

    def find(self) -> "ComponentIndex":
        if self._parent is None:
            return self
        self._parent = self._parent.find()
        return self._parent

    def __or__(self, other: "ComponentIndex") -> "ComponentIndex":
        a = self.find()
        b = other.find()

        if a is b:
            return a

        if a.component_type is not b.component_type:
            raise TypeError(
                f"component type mismatch: cannot identify {a.component_type.name} with {b.component_type.name}"
            )

        if a._rank < b._rank:
            a, b = b, a
        b._parent = a
        if a._rank == b._rank:
            a._rank += 1
        return a


@dataclasses.dataclass
class ComponentIndices:
    ids: Dict[type[ComponentType], ComponentIndex]

    def copy(self) -> "ComponentIndices":
        return ComponentIndices(self.ids.copy())

    def __or__(self, other):
        self = self.copy()
        for k, v in self.ids.items():
            if k in other.ids:
                self.ids[k] |= other.ids[k]
        return self


def discover_components(
    root: RecipeNode,
    root_only: bool,
    component_types: Optional[Set[type[ComponentType]]],
) -> Tuple[Dict[RecipeNode, ComponentIndices], Set[RecipeNode]]:
    if component_types is None:
        component_types = set(ComponentType.all_components.values())
    if not component_types:
        return {}, set()

    node_to_indices: Dict[RecipeNode, ComponentIndices] = {}
    processed_nodes: Set[RecipeNode] = set()

    nodes_queue = deque((
        (root, ComponentIndices({t: ComponentIndex(t) for t in component_types})),
    ))

    while nodes_queue:
        stack = [nodes_queue.popleft()]
        visitor = DiscoverComponentsVisitor(nodes_queue, stack, node_to_indices, component_types, root_only, processed_nodes)
        while stack:
            visitor.process(*stack.pop())

    out: Dict[RecipeNode, ComponentIndices] = {}
    for n, idxs in node_to_indices.items():
        out[n] = ComponentIndices({t: idxs.ids[t].find() for t in component_types})
    return out, processed_nodes


@dataclasses.dataclass
class DiscoverComponentsVisitor(RecipeVisitor):
    queue: deque[Tuple[RecipeNode, ComponentIndices]]
    stack: List[Tuple[RecipeNode, ComponentIndices]]
    node_to_indices: Dict[RecipeNode, ComponentIndices]
    component_types: Set[type[ComponentType]]
    root_only: bool
    processed_nodes: Set[RecipeNode]
    cur_indices: Optional[ComponentIndices] = None

    def process(self, node: RecipeNode, indices: ComponentIndices) -> None:
        self.cur_indices = self._attach_node(node, indices)
        self.processed_nodes.add(node)
        node.accept(self)

    def visit_literal(self, node: LiteralRecipeNode):
        assert self.cur_indices is not None
        for child in node.value_dict.values():
            if not isinstance(child, RecipeNode):
                continue

            child_indices = self.cur_indices.copy()
            for t in self.component_types:
                if t.literal_delineates(node):
                    child_indices.ids[t] = ComponentIndex(t)

            self._handle_child(self.cur_indices, child, child_indices)

    def visit_model(self, node: ModelRecipeNode):
        assert self.cur_indices is not None

    def visit_merge(self, node: MergeRecipeNode):
        assert self.cur_indices is not None
        per_type_child_indices: Dict[type[ComponentType], Dict[RecipeNode, ComponentIndex]] = {}
        for t in self.component_types:
            per_type_child_indices[t] = t.merge_delineate(node, self.cur_indices.ids[t])

        for child in (*node.bound_args.args, *node.bound_args.kwargs.values()):
            child_indices = self.cur_indices.copy()
            for t in self.component_types:
                child_indices.ids[t] = per_type_child_indices[t][child]

            self._handle_child(self.cur_indices, child, child_indices)

    def _handle_child(self, parent_idx: ComponentIndices, child: RecipeNode, child_indices: ComponentIndices) -> None:
        if self._is_boundary(parent_idx, child_indices):
            if self.root_only:
                attached = self._attach_node(child, child_indices)
                if not self._is_boundary(parent_idx, attached):
                    self.stack.append((child, attached))
            else:
                self.queue.append((child, child_indices))
        else:
            self.stack.append((child, child_indices))

    def _is_boundary(self, p: ComponentIndices, c: ComponentIndices) -> bool:
        for t in self.component_types:
            if p.ids[t].find() is not c.ids[t].find():
                return True
        return False

    def _attach_node(self, node: RecipeNode, indices: ComponentIndices) -> ComponentIndices:
        existing = self.node_to_indices.get(node)
        if existing is not None:
            indices |= existing

        self.node_to_indices[node] = indices
        return indices


def _components_to_solve(
    node_to_indices: Dict[RecipeNode, ComponentIndices],
    processed_nodes: Set[RecipeNode],
    component_type: type[ComponentType],
) -> Set[ComponentIndex]:
    components: Set[ComponentIndex] = set()
    for n in processed_nodes:
        idxs = node_to_indices.get(n)
        if idxs is None:
            continue
        components.add(idxs.ids[component_type])
    return components


def _nodes_in_components(
    node_to_indices: Dict[RecipeNode, ComponentIndices],
    component_type: type[ComponentType],
    ids_to_solve: Set[ComponentIndex],
) -> Dict[ComponentIndex, Set[RecipeNode]]:
    component_nodes: Dict[ComponentIndex, Set[RecipeNode]] = {}
    for n, idxs in node_to_indices.items():
        idx = idxs.ids[component_type]
        if idx in ids_to_solve:
            component_nodes.setdefault(idx, set()).add(n)
    return component_nodes


@dataclasses.dataclass
class ConfigMatchStats:
    intersection: Set[str] = dataclasses.field(default_factory=set)
    state_dict_misses: float = 0.0
    config_misses: float = 0.0
    count: int = 0

    def __or__(self, other):
        new_count = self.count + other.count
        if new_count == 0:
            return ConfigMatchStats(self.intersection.intersection(other.intersection), 0.0, 0.0, new_count)

        self_weight = self.count / new_count
        other_weight = other.count / new_count
        return ConfigMatchStats(
            self.intersection.intersection(other.intersection),
            self.state_dict_misses * self_weight + other.state_dict_misses * other_weight,
            self.config_misses * self_weight + other.config_misses * other_weight,
            new_count,
        )


def _meta_score(m: TensorMetadata) -> int:
    return int(m.shape is None) + int(m.dtype is None)


def is_config_stub(config: Optional[model_configs.ModelConfig]) -> bool:
    return getattr(config, "identifier", None) in (None, model_configs.INFER.identifier)


def generate_configs_stats(state_dict: Iterable[str]) -> Dict[str, ConfigMatchStats]:
    state_dict_set = set(state_dict)
    state_dict_len = len(state_dict_set)
    configs_stats = {}

    if state_dict_len == 0:
        return {}

    for config in model_configs.get_all():
        match_stats = create_config_stats(config, state_dict_set)

        # heuristic: accept config only if we match more than 10% of the keys of the state dict
        if len(match_stats.intersection) >= state_dict_len * 0.1:
            configs_stats[config.identifier] = match_stats

    return configs_stats


def create_config_stats(config: ModelConfig, dict_keys: Optional[Set[str]] = None) -> ConfigMatchStats:
    config_keys = set(config.keys())
    if dict_keys is None or (dict_keys_len := len(dict_keys)) == 0:
        return ConfigMatchStats(config_keys)

    dict_keys = {config.resolve_alias(key) for key in dict_keys}

    matched_keys = dict_keys & config_keys
    state_dict_extra = dict_keys - config_keys

    state_dict_miss = set()
    for k, meta in config.keys().items():
        if k in dict_keys or meta.optional:
            continue
        state_dict_miss.add(k)

    return ConfigMatchStats(matched_keys, len(state_dict_extra) / dict_keys_len, len(state_dict_miss) / len(config.keys()), 1)


@dataclasses.dataclass
class FinalizeVisitor(RecipeVisitor):
    node_to_indices: Dict[RecipeNode, ComponentIndices]
    solved_cfg: Optional[Dict[ComponentIndex, ModelConfig]]
    solved_ms: Optional[Dict[ComponentIndex, MergeSpace]]
    check_extra_keys: bool
    check_mandatory_keys: bool

    def visit_literal(self, node: LiteralRecipeNode):
        value_dict = {
            k: (v.accept(self) if isinstance(v, RecipeNode) else v)
            for k, v in node.value_dict.items()
        }

        cfg, ms = self._solve_info(node)
        check_model_config(value_dict, cfg, self.check_extra_keys, self.check_mandatory_keys, "<in-memory>")

        return LiteralRecipeNode(value_dict, cfg, ms)

    def visit_model(self, node: ModelRecipeNode):
        cfg, ms = self._solve_info(node)
        check_model_config(node.state_dict, cfg, self.check_extra_keys, self.check_mandatory_keys, str(node.path))

        if node.is_open:
            if cfg == node.model_config and ms == node.merge_space:
                return node
            return OpenModelRecipeNode(node.state_dict, node.path, cfg, ms)
        return ClosedModelRecipeNode(node.path, cfg, ms)

    def visit_merge(self, node: MergeRecipeNode):
        args = tuple(v.accept(self) for v in node.bound_args.args)
        kwargs = {k: v.accept(self) for k, v in node.bound_args.kwargs.items()}
        bound_args = node.merge_method.get_signature().bind(*args, **kwargs)

        cfg, ms = self._solve_info(node)

        return MergeRecipeNode(node.merge_method, bound_args, cfg, ms)

    def _solve_info(self, node):
        cfg = node.model_config
        ms = node.merge_space

        idxs = self.node_to_indices.get(node)
        if idxs is not None:
            if cfg is None and self.solved_cfg is not None:
                cfg = self.solved_cfg.get(idxs.ids[ModelConfigComponentType], cfg)
            if ms is None and self.solved_ms is not None:
                ms = self.solved_ms.get(idxs.ids[MergeSpaceComponentType], ms)

        return cfg, ms


def check_model_config(
    state_dict: Iterable[str],
    config: Optional[ModelConfig],
    check_extra_keys: bool,
    check_mandatory_keys: bool,
    state_dict_origin: str,
):
    if config is None:
        return

    state_dict_keys = set(state_dict)

    if check_extra_keys:
        config_keys = set(k_alias for k, v in config.keys().items() for k_alias in [k, *v.aliases])
        extra_keys = state_dict_keys - config_keys
        if extra_keys:
            logging.warning(
                f"State dict contains keys not present in the selected model config.\n"
                f"  Origin: {state_dict_origin}\n"
                f"  Config: {config.identifier}\n"
                f"  Extra keys ({len(extra_keys)}): {_fmt_list(sorted(extra_keys))}\n"
                "Note: this may be harmless (e.g., optimizer slots or unused tensors), but can indicate a wrong config selection."
            )

    if check_mandatory_keys:
        missing = []
        for key, meta in config.keys().items():
            if meta.optional:
                continue
            if key in state_dict_keys:
                continue
            if any(a in state_dict_keys for a in meta.aliases):
                continue
            missing.append((key, tuple(meta.aliases)))

        if missing:
            lines = "\n".join(
                f"    - {k} (aliases tried: {', '.join(a) if a else 'none'})"
                for k, a in missing
            )
            raise TypeError(
                "State dict is missing required keys for the selected model config.\n"
                f"  Origin: {state_dict_origin}\n"
                f"  Missing ({len(missing)}):\n{lines}\n"
                "Fix: choose the correct model config, or provide a valid checkpoint file."
            )


def _fmt_list(items, limit=12) -> str:
    items = list(items)
    if len(items) <= limit:
        return ", ".join(map(str, items))
    head = ", ".join(map(str, items[:limit]))
    return f"{head}, ... (+{len(items)-limit} more)"
