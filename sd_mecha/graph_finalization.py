import abc
import contextlib
import dataclasses
import logging
import pathlib
import torch
from collections import defaultdict, deque, OrderedDict
from typing import ContextManager, Dict, Generic, Iterable, Iterator, List, Mapping, Optional, Set, Tuple, TypeVar
from sd_mecha.extensions import merge_spaces, model_configs, model_dirs, model_formats
from sd_mecha.extensions.merge_methods import value_to_node
from sd_mecha.extensions.merge_spaces import MergeSpace, MergeSpaceSymbol
from sd_mecha.extensions.model_configs import KeyMetadata, ModelConfig, StructuralModelConfig
from sd_mecha.keys_map import ActiveKeyMap, KeyMap, RealizedKeyRelation
from sd_mecha.recipe_nodes import (
    ClosedModelRecipeNode, LiteralRecipeNode, MergeRecipeNode, ModelRecipeNode,
    OpenModelRecipeNode, RecipeNode, RecipeNodeOrValue, RecipeVisitor, TracingRecipeVisitor,
)
from sd_mecha.streaming import SafetensorsMapping, TensorMetadata


T = TypeVar("T")


@contextlib.contextmanager
def open_graph(
    root: RecipeNodeOrValue,
    buffer_size_per_dict: int = 0,
    root_only: bool = False,
    solve_model_config: bool = True,
    solve_merge_space: bool = True,
) -> ContextManager["RecipeGraph"]:
    graph, to_close = _open_graph_impl(
        root=root,
        buffer_size_per_dict=buffer_size_per_dict,
        root_only=root_only,
        solve_model_config=solve_model_config,
        solve_merge_space=solve_merge_space,
    )

    try:
        yield graph
    finally:
        for v in to_close:
            v.close()


def _open_graph_impl(
    *,
    root: RecipeNodeOrValue,
    buffer_size_per_dict: int,
    root_only: bool,
    solve_model_config: bool,
    solve_merge_space: bool,
) -> Tuple["RecipeGraph", Iterable[SafetensorsMapping]]:
    component_types = set()
    if solve_model_config:
        component_types.add(ModelConfigComponentType)
    if solve_merge_space:
        component_types.add(MergeSpaceComponentType)

    resolve_paths_visitor = ResolvePathsVisitor()
    root = resolve_paths_visitor.process(value_to_node(root))

    node_to_index_by_type: Dict[type[ComponentType], Dict[RecipeNode, ComponentIndex]] = {}
    for t in component_types:
        node_to_index_by_type[t] = discover_component(root, t, root_only)

    active_nodes = set().union(*(m.keys() for m in node_to_index_by_type.values()))

    opener = OpenActiveStateDictsVisitor(buffer_size_per_dict, active_nodes)
    root = opener.process(root)
    node_map = {k: opener.node_map.get(v, v) for k, v in resolve_paths_visitor.node_map.items()}
    del active_nodes  # open nodes differ from closed nodes, either remap or do not use past this point

    for t in component_types:
        node_to_index_by_type[t] = {opener.node_map.get(k, k): v for k, v in node_to_index_by_type[t].items()}

    candidates: Dict[type[ComponentType], Dict[ComponentIndex, ComponentCandidates]] = {}
    for t in component_types:
        candidates[t] = t.build_candidates(node_to_index_by_type[t])

    graph = RecipeGraph(root, node_to_index_by_type, node_map, candidates, root_only)
    return graph, opener.dicts_to_close


@dataclasses.dataclass
class RecipeGraph:
    root: RecipeNode
    node_to_index_by_type: Dict[type["ComponentType"], Dict[RecipeNode, "ComponentIndex"]]
    node_map: Dict[RecipeNode, RecipeNode]
    candidates: Dict[type["ComponentType"], Dict["ComponentIndex", "ComponentCandidates"]]
    is_root_only: bool

    def finalize(
        self,
        model_config: Optional[str | ModelConfig] = None,
        merge_space: Optional[str | MergeSpace] = None,
        model_config_preference: Optional[Iterable[str | ModelConfig]] = None,
        merge_space_preference: Optional[Iterable[str | MergeSpace]] = None,
        check_extra_keys: bool = True,
        check_mandatory_keys: bool = True,
    ) -> "FinalizeReturn":
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

        candidates = self._clone_candidates()
        component_types = {}
        if ModelConfigComponentType in candidates:
            component_types[ModelConfigComponentType] = (model_config, model_config_preference)
        if MergeSpaceComponentType in candidates:
            component_types[MergeSpaceComponentType] = (merge_space, merge_space_preference)

        for t, (return_hint, return_preference) in component_types.items():
            root_idx = self.node_to_index_by_type[t][self.root]
            root_t_candidates = candidates[t][root_idx]
            if return_hint is not None:
                root_t_candidates.apply_return_hint(return_hint, reason=f"return hint {return_hint}")
            if return_preference is not None:
                root_t_candidates.apply_preference(return_preference)

        solutions = {t: {idx: c.finalize() for idx, c in candidates[t].items()} for t in candidates}

        finalizer = FinalizeVisitor(
            node_to_index_by_type=self.node_to_index_by_type,
            solved_cfg=solutions.get(ModelConfigComponentType),
            solved_ms=solutions.get(MergeSpaceComponentType),
            check_extra_keys=check_extra_keys,
            check_mandatory_keys=check_mandatory_keys,
        )
        finalized_root = self.root.accept(finalizer)

        to_finalized_node = {original: finalizer.node_map[opened] for original, opened in self.node_map.items()}

        node_to_keys: Dict[RecipeNode, Set[str]] = {}
        realized_key_maps: Dict[MergeRecipeNode, ActiveKeyMap[RealizedKeyRelation]] = {}

        if ModelConfigComponentType in candidates and not self.is_root_only:
            t = ModelConfigComponentType

            finalized_node_to_indices: Dict[RecipeNode, Set[ComponentIndex]] = defaultdict(set)
            for old_node, index in self.node_to_index_by_type[t].items():
                finalized_node = finalizer.node_map[old_node]
                finalized_node_to_indices[finalized_node].add(index)

            component_keys: Dict[ComponentIndex, Set[str]] = {}
            component_metadata: Dict[ComponentIndex, Optional[Dict[str, TensorMetadata | KeyMetadata]]] = {}

            for idxs in finalized_node_to_indices.values():
                for idx in idxs:
                    solved = solutions[t][idx]
                    cand = candidates[t][idx]

                    if solved.identifier == "structural":
                        component_metadata[idx] = cand.possible_keys
                        component_keys[idx] = set(cand.possible_keys.keys())
                    else:
                        component_metadata[idx] = solved.keys()
                        stats = cand.stats or {}
                        match = stats.get(solved.identifier)
                        if match is not None:
                            component_keys[idx] = set(match.intersection) | set(cand.possible_keys)
                        else:
                            component_keys[idx] = set(cand.possible_keys) or set(solved.keys().keys())

            node_to_key_domain: Dict[RecipeNode, Set[str]] = {}
            node_to_metadata_domain: Dict[RecipeNode, Optional[Dict[str, KeyMetadata]]] = {}

            for node, idxs in finalized_node_to_indices.items():
                idxs = set(idxs)
                node_to_key_domain[node] = _union_key_sets(component_keys[idx] for idx in idxs)
                node_to_metadata_domain[node] = _merge_component_metadata(component_metadata[idx] for idx in idxs)

            keys_visitor = PropagatableKeyVisitor(node_to_key_domain, node_to_metadata_domain)
            keys_visitor.visit_all_keys(finalized_root)

            node_to_keys = {
                node: set(keys)
                for node, keys in keys_visitor.filtered_node_keys.items()
            }
            realized_key_maps = dict(keys_visitor.realized_key_maps)

        return FinalizeReturn(finalized_root, node_to_keys, realized_key_maps, to_finalized_node)

    @staticmethod
    def _dedupe_relations(relations: Iterable[RealizedKeyRelation]) -> List[RealizedKeyRelation]:
        seen = set()
        out = []
        for rel in relations:
            key = (
                rel.outputs,
                tuple((p, tuple(ks)) for p, ks in rel.inputs.items()),
                id(rel.meta) if callable(rel.meta) else repr(rel.meta),
            )
            if key not in seen:
                seen.add(key)
                out.append(rel)
        return out

    def root_candidates(
        self,
        model_config: Optional[str | ModelConfig] = None,
        merge_space: Optional[str | MergeSpace] = None,
        model_config_preference: Optional[Iterable[str | ModelConfig]] = None,
        merge_space_preference: Optional[Iterable[str | MergeSpace]] = None,
    ) -> "CandidatesReturn":
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

        candidates = self._clone_candidates()
        component_types = {}
        if ModelConfigComponentType in candidates:
            component_types[ModelConfigComponentType] = (model_config, model_config_preference)
        if MergeSpaceComponentType in candidates:
            component_types[MergeSpaceComponentType] = (merge_space, merge_space_preference)

        if ModelConfigComponentType not in candidates:
            raise RuntimeError("The recipe graph was opened without model config analysis.")

        for t, (return_hint, return_preference) in component_types.items():
            root_idx = self.node_to_index_by_type[t][self.root]
            root_t_candidates = candidates[t][root_idx]
            if return_hint is not None:
                root_t_candidates.apply_return_hint(return_hint, reason=f"return hint {return_hint}")
            if return_preference is not None:
                root_t_candidates.apply_preference(return_preference)

        return CandidatesReturn(**{
            t.name: candidates[t][self.node_to_index_by_type[t][self.root]] if t in candidates else None for t in (
                ModelConfigComponentType,
                MergeSpaceComponentType,
            )
        })

    def _clone_candidates(self) -> Dict[type["ComponentType"], Dict["ComponentIndex", "ComponentCandidates"]]:
        out = {}
        for t, by_idx in self.candidates.items():
            out[t] = {}
            for idx, c in by_idx.items():
                out[t][idx] = c.clone()
        return out


@dataclasses.dataclass
class FinalizeReturn:
    root: RecipeNode
    node_to_keys: Mapping[RecipeNode, Set[str]]
    realized_key_maps: Mapping[RecipeNode, ActiveKeyMap[RealizedKeyRelation]]
    to_finalized_node: Mapping[RecipeNode, RecipeNode]


@dataclasses.dataclass
class CandidatesReturn:
    model_config: Optional["ModelConfigCandidates"]
    merge_space: Optional["MergeSpaceCandidates"]


@dataclasses.dataclass
class ResolvePathsVisitor(RecipeVisitor):
    node_map: Dict[RecipeNode, RecipeNode] = dataclasses.field(default_factory=dict)

    def process(self, node: RecipeNode) -> RecipeNode:
        cached = self.node_map.get(node)
        if cached is not None:
            return cached

        out = node.accept(self)
        self.node_map[node] = out
        return out

    def visit_literal(self, node: LiteralRecipeNode):
        value_dict = {
            k: self.process(v) if isinstance(v, RecipeNode) else v
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
        args = tuple(self.process(v) for v in node.bound_args.args)
        kwargs = {k: self.process(v) for k, v in node.bound_args.kwargs.items()}
        bound_args = node.merge_method.get_signature().bind(*args, **kwargs)
        return MergeRecipeNode(node.merge_method, bound_args, node.model_config, node.merge_space)


@dataclasses.dataclass
class OpenActiveStateDictsVisitor(RecipeVisitor):
    buffer_size_per_dict: int
    active_nodes: Set[RecipeNode]
    dicts_to_close: List[SafetensorsMapping] = dataclasses.field(default_factory=list)
    dicts_cache: Dict[pathlib.Path, SafetensorsMapping] = dataclasses.field(default_factory=dict)
    node_map: Dict[RecipeNode, RecipeNode] = dataclasses.field(default_factory=dict)

    def process(self, node: RecipeNode) -> RecipeNode:
        cached = self.node_map.get(node)
        if cached is not None:
            return cached

        if node not in self.active_nodes:
            self.node_map[node] = node
            return node

        out = node.accept(self)
        self.node_map[node] = out
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
            self.dicts_to_close.append(sd)

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
class ComponentCandidates(TracingRecipeVisitor, abc.ABC, Generic[T]):
    def clone(self) -> "ComponentCandidates[T]":
        ...

    def apply_return_hint(self, cfg: T, *, reason: str) -> None:
        ...

    def apply_preference(self, prefs: Iterable[T]) -> None:
        ...

    @abc.abstractmethod
    def __iter__(self) -> Iterator[T]:
        ...

    @abc.abstractmethod
    def __bool__(self) -> bool:
        ...

    @abc.abstractmethod
    def finalize(self) -> T:
        ...


class ComponentType(abc.ABC, Generic[T]):
    all_components: Dict[str, type["ComponentType"]] = {}

    name: str
    Candidates: type[ComponentCandidates[T]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        cls.name = kwargs.get("component_name", cls.__name__)
        if cls.name in cls.all_components:
            raise TypeError(f"Component {cls.name} already exists.")

        cls.all_components[cls.name] = cls

    @classmethod
    def build_candidates(
        cls,
        node_to_index: Dict[RecipeNode, "ComponentIndex"],
    ) -> Dict["ComponentIndex", ComponentCandidates[T]]:
        candidates = {rep.find(): cls.Candidates() for rep in node_to_index.values()}
        for n, rep in node_to_index.items():
            n.accept(
                candidates[rep.find()],
                node_to_index=node_to_index,
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
    possible_keys: Dict[str, TensorMetadata] = dataclasses.field(default_factory=dict)
    common_keys: Optional[Dict[str, TensorMetadata]] = None

    def clone(self) -> "ModelConfigCandidates":
        out = ModelConfigCandidates()
        out.requires_known_config = self.requires_known_config
        out.explicit_ids = set(self.explicit_ids)
        out.possible_keys = dict(self.possible_keys)
        out.common_keys = None if self.common_keys is None else dict(self.common_keys)
        if self.stats is None:
            out.stats = None
        else:
            out.stats = {k: dataclasses.replace(v, intersection=set(v.intersection)) for k, v in self.stats.items()}
        return out

    def apply_return_hint(self, cfg: ModelConfig, *, reason: str) -> None:
        self.requires_known_config = True
        self.explicit_ids.add(cfg.identifier)
        self._constrain_to_id(cfg, reason=reason)
        self._update_with_metadata(cfg.metadata(), cfg)

    def apply_preference(self, prefs: Iterable[ModelConfig]) -> None:
        if self.stats is not None and not self.stats:
            return

        for mc in prefs:
            if self.stats is None or mc.identifier in self.stats:
                self.apply_return_hint(mc, reason=f"return preference {mc.identifier}")
                break

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

    def visit_merge(self, node: MergeRecipeNode, node_to_index: Dict[RecipeNode, "ComponentIndex"], candidates: Dict["ComponentIndex", "ModelConfigCandidates"]):
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

            child_rep = node_to_index.get(child)
            if child_rep is None:
                continue

            child_candidates = candidates.get(child_rep)
            if child_candidates is None:
                continue

            child_candidates.apply_return_hint(cfg, reason=f"merge input {cfg.identifier}")

    def _update_with_metadata(self, metadata: Mapping[str, TensorMetadata], hint: Optional[ModelConfig]) -> None:
        self._update_common_keys(metadata)
        self._update_possible_keys(metadata)

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
                hint_keys = set(hint.keys())
                prev = self.stats.get(hint.identifier, ConfigMatchStats(hint_keys))
                self.stats[hint.identifier] = prev | cfg_stat

    def _update_common_keys(self, metadata: Mapping[str, TensorMetadata]) -> None:
        if self.common_keys is None:
            self.common_keys = dict(metadata)
            return

        out = {}
        for k, m1 in self.common_keys.items():
            m2 = metadata.get(k)
            if m2 is None:
                continue
            out[k] = m1 if _meta_cost(m1) < _meta_cost(m2) else m2
        self.common_keys = out

    def _update_possible_keys(self, metadata: Mapping[str, TensorMetadata]) -> None:
        for k, m2 in metadata.items():
            m1 = self.possible_keys.get(k)
            if m1 is None:
                self.possible_keys[k] = m2
            else:
                self.possible_keys[k] = _union_meta(m1, m2)

    def _constrain_to_id(self, config: ModelConfig, *, reason: str) -> None:
        if self.stats is None:
            keys = set(config.keys())
            self.stats = {config.identifier: ConfigMatchStats(keys)}
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

    def __iter__(self):
        if self.stats is None:
            return iter(())

        for cfg, stat in sorted(self.stats.items(), key=lambda kv: kv[1].state_dict_misses):
            yield model_configs.resolve(cfg)

    def __bool__(self) -> bool:
        return bool(self.stats)

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

    def clone(self) -> "MergeSpaceCandidates":
        out = MergeSpaceCandidates()
        out.allowed = self.allowed.copy() if self.allowed is not None else None
        return out

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

    def visit_merge(self, node: MergeRecipeNode, node_to_index: Dict[RecipeNode, "ComponentIndex"], candidates: Dict["ComponentIndex", "MergeSpaceCandidates"]):
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

            child_rep = node_to_index.get(child)
            if child_rep is None:
                continue

            child_candidates = candidates.get(child_rep)
            if child_candidates is None:
                continue

            if isinstance(param_ms, MergeSpaceSymbol):
                child_candidates.constrain(set(param_ms.merge_spaces), reason="merge input symbol constraint")
            else:
                assert isinstance(param_ms, set), f"Unexpected merge space type: {type(param_ms)}"
                child_candidates.constrain(set(param_ms), reason="merge input fixed-set constraint")

    def __iter__(self):
        if self.allowed is None:
            return iter(())

        yield from self.allowed

    def __bool__(self) -> bool:
        return bool(self.allowed)

    def finalize(self) -> MergeSpace:
        if self.allowed is None:
            all_spaces = tuple(ms.identifier for ms in merge_spaces.get_all())
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

    def __eq__(self, other):
        return self.find() is other.find()

    def __hash__(self):
        return hash(id(self.find()))


def discover_component(
    root: RecipeNode,
    component_type: type[ComponentType],
    root_only: bool,
) -> Dict[RecipeNode, ComponentIndex]:
    node_to_index: Dict[RecipeNode, ComponentIndex] = {}

    root_index = ComponentIndex(component_type)
    nodes_queue = deque((
        (root, root_index),
    ))
    seen = set()

    while nodes_queue:
        stack = [nodes_queue.popleft()]
        visitor = DiscoverComponentVisitor(nodes_queue, stack, node_to_index, component_type, seen)
        while stack:
            visitor.process(*stack.pop())

    root_idx = node_to_index[root].find()
    out: Dict[RecipeNode, ComponentIndex] = {}
    for n, idx in node_to_index.items():
        idx = idx.find()
        if not root_only or idx == root_idx:
            out[n] = idx
    return out


@dataclasses.dataclass
class DiscoverComponentVisitor(RecipeVisitor):
    queue: deque[Tuple[RecipeNode, ComponentIndex]]
    stack: List[Tuple[RecipeNode, ComponentIndex]]
    node_to_index: Dict[RecipeNode, ComponentIndex]
    component_type: type[ComponentType]
    seen: Set[Tuple[RecipeNode, int]]
    cur_index: Optional[ComponentIndex] = None

    def process(self, node: RecipeNode, index: ComponentIndex) -> None:
        self.cur_index = self._attach_node(node, index)
        key = node, id(self.cur_index)
        if key in self.seen:
            return
        self.seen.add(key)
        node.accept(self)

    def visit_literal(self, node: LiteralRecipeNode):
        assert self.cur_index is not None
        for child in node.value_dict.values():
            if not isinstance(child, RecipeNode):
                continue

            child_index = self.cur_index
            if self.component_type.literal_delineates(node):
                child_index = ComponentIndex(self.component_type)

            self._handle_child(self.cur_index, child, child_index)

    def visit_model(self, node: ModelRecipeNode):
        assert self.cur_index is not None

    def visit_merge(self, node: MergeRecipeNode):
        assert self.cur_index is not None
        child_indices = self.component_type.merge_delineate(node, self.cur_index)

        for child in (*node.bound_args.args, *node.bound_args.kwargs.values()):
            child_idx = child_indices[child]
            self._handle_child(self.cur_index, child, child_idx)

    def _handle_child(self, parent_idx: ComponentIndex, child: RecipeNode, child_idx: ComponentIndex) -> None:
        child_idx = self._attach_node(child, child_idx)
        if child_idx == parent_idx:
            self.stack.append((child, child_idx))
        else:
            self.queue.append((child, child_idx))

    def _attach_node(self, node: RecipeNode, idx: ComponentIndex) -> ComponentIndex:
        existing_idx = self.node_to_index.get(node)
        if existing_idx is not None:
            idx |= existing_idx

        self.node_to_index[node] = idx
        return idx.find()


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


def _meta_cost(m: TensorMetadata) -> int:
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
    node_to_index_by_type: Dict[type[ComponentType], Dict[RecipeNode, ComponentIndex]]
    solved_cfg: Optional[Dict[ComponentIndex, ModelConfig]]
    solved_ms: Optional[Dict[ComponentIndex, MergeSpace]]
    check_extra_keys: bool
    check_mandatory_keys: bool
    node_map: Dict[RecipeNode, RecipeNode] = dataclasses.field(default_factory=dict)

    def visit_literal(self, node: LiteralRecipeNode):
        value_dict = {
            k: (v.accept(self) if isinstance(v, RecipeNode) else v)
            for k, v in node.value_dict.items()
        }

        cfg, ms = self._solve_info(node)
        check_model_config(value_dict, cfg, self.check_extra_keys, self.check_mandatory_keys, "<in-memory>")

        res = LiteralRecipeNode(value_dict, cfg, ms)
        self.node_map[node] = res
        return res

    def visit_model(self, node: ModelRecipeNode):
        cfg, ms = self._solve_info(node)
        check_model_config(node.state_dict, cfg, self.check_extra_keys, self.check_mandatory_keys, str(node.path))

        if node.is_open:
            if cfg == node.model_config and ms == node.merge_space:
                res = node
            else:
                res = OpenModelRecipeNode(node.state_dict, node.path, cfg, ms)
        else:
            res = ClosedModelRecipeNode(node.path, cfg, ms)
        self.node_map[node] = res
        return res

    def visit_merge(self, node: MergeRecipeNode):
        args = tuple(v.accept(self) for v in node.bound_args.args)
        kwargs = {k: v.accept(self) for k, v in node.bound_args.kwargs.items()}
        bound_args = node.merge_method.get_signature().bind(*args, **kwargs)

        cfg, ms = self._solve_info(node)

        res = MergeRecipeNode(node.merge_method, bound_args, cfg, ms)
        self.node_map[node] = res
        return res

    def _solve_info(self, node):
        cfg = node.model_config
        ms = node.merge_space

        if cfg is None and self.solved_cfg is not None:
            cfg_idx = self.node_to_index_by_type[ModelConfigComponentType].get(node)
            if cfg_idx is not None:
                cfg = self.solved_cfg.get(cfg_idx, cfg)
        if ms is None and self.solved_ms is not None:
            ms_idx = self.node_to_index_by_type[MergeSpaceComponentType].get(node)
            if ms_idx is not None:
                ms = self.solved_ms.get(ms_idx, ms)

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


def _union_key_sets(key_sets: Iterable[Set[str]]) -> Set[str]:
    out = set()
    for ks in key_sets:
        out.update(ks)
    return out


def _merge_component_metadata(
    metadata_dicts: Iterable[Optional[Dict[str, TensorMetadata | KeyMetadata]]]
) -> Optional[Dict[str, KeyMetadata]]:
    metadata_dicts = [d for d in metadata_dicts if d is not None]
    if not metadata_dicts:
        return None

    common_keys = set(metadata_dicts[0].keys())
    for d in metadata_dicts[1:]:
        common_keys.intersection_update(d.keys())

    out: Dict[str, KeyMetadata] = {}
    for k in common_keys:
        best = metadata_dicts[0][k]
        for d in metadata_dicts[1:]:
            cand = d[k]
            # Keep the less informative / more conservative metadata.
            # This matches the spirit of ModelConfigCandidates._update_common_keys().
            best = _intersect_meta(best, cand)
        out[k] = best

    return out


def _intersect_meta(a, b):
    return KeyMetadata(
        a.shape if b.shape == a.shape else None,
        a.dtype if b.dtype == a.dtype else None,
        [a for a in getattr(a, "aliases", ()) if a in set(getattr(b, "aliases", ()))],
        getattr(a, "optional", True) and getattr(b, "optional", True),
    )


def _union_meta(a, b):
    return TensorMetadata(
        b.shape if a.shape is None else a.shape,
        b.dtype if a.dtype is None else a.dtype,
        # [a for a in getattr(a, "aliases", ()) if a not in getattr(b, "aliases", ())] + getattr(b, "aliases", []),
        # getattr(a, "optional", True) or getattr(b, "optional", True),
    )


@dataclasses.dataclass
class PropagatableKeyVisitor(RecipeVisitor):
    node_to_key_domain: Dict[RecipeNode, Set[str]]
    node_to_metadata_domain: Dict[RecipeNode, Optional[Dict[str, KeyMetadata]]]

    filtered_node_keys: Dict[RecipeNode, Set[str]] = dataclasses.field(default_factory=lambda: defaultdict(set))
    realized_key_maps: Dict[MergeRecipeNode, ActiveKeyMap[RealizedKeyRelation]] = dataclasses.field(default_factory=dict)

    # Exact-key memo for literal/model nodes only
    _key_memo: Dict[Tuple[RecipeNode, str], bool] = dataclasses.field(default_factory=dict)

    # Partition memo for merge nodes
    _partition_memo: Dict[
        Tuple[MergeRecipeNode, Tuple[str, ...]],
        Optional[RealizedKeyRelation],
    ] = dataclasses.field(default_factory=dict)

    # Exact active output-key -> realized relation
    _active_relations_by_key: Dict[
        MergeRecipeNode,
        Dict[str, RealizedKeyRelation],
    ] = dataclasses.field(default_factory=lambda: defaultdict(dict))

    _root: Optional[RecipeNode] = None
    _MISSING = object()

    def visit_all_keys(self, root: RecipeNode):
        self._root = root

        if isinstance(root, MergeRecipeNode):
            for outputs in self._root_output_groups(root):
                probe_key = outputs[0]
                if not root.accept(self, probe_key):
                    continue

                relation = self._partition_memo[(root, outputs)]
                assert relation is not None

                # Root is special: if a root partition is realizable,
                # all outputs in that root partition belong to the final model.
                for k in outputs:
                    self.filtered_node_keys[root].add(k)
                    self._active_relations_by_key[root][k] = relation
        else:
            for k in self._possible_keys(root):
                root.accept(self, k)

        self.realized_key_maps = {
            node: ActiveKeyMap(by_key)
            for node, by_key in self._active_relations_by_key.items()
        }

    def visit_literal(self, node: LiteralRecipeNode, output_key: str) -> bool:
        cached = self._key_memo.get((node, output_key), self._MISSING)
        if cached is not self._MISSING:
            if cached:
                self.filtered_node_keys[node].add(output_key)
            return cached

        ok = False
        if output_key in self._possible_keys(node):
            for candidate_key in (output_key, *node.model_config.aliases().get(output_key, ())):
                if candidate_key not in node.value_dict:
                    continue
                child = node.value_dict[candidate_key]
                if isinstance(child, RecipeNode):
                    ok = child.accept(self, output_key)
                else:
                    ok = True
                if ok:
                    break

        if ok:
            self.filtered_node_keys[node].add(output_key)

        self._key_memo[(node, output_key)] = ok
        return ok

    def visit_model(self, node: ModelRecipeNode, output_key: str) -> bool:
        cached = self._key_memo.get((node, output_key), self._MISSING)
        if cached is not self._MISSING:
            if cached:
                self.filtered_node_keys[node].add(output_key)
            return cached

        ok = False
        if output_key in self._possible_keys(node):
            aliases = node.model_config.aliases().get(output_key, ())
            ok = any(k in node.state_dict for k in (output_key, *aliases))

        if ok:
            self.filtered_node_keys[node].add(output_key)

        self._key_memo[(node, output_key)] = ok
        return ok

    def visit_merge(self, node: MergeRecipeNode, output_key: str) -> bool:
        if output_key not in self._possible_keys(node):
            return False

        relation = node.key_map().get(output_key)
        if relation is None:
            return False

        outputs = tuple(relation.outputs)
        memo_key = (node, outputs)

        cached = self._partition_memo.get(memo_key, self._MISSING)
        if cached is not self._MISSING:
            if cached is None:
                return False

            # Exact-key activation only
            self.filtered_node_keys[node].add(output_key)
            self._active_relations_by_key[node][output_key] = cached
            return True

        all_args = node.all_args()

        for clause in relation.clauses:
            if self._clause_succeeds(all_args, clause.by_param):
                realized = RealizedKeyRelation(
                    outputs=relation.outputs,
                    inputs=clause.by_param,
                    meta=clause.meta,
                )
                self._partition_memo[memo_key] = realized

                # Exact-key activation only
                self.filtered_node_keys[node].add(output_key)
                self._active_relations_by_key[node][output_key] = realized
                return True

        self._partition_memo[memo_key] = None
        return False

    def _clause_succeeds(self, all_args, by_param) -> bool:
        for child_name, required_child_keys in by_param.items():
            children = all_args.get(child_name)
            if not children:
                return False

            for child_key in required_child_keys:
                found = False
                for child in children:
                    if child.accept(self, child_key):
                        found = True
                        break
                if not found:
                    return False

        return True

    def _root_output_groups(self, node: MergeRecipeNode) -> list[Tuple[str, ...]]:
        possible = self._possible_keys(node)
        key_map = node.key_map()
        seen = set()
        out = []

        for k in possible:
            if k in seen:
                continue
            relation = key_map.get(k)
            if relation is None:
                seen.add(k)
                out.append((k,))
                continue
            group = tuple(x for x in relation.outputs if x in possible)
            if not group:
                seen.add(k)
                out.append((k,))
                continue
            seen.update(group)
            out.append(group)

        return out

    def _possible_keys(self, node: RecipeNode) -> Set[str]:
        return self.node_to_key_domain.get(node, set())

    def _leaf_has_key(self, node: RecipeNode, output_key: str, mapping) -> bool:
        cfg = getattr(node, "model_config", None)
        aliases = ()
        if cfg is not None:
            aliases = tuple(cfg.aliases().get(output_key, ()))
        return any(k in mapping for k in (output_key, *aliases))


def _fmt_list(items, limit=12) -> str:
    items = list(items)
    if len(items) <= limit:
        return ", ".join(map(str, items))
    head = ", ".join(map(str, items[:limit]))
    return f"{head}, ... (+{len(items)-limit} more)"
