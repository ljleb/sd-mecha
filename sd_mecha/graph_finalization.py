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

    root = value_to_node(root).accept(ResolvePathsVisitor())

    node_to_indices, processed_nodes = discover_components(root, root_only, set(component_types))

    active_nodes = processed_nodes if root_only else set(node_to_indices)
    opener = OpenActiveStateDictsVisitor(buffer_size_per_dict, active_nodes)
    root = opener.process(root)
    new_nodes_map = opener.transform_cache

    node_to_indices = {new_nodes_map.get(k, k): v for k, v in node_to_indices.items()}
    processed_nodes = {new_nodes_map.get(v, v) for v in processed_nodes}
    del active_nodes

    candidates: Dict[type[ComponentType], Dict[ComponentIndex, ComponentCandidates]] = {}
    for t in component_types:
        candidates[t] = t.build_candidates(node_to_indices, processed_nodes)

    graph = RecipeGraph(root, node_to_indices, new_nodes_map, processed_nodes, candidates, root_only)

    return graph, opener.dicts_to_close


@dataclasses.dataclass
class RecipeGraph:
    root: RecipeNode
    node_to_indices: Dict[RecipeNode, "ComponentIndices"]
    to_open_node: Dict[RecipeNode, RecipeNode]
    processed_nodes: Set[RecipeNode]
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

        root_idx = self.node_to_indices[self.root].ids
        for t, (return_hint, return_preference) in component_types.items():
            root_idx_t = root_idx[t]
            root_t_candidates = candidates[t][root_idx_t]
            if return_hint is not None:
                root_t_candidates.apply_return_hint(return_hint, reason=f"return hint {return_hint}")
            if return_preference is not None:
                root_t_candidates.apply_preference(return_preference)

        solutions = {t: {idx: c.finalize() for idx, c in candidates[t].items()} for t in candidates}

        finalizer = FinalizeVisitor(
            node_to_indices=self.node_to_indices,
            solved_cfg=solutions.get(ModelConfigComponentType),
            solved_ms=solutions.get(MergeSpaceComponentType),
            check_extra_keys=check_extra_keys,
            check_mandatory_keys=check_mandatory_keys,
        )
        finalized_root = self.root.accept(finalizer)

        to_finalized_node = {original: finalizer.to_new_node[open] for original, open in self.to_open_node.items()}

        node_to_keys = {}
        if ModelConfigComponentType in candidates and not self.is_root_only:
            t = ModelConfigComponentType

            finalized_node_to_indices: Dict[RecipeNode, Set[ComponentIndex]] = defaultdict(set)
            for old_node, indices in self.node_to_indices.items():
                finalized_node = finalizer.to_new_node[old_node]
                finalized_node_to_indices[finalized_node].add(indices.ids[t])

            component_keys = {
                idx: candidates[t][idx].stats[solutions[t][idx].identifier].intersection.copy()
                for idxs in finalized_node_to_indices.values()
                for idx in idxs
            }

            component_metadata: Dict[ComponentIndex, Optional[Dict[str, TensorMetadata | KeyMetadata]]] = {
                idx: (
                    candidates[t][idx].common_keys
                    if solutions[t][idx].identifier == "structural"
                    else solutions[t][idx].keys()
                )
                for idxs in finalized_node_to_indices.values()
                for idx in idxs
            }

            node_to_key_domain: Dict[RecipeNode, Set[str]] = {}
            node_to_metadata_domain: Dict[RecipeNode, Optional[Dict[str, TensorMetadata | KeyMetadata]]] = {}

            for node, idxs in finalized_node_to_indices.items():
                idxs = set(idxs)
                node_to_key_domain[node] = _intersect_key_sets(component_keys[idx] for idx in idxs)
                node_to_metadata_domain[node] = _merge_component_metadata(component_metadata[idx] for idx in idxs)

            keys_visitor = PropagatableKeyVisitor(node_to_key_domain, node_to_metadata_domain)
            keys_visitor.visit_all_keys(finalized_root)
            node_to_keys = dict(keys_visitor.filtered_node_keys)

        return FinalizeReturn(finalized_root, node_to_keys, to_finalized_node)

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

        root_idx = self.node_to_indices[self.root].ids
        for t, (return_hint, return_preference) in component_types.items():
            root_idx_t = root_idx[t]
            root_t_candidates = candidates[t][root_idx_t]
            if return_hint is not None:
                root_t_candidates.apply_return_hint(return_hint, reason=f"return hint {return_hint}")
            if return_preference is not None:
                root_t_candidates.apply_preference(return_preference)

        return CandidatesReturn(**{
            t.name: candidates[t][root_idx[t]] if t in candidates else None for t in (
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
    to_finalized_node: Mapping[RecipeNode, RecipeNode]


@dataclasses.dataclass
class CandidatesReturn:
    model_config: Optional["ModelConfigCandidates"]
    merge_space: Optional["MergeSpaceCandidates"]


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
    dicts_to_close: List[SafetensorsMapping] = dataclasses.field(default_factory=list)
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

    def clone(self) -> "ModelConfigCandidates":
        out = ModelConfigCandidates()
        out.requires_known_config = self.requires_known_config
        out.explicit_ids = set(self.explicit_ids)
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

    def __iter__(self):
        if self.allowed is None:
            return iter(())

        yield from self.allowed

    def __bool__(self) -> bool:
        return bool(self.allowed)

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
    to_new_node: Dict[RecipeNode, RecipeNode] = dataclasses.field(default_factory=dict)

    def visit_literal(self, node: LiteralRecipeNode):
        value_dict = {
            k: (v.accept(self) if isinstance(v, RecipeNode) else v)
            for k, v in node.value_dict.items()
        }

        cfg, ms = self._solve_info(node)
        check_model_config(value_dict, cfg, self.check_extra_keys, self.check_mandatory_keys, "<in-memory>")

        res = LiteralRecipeNode(value_dict, cfg, ms)
        self.to_new_node[node] = res
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
        self.to_new_node[node] = res
        return res

    def visit_merge(self, node: MergeRecipeNode):
        args = tuple(v.accept(self) for v in node.bound_args.args)
        kwargs = {k: v.accept(self) for k, v in node.bound_args.kwargs.items()}
        bound_args = node.merge_method.get_signature().bind(*args, **kwargs)

        cfg, ms = self._solve_info(node)

        res = MergeRecipeNode(node.merge_method, bound_args, cfg, ms)
        self.to_new_node[node] = res
        return res

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


def _intersect_key_sets(key_sets: Iterable[Set[str]]) -> Set[str]:
    key_sets = list(key_sets)
    if not key_sets:
        return set()

    out = set(key_sets[0])
    for ks in key_sets[1:]:
        out.intersection_update(ks)
    return out


def _merge_component_metadata(
    metadata_dicts: Iterable[Optional[Dict[str, TensorMetadata | KeyMetadata]]]
) -> Optional[Dict[str, TensorMetadata | KeyMetadata]]:
    metadata_dicts = [d for d in metadata_dicts if d is not None]
    if not metadata_dicts:
        return None

    common_keys = set(metadata_dicts[0].keys())
    for d in metadata_dicts[1:]:
        common_keys.intersection_update(d.keys())

    out: Dict[str, TensorMetadata | KeyMetadata] = {}
    for k in common_keys:
        best = metadata_dicts[0][k]
        for d in metadata_dicts[1:]:
            cand = d[k]
            # Keep the less informative / more conservative metadata.
            # This matches the spirit of ModelConfigCandidates._update_common_keys().
            best = best if _meta_score(best) >= _meta_score(cand) else cand
        out[k] = best

    return out


@dataclasses.dataclass
class PropagatableKeyVisitor(RecipeVisitor):
    node_to_key_domain: Dict[RecipeNode, Set[str]]
    node_to_metadata_domain: Dict[RecipeNode, Optional[Dict[str, TensorMetadata | KeyMetadata]]]
    filtered_node_keys: Dict[RecipeNode, Set[str]] = dataclasses.field(default_factory=lambda: defaultdict(set))

    def visit_all_keys(self, root: RecipeNode):
        root_keys = self.node_to_key_domain[root]
        for root_output_key in root_keys:
            key_visitor = dataclasses.replace(self, filtered_node_keys=defaultdict(set))
            if root.accept(key_visitor, root_output_key):
                for node, keys in key_visitor.filtered_node_keys.items():
                    self.filtered_node_keys[node].update(keys)

    def visit_literal(self, node: LiteralRecipeNode, output_key: str) -> bool:
        if output_key in self._possible_keys(node) and output_key in node.value_dict:
            self._filtered_keys(node).add(output_key)
            child = node.value_dict[output_key]
            if isinstance(child, RecipeNode):
                return child.accept(self, output_key)
            return True

        return self._key_optional(node, output_key)

    def visit_model(self, node: ModelRecipeNode, output_key: str) -> bool:
        if output_key in self._possible_keys(node) and output_key in node.state_dict:
            self._filtered_keys(node).add(output_key)
            return True

        return self._key_optional(node, output_key)

    def visit_merge(self, node: MergeRecipeNode, output_key: str) -> bool:
        if can_merge := output_key in self._possible_keys(node):
            key_map = node.key_map()
            if output_key not in key_map:
                return False

            for child_name, children in node.all_args().items():
                child_keys = key_map[output_key].inputs.get(child_name)
                if not child_keys:
                    continue
                for child_key in child_keys:
                    for child in children:
                        can_merge = can_merge and child.accept(self, child_key)

        if can_merge:
            self._filtered_keys(node).add(output_key)

        return can_merge or self._key_optional(node, output_key)

    def _possible_keys(self, node: RecipeNode) -> Set[str]:
        return self.node_to_key_domain[node]

    def _key_optional(self, node: RecipeNode, key: str) -> bool:
        metadata = self.node_to_metadata_domain[node]
        if metadata is not None and key in metadata:
            return getattr(metadata[key], "optional", True)
        return True

    def _filtered_keys(self, node: RecipeNode) -> Set[str]:
        return self.filtered_node_keys[node]


def _fmt_list(items, limit=12) -> str:
    items = list(items)
    if len(items) <= limit:
        return ", ".join(map(str, items))
    head = ", ".join(map(str, items[:limit]))
    return f"{head}, ... (+{len(items)-limit} more)"
