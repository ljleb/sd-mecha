import contextlib
import dataclasses
import logging
import pathlib
import torch
from collections import OrderedDict
from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence, Set
from sd_mecha.extensions import merge_spaces, model_configs, model_dirs, model_formats
from sd_mecha.extensions.merge_methods import value_to_node
from sd_mecha.extensions.merge_spaces import MergeSpace, MergeSpaceSymbol
from sd_mecha.extensions.model_configs import ModelConfig, StructuralModelConfig
from sd_mecha.recipe_nodes import (
    LiteralRecipeNode, MergeRecipeNode, ModelRecipeNode, OpenModelRecipeNode, RecipeNode,
    RecipeNodeOrValue, TracingRecipeVisitor,
)
from sd_mecha.streaming import SafetensorsMapping, TensorMetadata


@contextlib.contextmanager
def open_graph(
    root: RecipeNodeOrValue,
    buffer_size_per_dict: int = 0,
    check_extra_keys: bool = False,
    check_mandatory_keys: bool = False,
    return_model_config: Optional[str | ModelConfig] = None,
    return_merge_space: Optional[str | MergeSpace] = None,
    return_merge_space_preference: Optional[str | MergeSpace] = None,
) -> Iterator[RecipeNode]:
    root = value_to_node(root)

    state_dicts_visitor = OpenStateDictsVisitor(buffer_size_per_dict=buffer_size_per_dict)
    root.accept(state_dicts_visitor)
    state_dicts = state_dicts_visitor.state_dicts
    dicts_cache = state_dicts_visitor.dicts_cache

    model_configs_visitor = ModelConfigsVisitor(state_dicts=state_dicts)
    root.accept(model_configs_visitor)

    merge_spaces_visitor = MergeSpacesVisitor()
    root.accept(merge_spaces_visitor)

    return_model_config = model_configs.resolve(return_model_config) if isinstance(return_model_config, str) else return_model_config
    return_merge_space = merge_spaces.resolve(return_merge_space) if isinstance(return_merge_space, str) else return_merge_space
    return_merge_space_preference = merge_spaces.resolve(return_merge_space_preference) if isinstance(return_merge_space_preference, str) else return_merge_space_preference

    if return_model_config is not None:
        root_cands = model_configs_visitor.candidates[root].find()
        root_cands |= create_config_candidates(return_model_config.metadata(), return_model_config)

    if return_merge_space is not None:
        root_ms_cands = merge_spaces_visitor.candidates[root].find()
        previous_constraints = set(root_ms_cands.allowed) if root_ms_cands.allowed else {}
        root_ms_cands.constrain({return_merge_space})
        if root_ms_cands.is_empty():
            raise RuntimeError(
                "Return merge-space hint conflicts with existing constraints.\n"
                f"  Provided hint: {return_merge_space}\n"
                f"  Existing constraints: {previous_constraints}\n"
                "Fix: remove the hint or adjust upstream merge-space constraints."
            )
    if return_merge_space_preference is not None:
        root_cands = merge_spaces_visitor.candidates[root].find()
        allowed = root_cands.allowed
        chosen = None
        if allowed is None:
            chosen = return_merge_space_preference
        elif len(allowed) > 1:
            if return_merge_space_preference in allowed:
                chosen = return_merge_space_preference
            else:
                chosen = next(iter(allowed))

        if chosen is not None:
            root_cands.constrain({chosen})

    finalize_visitor = FinalizeVisitor(
        check_extra_keys=check_extra_keys,
        check_mandatory_keys=check_mandatory_keys,
        cfg_candidates=model_configs_visitor.candidates,
        ms_candidates=merge_spaces_visitor.candidates,
        state_dicts=state_dicts,
    )
    finalized_root = root.accept(finalize_visitor)

    del finalize_visitor, merge_spaces_visitor, model_configs_visitor, state_dicts, state_dicts_visitor

    try:
        yield finalized_root
    finally:
        for v in reversed(dicts_cache.values()):
            v.close()


@dataclasses.dataclass
class OpenStateDictsVisitor(TracingRecipeVisitor):
    buffer_size_per_dict: int = 0
    dicts_cache: Dict[pathlib.Path, SafetensorsMapping] = dataclasses.field(default_factory=dict)
    state_dicts: Dict[ModelRecipeNode, SafetensorsMapping] = dataclasses.field(default_factory=dict)
    seen_nodes: Set[RecipeNode] = dataclasses.field(default_factory=set)

    def visit_literal(self, node: LiteralRecipeNode):
        self.seen_nodes.add(node)

        for k, v in node.value_dict.items():
            if isinstance(v, RecipeNode) and v not in self.seen_nodes:
                v.accept(self)

    def visit_model(self, node: ModelRecipeNode):
        self.seen_nodes.add(node)
        self.state_dicts[node] = self.__load_dict(node)

    def visit_merge(self, node: MergeRecipeNode):
        self.seen_nodes.add(node)
        for v in node.bound_args.arguments.values():
            if v not in self.seen_nodes:
                v.accept(self)

    def __load_dict(self, node: ModelRecipeNode):
        path = model_dirs.normalize_path(node.path)
        if path not in self.dicts_cache:
            if not path.exists():
                searched = [str(d) for d in model_dirs.get_all()] if not path.is_absolute() else []
                raise FileNotFoundError(
                    "Model file not found.\n"
                    f"  Requested: {node.path}\n"
                    f"  Resolved:  {path}\n"
                    + (f"  Searched base dirs: {searched}\n" if searched else "")
                    + "Fix: provide an absolute path, or add the correct base directory to model_dirs."
                )

            matching_formats = []
            for model_format in model_formats.get_all():
                if model_format.matches(path):
                    matching_formats.append(model_format)

            if len(matching_formats) > 1:
                ids = [f.identifier for f in matching_formats]
                raise RuntimeError(
                    "Model format detection is ambiguous.\n"
                    f"  Model path: {path}\n"
                    f"  Matched formats: {ids}\n"
                    "Fix: rename the file to use an unambiguous extension, or restrict enabled model formats."
                )
            if len(matching_formats) < 1:
                known = [f.identifier for f in model_formats.get_all()]
                raise RuntimeError(
                    "Model format detection failed: no registered format matches this file.\n"
                    f"  Model path: {path}\n"
                    f"  Known formats: {known}\n"
                    "Fix: install/enable the format plugin for this file type, or convert the model to a supported format."
                )

            self.dicts_cache[path] = matching_formats[0].get_read_dict(path, self.buffer_size_per_dict)

        return self.dicts_cache[path]


@dataclasses.dataclass
class ConfigMatchStats:
    intersection: Set[str] = dataclasses.field(default_factory=set)
    state_dict_misses: float = 0.0
    config_misses: float = 0.0
    count: int = 1

    def __add__(self, other):
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


@dataclasses.dataclass
class ConfigCandidates:
    stats: Optional[Dict[str, ConfigMatchStats]] = None
    requires_known_config: bool = False
    explicit_ids: Set[str] = dataclasses.field(default_factory=set)
    common_keys: Optional[Dict[str, TensorMetadata]] = None
    _parent: Optional["ConfigCandidates"] = None
    _rank: int = 0

    def find(self) -> "ConfigCandidates":
        # path compression
        if self._parent is None:
            return self
        self._parent = self._parent.find()
        return self._parent

    def __or__(self, other):
        ra = self.find()
        rb = other.find()
        if ra is rb:
            return ra

        if ra._rank < rb._rank:
            ra, rb = rb, ra

        explicit_ids = ra.explicit_ids | rb.explicit_ids
        if len(explicit_ids) > 1:
            raise RuntimeError(
                "Incompatible model config constraints: multiple explicit configs in one component.\n"
                f"  Explicit configs: {sorted(explicit_ids)}\n"
                "Fix: remove conflicting model_config_hint values or split the recipe into separate components."
            )

        rb._parent = ra
        if ra._rank == rb._rank:
            ra._rank += 1

        ra.common_keys = _merge_common_keys(ra.common_keys, rb.common_keys)
        ra.requires_known_config = ra.requires_known_config or rb.requires_known_config
        ra.explicit_ids = explicit_ids
        ra.stats = _merge_stats(ra.stats, rb.stats, ra.requires_known_config)
        return ra


def _merge_common_keys(
    a: Optional[Dict[str, TensorMetadata]],
    b: Optional[Dict[str, TensorMetadata]],
) -> Optional[Dict[str, TensorMetadata]]:
    if a is None:
        return b
    if b is None:
        return a

    out: Dict[str, TensorMetadata] = {}
    for k, m1 in a.items():
        m2 = b.get(k)
        if m2 is None:
            continue

        out[k] = m1 if _meta_score(m1) < _meta_score(m2) else m2

    return out


def _meta_score(m: TensorMetadata) -> int:
    return int(m.shape is None) + int(m.dtype is None)


def _merge_stats(
    a: Optional[Dict[str, ConfigMatchStats]],
    b: Optional[Dict[str, ConfigMatchStats]],
    requires_known_config: bool,
) -> Optional[Dict[str, ConfigMatchStats]]:
    if a is None:
        return b
    if b is None:
        return a

    common_ids = set(a.keys()).intersection(b.keys())
    if not common_ids and requires_known_config:
        raise RuntimeError(
            "Incompatible model config constraints: candidate set intersection is empty.\n"
            f"  Left:  {sorted(a.keys())}\n"
            f"  Right: {sorted(b.keys())}\n"
            "Fix: use compatible model configs or use a compatible merge method."
        )

    merged: Dict[str, ConfigMatchStats] = {}
    for cfg_id in common_ids:
        merged[cfg_id] = a[cfg_id] + b[cfg_id]
    return merged


def create_config_candidates(
    metadata: Mapping[str, TensorMetadata],
    config_hint: Optional[ModelConfig] = None,
) -> ConfigCandidates:
    common_keys = dict(metadata)
    explicit_ids = set()

    if is_config_stub(config_hint):
        stats = generate_configs_stats(metadata)
    else:
        stats = {config_hint.identifier: generate_config_stats(config_hint, set(metadata))}
        explicit_ids.add(config_hint.identifier)

    return ConfigCandidates(
        stats,
        config_hint is not None,
        explicit_ids,
        common_keys,
    )


def is_config_stub(config: Optional[model_configs.ModelConfig]) -> bool:
    return getattr(config, "identifier", None) in (None, model_configs.INFER.identifier)


@dataclasses.dataclass
class ModelConfigsVisitor(TracingRecipeVisitor):
    state_dicts: Mapping[ModelRecipeNode, SafetensorsMapping] = dataclasses.field(default_factory=dict)
    candidates: Dict[RecipeNode, ConfigCandidates] = dataclasses.field(default_factory=dict)
    connected_candidates: ConfigCandidates = dataclasses.field(default_factory=ConfigCandidates)

    def visit_literal(self, node: LiteralRecipeNode):
        self._attach_node_to_component(node)

        metadata = {}
        for k, v in node.value_dict.items():
            if isinstance(v, RecipeNode):
                v.accept(self)
            elif isinstance(v, torch.Tensor):
                metadata[k] = TensorMetadata(v.shape, v.dtype)
            elif k not in metadata:
                metadata[k] = TensorMetadata(None, None)

        candidates = create_config_candidates(metadata, node.model_config)
        self.connected_candidates |= candidates
        self.candidates[node] = self.connected_candidates.find()

    def visit_model(self, node: ModelRecipeNode):
        self._attach_node_to_component(node)

        state_dict = self.state_dicts[node]
        metadata = OrderedDict(state_dict.metadata())

        candidates = create_config_candidates(metadata, node.model_config)
        self.connected_candidates |= candidates
        self.candidates[node] = self.connected_candidates.find()

    def visit_merge(self, node: MergeRecipeNode):
        self._attach_node_to_component(node)

        input_configs = node.merge_method.get_input_configs().as_dict(len(node.bound_args.args))
        return_config = node.merge_method.get_signature().return_annotation.data.model_config

        if return_config is not None:
            if self.candidates[node].stats is not None and return_config.identifier not in self.candidates[node].stats:
                expected = sorted(self.candidates[node].stats.keys())
                raise RuntimeError(
                    "Model config constraint violation at merge method output.\n"
                    f"  Merge method: {node.merge_method.identifier}\n"
                    f"  Return config: {return_config.identifier}\n"
                    f"  Expected one of: {expected}\n"
                    "Fix: add a model_config_hint to the relevant inputs, or use a merge method whose return config matches the downstream expectation."
                )

            self.connected_candidates |= create_config_candidates(return_config.metadata(), return_config)
            self.candidates[node] = self.connected_candidates.find()

        for k, v in (*enumerate(node.bound_args.args), *node.bound_args.kwargs.items()):
            child_config = input_configs[k]
            child_connected_candidates = create_config_candidates(child_config.metadata(), child_config) if not is_config_stub(child_config) else self.connected_candidates
            v.accept(dataclasses.replace(self, connected_candidates=child_connected_candidates))

    def _attach_node_to_component(self, node: RecipeNode) -> None:
        """
        Ensure `node` belongs to the same connected-component candidate object as `self.connected_candidates`.
        If `node` was seen before through another parent path, union the components.
        """
        if node in self.candidates:
            self.connected_candidates |= self.candidates[node]
        self.candidates[node] = self.connected_candidates.find()


def generate_configs_stats(state_dict: Iterable[str]) -> Dict[str, ConfigMatchStats]:
    state_dict_set = set(state_dict)
    state_dict_len = len(state_dict_set)
    configs_stats = {}

    if state_dict_len == 0:
        return {}

    for config in model_configs.get_all():
        match_stats = generate_config_stats(config, state_dict_set)

        # heuristic: accept config only if we match more than 10% of the keys of the state dict
        if len(match_stats.intersection) >= state_dict_len * 0.1:
            configs_stats[config.identifier] = match_stats

    return configs_stats


def generate_config_stats(config: ModelConfig, dict_keys: Optional[Set[str]] = None) -> ConfigMatchStats:
    config_keys = set(config.keys())
    if dict_keys is None or (dict_keys_len := len(dict_keys)) == 0:
        return ConfigMatchStats(config_keys, 0.0, 0.0, 0)

    dict_keys = {config.resolve_alias(key) for key in dict_keys}

    matched_keys = dict_keys & config_keys
    state_dict_extra = dict_keys - config_keys

    state_dict_miss = set()
    for k, meta in config.keys().items():
        if k in dict_keys or meta.optional:
            continue
        state_dict_miss.add(k)

    return ConfigMatchStats(matched_keys, len(state_dict_extra) / dict_keys_len, len(state_dict_miss) / len(config.keys()))


def finalize_component_config(candidates: ConfigCandidates) -> ModelConfig:
    if not candidates.stats:
        if candidates.requires_known_config:
            if not candidates.explicit_ids:
                sample = sorted((candidates.common_keys or {}).keys())
                raise RuntimeError(
                    "Model config inference failed: INFER was requested but no known config matches.\n"
                    f"  Keys: {sample}\n"
                    "Fix: ensure inputs correspond to a known config, or use the structural model config."
                )
            if len(candidates.explicit_ids) == 1:
                cfg = next(iter(candidates.explicit_ids))
                raise RuntimeError(
                    "Model config constraint violation: the model config did not match the config requirement of the merge method parameter.\n"
                    f"  Required: {cfg}\n"
                    "Fix: provide checkpoints/inputs matching the requested config, or let config auto-detection determine an appropriate config type."
                )
            raise RuntimeError(
                "Model config constraint violation: incompatible configs were mixed together.\n"
                f"  Requested configs: {sorted(candidates.explicit_ids)}\n"
                "Fix: provide checkpoints/inputs matching the requested config, or let config auto-detection determine an appropriate shared config type."
            )

        keys = candidates.common_keys if candidates.common_keys is not None else {}
        return StructuralModelConfig(keys)

    best_id = min(
        candidates.stats.items(),
        key=lambda kv: kv[1].state_dict_misses
    )[0]
    return model_configs.resolve(best_id)


def check_model_config(state_dict: Iterable[str], config: ModelConfig, check_extra_keys: bool, check_mandatory_keys: bool, state_dict_origin: str):
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
            raise RuntimeError(
                "State dict is missing required keys for the selected model config.\n"
                f"  Origin: {state_dict_origin}\n"
                f"  Missing ({len(missing)}):\n{lines}\n"
                "Fix: choose the correct model config, or provide a valid checkpoint file."
            )


@dataclasses.dataclass
class MergeSpaceCandidates:
    allowed: Optional[Set[MergeSpace]] = None
    _parent: Optional["MergeSpaceCandidates"] = None
    _rank: int = 0

    def find(self) -> "MergeSpaceCandidates":
        if self._parent is None:
            return self
        self._parent = self._parent.find()
        return self._parent

    def constrain(self, constraint: Set[MergeSpace]) -> None:
        self_ref = self.find()
        if self_ref.allowed is None:
            self_ref.allowed = set(constraint)
        else:
            self_ref.allowed.intersection_update(constraint)

    def is_empty(self) -> bool:
        self_ref = self.find()
        return self_ref.allowed is not None and len(self_ref.allowed) == 0

    def __or__(self, other) -> "MergeSpaceCandidates":
        ra = self.find()
        rb = other.find()
        if ra is rb:
            return ra

        if ra._rank < rb._rank:
            ra, rb = rb, ra
        rb._parent = ra
        if ra._rank == rb._rank:
            ra._rank += 1

        if ra.allowed is None:
            ra.allowed = None if rb.allowed is None else set(rb.allowed)
        elif rb.allowed is not None:
            ra.allowed.intersection_update(rb.allowed)

        return ra


@dataclasses.dataclass
class MergeSpacesVisitor(TracingRecipeVisitor):
    candidates: Dict[RecipeNode, MergeSpaceCandidates] = dataclasses.field(default_factory=dict)
    connected_candidates: MergeSpaceCandidates = dataclasses.field(default_factory=MergeSpaceCandidates)

    def visit_literal(self, node: LiteralRecipeNode):
        self._attach_node_to_ms_component(node)
        self.__update_candidates(node.merge_space)

        for v in node.value_dict.values():
            if isinstance(v, RecipeNode):
                v.accept(self)

    def visit_model(self, node: ModelRecipeNode):
        self._attach_node_to_ms_component(node)
        self.__update_candidates(node.merge_space)

    def visit_merge(self, node: MergeRecipeNode):
        self._attach_node_to_ms_component(node)

        input_merge_spaces = node.merge_method.get_input_merge_spaces().as_dict(len(node.bound_args.args))
        param_names = node.merge_method.get_param_names().as_dict(len(node.bound_args.args))
        return_ms = node.merge_method.get_signature().return_annotation.data.merge_space
        if return_ms is None:
            return_ms = node.merge_method.default_merge_space

        symbol_to_candidates: Dict[MergeSpaceSymbol, MergeSpaceCandidates] = {}
        if isinstance(return_ms, MergeSpaceSymbol):
            symbol_to_candidates[return_ms] = self.connected_candidates
            self.connected_candidates.constrain(return_ms.merge_spaces)
        else:
            self.connected_candidates.constrain({return_ms})

        if self.connected_candidates.is_empty():
            raise RuntimeError(
                "Merge-space constraints became unsatisfiable at merge method return.\n"
                f"  Merge method: {node.merge_method.identifier}\n"
                f"  Return merge space: {return_ms}\n"
                "Fix: Pick a merge method/merge space compatible with the connected graph."
            )

        for param_idx, child in (*enumerate(node.bound_args.args), *node.bound_args.kwargs.items()):
            param_ms = input_merge_spaces[param_idx]

            if isinstance(param_ms, MergeSpaceSymbol):
                child_cands = symbol_to_candidates.get(param_ms)
                if child_cands is None:
                    child_cands = MergeSpaceCandidates()
                    symbol_to_candidates[param_ms] = child_cands

                child_cands.constrain(param_ms.merge_spaces)
                if child_cands.is_empty():
                    raise RuntimeError(
                        "Merge-space constraints became unsatisfiable at merge input.\n"
                        f"  Merge method: {node.merge_method.identifier}\n"
                        f"  Input: {param_names[param_idx]}\n"
                        f"  Required by signature: {param_ms}\n"
                        "Fix: update the merge space on the offending input, or use a different merge method that can accept the input."
                    )

                child.accept(dataclasses.replace(self, connected_candidates=child_cands))
            else:
                assert isinstance(param_ms, set), f"Unexpected merge space type: {type(param_ms)}"
                child_cands = MergeSpaceCandidates()
                child_cands.constrain(param_ms)
                if child_cands.is_empty():
                    raise RuntimeError(
                        f"merge space constraints became unsatisfiable at merge node '{node.merge_method.identifier}' "
                        f"(fixed-set input)"
                    )
                child.accept(dataclasses.replace(self, connected_candidates=child_cands))

    def _attach_node_to_ms_component(self, node: RecipeNode) -> None:
        if node in self.candidates:
            self.connected_candidates |= self.candidates[node]
        self.candidates[node] = self.connected_candidates.find()

    def __update_candidates(self, merge_space: Optional[MergeSpace]):
        if merge_space is None:
            return
        before = None if self.connected_candidates.allowed is None else {ms.identifier for ms in self.connected_candidates.allowed}
        self.connected_candidates.constrain({merge_space})
        if self.connected_candidates.is_empty():
            raise RuntimeError(
                "argument merge space conflicts with existing merge-space constraints.\n"
                f"  argument: {merge_space.identifier}\n"
                f"  allowed: {sorted(before)}\n"
                "Fix: remove the conflicting constraint, or ensure all connected nodes share a compatible merge space."
            )


def finalize_component_merge_space(cands: MergeSpaceCandidates, *, origin: str = "") -> MergeSpace:
    prefix = f"{origin}\n" if origin else ""
    if cands.allowed is None:
        all_spaces = tuple(sorted(ms.identifier for ms in merge_spaces.get_all()))
        raise RuntimeError(
            prefix
            + "Merge-space inference failed: no constraints were provided.\n"
            f"  Known merge spaces: {all_spaces}\n"
            "Fix: provide a merge_space_hint, or use a merge method that specifies/propagates merge spaces."
        )

    if len(cands.allowed) == 0:
        raise RuntimeError(prefix + "Merge-space constraints are unsatisfiable (empty intersection).")

    if len(cands.allowed) > 1:
        opts = tuple(sorted(ms.identifier for ms in cands.allowed))
        raise RuntimeError(
            prefix
            + "Merge-space inference is ambiguous: multiple candidates remain.\n"
            f"  Candidates: {opts}\n"
            "Fix: add a merge_space_hint to disambiguate."
        )

    return next(iter(cands.allowed))


@dataclasses.dataclass
class FinalizeVisitor(TracingRecipeVisitor):
    check_extra_keys: bool = False
    check_mandatory_keys: bool = False
    cfg_candidates: Dict[RecipeNode, ConfigCandidates] = dataclasses.field(default_factory=dict)
    ms_candidates: Dict[RecipeNode, MergeSpaceCandidates] = dataclasses.field(default_factory=dict)
    state_dicts: Dict[ModelRecipeNode, SafetensorsMapping] = dataclasses.field(default_factory=dict)
    cfg_candidates_cache: Dict[int, ModelConfig] = dataclasses.field(default_factory=dict)
    ms_candidates_cache: Dict[int, MergeSpace] = dataclasses.field(default_factory=dict)

    def visit_literal(self, node: LiteralRecipeNode):
        cfg, ms = self.__get_cfg_ms(node)
        check_model_config(node.value_dict, cfg, self.check_extra_keys, self.check_mandatory_keys, "<in-memory>")
        value_dict = {k: v.accept(self) if isinstance(v, RecipeNode) else v for k, v in node.value_dict.items()}
        return LiteralRecipeNode(value_dict, cfg, ms)

    def visit_model(self, node: ModelRecipeNode):
        cfg, ms = self.__get_cfg_ms(node)
        sd = self.state_dicts[node]
        check_model_config(sd, cfg, self.check_extra_keys, self.check_mandatory_keys, str(node.path))
        return OpenModelRecipeNode(sd, node.path, cfg, ms)

    def visit_merge(self, node: MergeRecipeNode):
        cfg, ms = self.__get_cfg_ms(node)
        args = tuple(v.accept(self) for v in node.bound_args.args)
        kwargs = {k: v.accept(self) for k, v in node.bound_args.kwargs.items()}
        bound_args = node.merge_method.get_signature().bind(*args, **kwargs)
        res = MergeRecipeNode(node.merge_method, bound_args, cfg, ms)
        return res

    def __get_cfg_ms(self, node: RecipeNode):
        cfg_candidates = self.cfg_candidates[node].find()
        if id(cfg_candidates) not in self.cfg_candidates_cache:
            config = finalize_component_config(cfg_candidates)
            self.cfg_candidates_cache[id(cfg_candidates)] = config
        cfg = self.cfg_candidates_cache[id(cfg_candidates)]

        ms_candidates = self.ms_candidates[node].find()
        if id(ms_candidates) not in self.ms_candidates_cache:
            merge_space = finalize_component_merge_space(ms_candidates, origin=f"At node: {repr(node)}")
            self.ms_candidates_cache[id(ms_candidates)] = merge_space
        ms = self.ms_candidates_cache[id(ms_candidates)]

        return cfg, ms


def _fmt_list(items, limit=12) -> str:
    items = list(items)
    if len(items) <= limit:
        return ", ".join(map(str, items))
    head = ", ".join(map(str, items[:limit]))
    return f"{head}, ... (+{len(items)-limit} more)"
