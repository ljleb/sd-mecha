"""
Simple KeyMapBuilder DSL using *existing* ModelConfig objects.

Assumptions about ModelConfig (already defined in your library; imported elsewhere):
- cfg_a == cfg_b  -> True iff they share the same *set of keys* (transitive)
- cfg.identifier  -> unique string identifying the keyset
- cfg.keys()      -> dict[str, Any] mapping key -> properties (value metadata)

What this script provides:
- Users implement:      def map_keys(b: KeyMapBuilder) -> None
- They declare relations:
    b["out1", "out2"] = b.param1.keys["in_a", "in_b"] | b.param2.keys["in_x"]
    b["out"] = (b.param1.keys["in_a"] | b.param2.keys["in_x"]) @ {"meta": ...}

- Output keys are validated against output_config.
- Input keys are validated against each param's config.
- STRICT output key uniqueness:
    * no duplicates within a single output group
    * no output key may appear in more than one group
- STRICT input key uniqueness:
    * no duplicates within a param selection
    * combining with | forbids overlap for the same param

- Convenience:
    * b.param.keys() iterates available keys for that param (in cfg.keys() order)
    * b.param.keys.items() iterates (key, props) for filtering with metadata
    * b.keys() / b.keys[...] is only enabled if ALL input configs are equal (same keyset)
      (checked via cfg equality); otherwise it errors.
    * output keys are only accepted if they exist in output_config.keys()

Built result:
- keys_map = b.build()
- for out_keys, rel in keys_map.items():
      out_keys, input_keys = rel
      rel.meta
where:
- out_keys is tuple[str, ...] (order preserved)
- input_keys is a read-only mapping param -> tuple[str, ...] (order preserved)
"""
from collections import OrderedDict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Set, Tuple
from collections.abc import Mapping as ABCMapping


@dataclass(frozen=True, slots=True)
class KeyRelation:
    """
    One declared relation:
      outputs -> inputs (+ optional metadata)

    Unpacking yields ONLY (outputs, inputs):
      out_keys, in_keys = relation
    """
    outputs: Tuple[str, ...]
    inputs: Mapping[str, Tuple[str, ...]]
    meta: Any = None

    def __iter__(self) -> Iterator[object]:
        yield self.outputs
        yield self.inputs


@dataclass(frozen=True, slots=True)
class Req:
    """
    Requirements:
      param -> tuple(keys...)  (order preserved; no duplicates allowed)
    """
    by_param: Mapping[str, Tuple[str, ...]]

    def __or__(self, other: "Req") -> "Req":
        merged: Dict[str, Tuple[str, ...]] = OrderedDict(self.by_param)

        for p, ks in other.by_param.items():
            if p in merged:
                raise ValueError("Each parameter should be included at most once for each set of output key.")
            merged[p] = ks

        return Req(MappingProxyType(merged))

    def __matmul__(self, meta: Any) -> "AnnotatedReq":
        return AnnotatedReq(self, meta)


@dataclass(frozen=True, slots=True)
class AnnotatedReq:
    req: Req
    meta: Any

    def __or__(self, other: object) -> "AnnotatedReq":
        raise ValueError("Do not combine after annotating. Use: (req1 | req2) @ meta")

    def __ror__(self, other: object) -> "AnnotatedReq":
        raise ValueError("Do not combine after annotating. Use: (req1 | req2) @ meta")


def _norm_one(s: str, *, what: str) -> str:
    if not isinstance(s, str):
        raise ValueError(f"{what} must be a str, got {type(s).__name__}")
    return s


def _to_tuple(x: object, *, what: str) -> Tuple[str, ...]:
    """
    Convert selection-like input into tuple[str,...] preserving order.
    Accepts:
      - "k" -> ("k",)
      - ("k1","k2") / ["k1","k2"] / any iterable[str]
    """
    if isinstance(x, str):
        return (_norm_one(x, what=what),)

    try:
        it = iter(x)
    except TypeError:
        raise ValueError(f"{what} must be a str or iterable[str], got {type(x).__name__}")

    out: list[str] = []
    for v in it:
        out.append(_norm_one(v, what=what))
    return tuple(out)


def _ensure_no_dupes(seq: Tuple[str, ...], *, what: str) -> None:
    seen: Set[str] = set()
    dupes: list[str] = []
    for s in seq:
        if s in seen:
            dupes.append(s)
        else:
            seen.add(s)
    if dupes:
        raise ValueError(f"Duplicate {what}: {dupes}")


class ParamKeysAccessor:
    """
    b.param.keys()       -> iterate keys (strings) for that param
    b.param.keys.items() -> iterate (key, props) from cfg.keys()
    b.param.keys[...]    -> select tuple of keys for that param
    """
    def __init__(self, builder: "KeyMapBuilder", param: str):
        self._b = builder
        self._p = param

    def __call__(self) -> Iterator[str]:
        yield from self._b._input_keys_for(self._p)

    def items(self) -> Iterator[Tuple[str, Any]]:
        yield from self._b._input_items_for(self._p)

    def __getitem__(self, sel: object) -> Req:
        keys = _to_tuple(sel, what="input key")
        if not keys:
            raise ValueError("Empty key selection")
        _ensure_no_dupes(keys, what=f"input keys (param '{self._p}')")
        for k in keys:
            self._b._validate_input_key(self._p, k)
        return Req(MappingProxyType({self._p: keys}))


class AllKeysAccessor:
    """
    Enabled only if all input configs are equal (same keyset).
    b.keys()       -> iterate keys shared by all params
    b.keys.items() -> iterate (key, props) from the shared config
    b.keys[...]    -> select same key tuple across ALL params (validated)
    """
    def __init__(self, builder: "KeyMapBuilder"):
        self._b = builder

    def _require_enabled(self) -> None:
        if not self._b._all_inputs_share_config:
            ids = {p: self._b.input_configs[p].identifier for p in self._b.params}
            raise ValueError(
                "b.keys() is only available when all input configs share the same keyset. "
                f"Got config identifiers per param: {ids}"
            )

    def __call__(self) -> Iterator[str]:
        self._require_enabled()
        yield from self._b._shared_input_keys()

    def items(self) -> Iterator[Tuple[str, Any]]:
        self._require_enabled()
        yield from self._b._shared_input_items()

    def __getitem__(self, sel: object) -> Req:
        self._require_enabled()
        keys = _to_tuple(sel, what="input key")
        if not keys:
            raise ValueError("Empty key selection")
        _ensure_no_dupes(keys, what="input keys (same-across-all-params selection)")

        for k in keys:
            if k not in self._b._shared_input_keyset:
                hint = ", ".join(list(self._b._shared_input_keyset)[:10])
                raise ValueError(f"Unknown shared input key '{k}'. Shared keys include: {hint}")

        by: Dict[str, Tuple[str, ...]] = {p: keys for p in self._b.params}
        return Req(MappingProxyType(by))


class OutputKeysAccessor:
    """
    b.out.keys()       -> iterate output keys
    b.out.keys.items() -> iterate (key, props) for output config
    """
    def __init__(self, builder: "KeyMapBuilder"):
        self._b = builder

    def __call__(self) -> Iterator[str]:
        yield from self._b._output_keys()

    def items(self) -> Iterator[Tuple[str, Any]]:
        yield from self._b._output_items()


class ParamProxy:
    def __init__(self, builder: "KeyMapBuilder", name: str):
        self.name = name
        self.keys = ParamKeysAccessor(builder, name)


class OutputProxy:
    def __init__(self, builder: "KeyMapBuilder"):
        self.keys = OutputKeysAccessor(builder)


class KeyMap:
    """
    Mapping-like container: output-group -> KeyRelation
    Keys are output-groups: tuple[str, ...]
    """
    def __init__(self, n_to_n_map: Mapping[Tuple[str, ...], KeyRelation]):
        self.n_to_n_map = OrderedDict(n_to_n_map)
        self.simple_map = {
            k: g
            for ks, g in self.n_to_n_map.items()
            for k in ks
        }

    def __getitem__(self, k: str | Tuple[str, ...]) -> KeyRelation:
        if isinstance(k, str):
            return self.simple_map[k]
        else:
            return self.n_to_n_map[k]

    def __contains__(self, k: str | Tuple[str, ...]) -> bool:
        if isinstance(k, str):
            return k in self.simple_map
        else:
            return k in self.n_to_n_map


class IdentityDict(ABCMapping):
    def __init__(self, value):
        self.value = value

    def __getitem__(self, __key):
        return self.value

    def __len__(self):
        raise RuntimeError("Invalid operation.")

    def __iter__(self):
        raise RuntimeError("Invalid operation.")


class KeyMapBuilder:
    """
    Construct with:
      input_configs:  param_name -> ModelConfig
      output_config:  ModelConfig

    The builder validates:
      - output keys exist in output_config.keys()
      - input keys exist in each input_config.keys()
      - output key uniqueness across entire spec (strict)
    """

    def __init__(self, input_configs: Mapping[str, Optional[Any]], output_config: Any):
        if not input_configs:
            raise ValueError("At least one input parameter config is required.")
        self.input_configs: Dict[str, Any] = OrderedDict(input_configs)
        self.output_config: Any = output_config
        self.params: Tuple[str, ...] = tuple(self.input_configs.keys())

        for p in self.params:
            setattr(self, p, ParamProxy(self, p))

        self.out = OutputProxy(self)

        self._all_inputs_share_config = self._compute_all_inputs_share_config()
        self.keys = AllKeysAccessor(self)

        self._shared_input_keyset: Set[str] = set()
        if self._all_inputs_share_config:
            first_cfg = self.input_configs[self.params[0]]
            self._shared_input_keyset = set(first_cfg.keys().keys())

        self._relations: Dict[Tuple[str, ...], KeyRelation] = {}
        self._out_key_owner: Dict[str, Tuple[str, ...]] = {}

    def _compute_all_inputs_share_config(self) -> bool:
        cfgs = [self.input_configs[p] for p in self.params]
        first = cfgs[0]
        return all(cfg == first for cfg in cfgs[1:])

    def _input_keydict_for(self, param: str) -> Dict[str, Any]:
        cfg = self.input_configs[param]
        kd = cfg.keys()
        if not isinstance(kd, dict):
            kd = OrderedDict(kd)
        return kd

    def _input_keys_for(self, param: str) -> Iterator[str]:
        yield from self._input_keydict_for(param).keys()

    def _input_items_for(self, param: str) -> Iterator[Tuple[str, Any]]:
        yield from self._input_keydict_for(param).items()

    def _shared_input_keys(self) -> Iterator[str]:
        first_cfg = self.input_configs[self.params[0]]
        yield from first_cfg.keys().keys()

    def _shared_input_items(self) -> Iterator[Tuple[str, Any]]:
        first_cfg = self.input_configs[self.params[0]]
        yield from first_cfg.keys().items()

    def _output_keydict(self) -> Dict[str, Any]:
        kd = self.output_config.keys()
        if not isinstance(kd, dict):
            kd = OrderedDict(kd)
        return kd

    def _output_keys(self) -> Iterator[str]:
        yield from self._output_keydict().keys()

    def _output_items(self) -> Iterator[Tuple[str, Any]]:
        yield from self._output_keydict().items()

    # ----- validation -----

    def _validate_input_key(self, param: str, key: str) -> None:
        k = _norm_one(key, what="input key")
        kd = self._input_keydict_for(param)
        if k not in kd:
            hint = ", ".join(list(kd.keys())[:10])
            cfg_id = getattr(self.input_configs[param], "identifier", "<unknown>")
            raise ValueError(
                f"{param}: unknown input key '{k}' (config={cfg_id}). "
                f"Possible includes: {hint}"
            )

    def _validate_output_key(self, key: str) -> None:
        k = _norm_one(key, what="output key")
        kd = self._output_keydict()
        if k not in kd:
            hint = ", ".join(list(kd.keys())[:10])
            cfg_id = getattr(self.output_config, "identifier", "<unknown>")
            raise ValueError(
                f"Unknown output key '{k}' (output_config={cfg_id}). Possible includes: {hint}"
            )

    def __setitem__(self, out_keys: object, rhs: object) -> None:
        # Normalize outputs, preserve order, STRICTLY forbid duplicates
        outs = _to_tuple(out_keys, what="output key")
        if not outs:
            raise ValueError("Output group must contain at least one key.")
        _ensure_no_dupes(outs, what="output keys (within the same output group)")

        # Validate output keys exist
        for k in outs:
            self._validate_output_key(k)

        # Enforce: an output key can appear in only one group overall
        for k in outs:
            if k in self._out_key_owner:
                prev = self._out_key_owner[k]
                raise ValueError(f"Output key '{k}' is already used in output group {prev}")
        for k in outs:
            self._out_key_owner[k] = outs

        # Normalize RHS into (Req, meta)
        if isinstance(rhs, AnnotatedReq):
            req = rhs.req
            meta = rhs.meta
        elif isinstance(rhs, Req):
            req = rhs
            meta = None
        else:
            raise ValueError("RHS must be a Req or (Req @ meta).")

        # Validate inputs and freeze mapping (param order preserved as provided in req.by_param)
        inputs_mut: Dict[str, Tuple[str, ...]] = {}
        for p, ks in req.by_param.items():
            if p not in self.input_configs:
                raise ValueError(f"Unknown parameter '{p}' in requirements.")

            _ensure_no_dupes(ks, what=f"input keys (param '{p}')")
            for k in ks:
                self._validate_input_key(p, k)

            inputs_mut[p] = ks

        rel = KeyRelation(outputs=outs, inputs=MappingProxyType(inputs_mut), meta=meta)
        self._relations[outs] = rel

    def build(self) -> KeyMap:
        return KeyMap(self._relations)


def build_keys_map(
    map_keys: Callable[[KeyMapBuilder], None],
    input_configs: Mapping[str, Any],
    output_config: Any,
) -> KeyMap:
    b = KeyMapBuilder(input_configs=input_configs, output_config=output_config)
    map_keys(b)
    return b.build()
