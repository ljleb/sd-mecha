import dataclasses
import inspect
from collections import OrderedDict
from types import MappingProxyType
from typing import Any, Callable, Dict, Generic, Iterator, Mapping, Optional, Set, Tuple, TypeVar, Union


@dataclasses.dataclass(frozen=True, slots=True)
class KeyRelation:
    outputs: Tuple[str, ...]
    clauses: Tuple["Clause", ...]

    def __iter__(self):
        yield self.outputs
        yield self.clauses


@dataclasses.dataclass(frozen=True, slots=True)
class RealizedKeyRelation:
    outputs: Tuple[str, ...]
    inputs: Mapping[str, Tuple[str, ...]]
    meta: Any = None

    def __iter__(self):
        yield self.outputs
        yield self.inputs


@dataclasses.dataclass(frozen=True, slots=True)
class Clause:
    by_param: Mapping[str, Tuple[str, ...]]
    meta: Any = None

    def __and__(self, other: "Clause") -> "Clause":
        if not isinstance(other, Clause):
            return NotImplemented

        merged: Dict[str, Tuple[str, ...]] = OrderedDict(self.by_param)
        for p, ks in other.by_param.items():
            if p in merged:
                raise ValueError(
                    "Each parameter may appear at most once within one conjunction."
                )
            merged[p] = ks

        meta = self._merge_clause_meta(self.meta, other.meta)
        return Clause(MappingProxyType(merged), meta=meta)

    def __rand__(self, other: "Clause") -> "Clause":
        return self & other

    def __or__(self, other: Optional[Union["Clause", "ReqExpr"]]) -> "ReqExpr":
        if other is None:
            return ReqExpr((self,))
        if isinstance(other, Clause):
            return ReqExpr((self, other))
        if isinstance(other, ReqExpr):
            return ReqExpr((self, *other.clauses))
        return NotImplemented

    def __ror__(self, other: Optional[Union["Clause", "ReqExpr"]]) -> "ReqExpr":
        return self | other

    def __matmul__(self, meta: Any) -> "Clause":
        if self.meta is not None:
            raise ValueError("Clause metadata was already assigned.")
        return Clause(self.by_param, meta=meta)

    @staticmethod
    def _merge_clause_meta(a: Any, b: Any) -> Any:
        if a is None:
            return b
        if b is None:
            return a
        return a | b


@dataclasses.dataclass(frozen=True, slots=True)
class ReqExpr:
    clauses: Tuple[Clause, ...]

    def __and__(self, other: object) -> "ReqExpr":
        if isinstance(other, Clause):
            other = ReqExpr((other,))
        if not isinstance(other, ReqExpr):
            return NotImplemented

        out = []
        for a in self.clauses:
            for b in other.clauses:
                merged: Dict[str, Tuple[str, ...]] = OrderedDict()

                for src in (a.by_param, b.by_param):
                    for p, ks in src.items():
                        prev = merged.get(p, ())
                        merged[p] = prev + tuple(k for k in ks if k not in prev)

                meta = self._merge_meta(a.meta, b.meta)

                out.append(Clause(MappingProxyType(merged), meta=meta))
        return ReqExpr(tuple(out))

    @staticmethod
    def _merge_meta(a, b):
        if a is None:
            return b
        if b is None:
            return a
        try:
            return a | b
        except TypeError as e:
            raise ValueError(
                f"Cannot compose clause metadata of types "
                f"{type(a).__name__} and {type(b).__name__}"
            ) from e

    def __rand__(self, other: object) -> "ReqExpr":
        return self & other

    def __or__(self, other: object) -> "ReqExpr":
        if other is None:
            return self
        if isinstance(other, Clause):
            return ReqExpr((*self.clauses, other))
        if isinstance(other, ReqExpr):
            return ReqExpr((*self.clauses, *other.clauses))
        return NotImplemented

    def __ror__(self, other: object) -> "ReqExpr":
        return self | other

    def __matmul__(self, meta: Any) -> "ReqExpr":
        return ReqExpr(tuple(clause @ meta for clause in self.clauses))


class ClassObject(type):
    _type_cache = {}

    def __new__(mcls, name, bases, namespace, **kwargs):
        bad_methods = [
            n
            for n, value in namespace.items()
            if inspect.isfunction(value)
        ]
        if bad_methods:
            names = ", ".join(sorted(bad_methods))
            raise TypeError(
                f"{name} is a class object type and cannot define instance methods: {names}"
            )

        bases = mcls._reduce_bases(bases)
        return super().__new__(mcls, name, bases, namespace, **kwargs)

    @staticmethod
    def _reduce_bases(bases):
        bases = tuple(dict.fromkeys(bases))
        return tuple(
            base
            for base in bases
            if not any(base is not other and issubclass(other, base) for other in bases)
        )

    @staticmethod
    def _find_bound_hook(cls, name: str):
        for base in cls.__mro__:
            hook = base.__dict__.get(name)
            if hook is not None:
                return hook.__get__(None, cls)
        return None

    @classmethod
    def compose_type(mcls, bases):
        bases = mcls._reduce_bases(bases)

        if len(bases) == 1:
            return bases[0]

        cached = mcls._type_cache.get(bases)
        if cached is not None:
            return cached

        name = "__".join(base.__name__ for base in bases)
        composed_cls = mcls(name, bases, {})
        mcls._type_cache[bases] = composed_cls
        return composed_cls

    def __call__(cls, *args, **kwargs):
        hook = ClassObject._find_bound_hook(cls, "__call__")
        if hook is not None:
            return hook(*args, **kwargs)

        raise TypeError(
            f"{cls.__name__} is a class object type and should not be instantiated. "
            f"Use `{cls.__name__}` directly instead as a regular object."
        )

    def __or__(cls, other):
        hook = ClassObject._find_bound_hook(cls, "__or__")
        if hook is not None:
            return hook(other)
        return NotImplemented

    def __ror__(cls, other):
        hook = ClassObject._find_bound_hook(cls, "__ror__")
        if hook is not None:
            return hook(other)
        return NotImplemented


class ComposeObject(metaclass=ClassObject):
    @classmethod
    def plan_bases(cls):
        return (cls,)

    @classmethod
    def __or__(cls: ClassObject, other):
        if not (isinstance(other, type) and issubclass(other, ComposeObject)):
            return NotImplemented

        bases = (cls, *((other,) if other != cls else ()))
        return type(cls).compose_type(bases)

    @classmethod
    def __ror__(cls, other):
        if not (isinstance(other, type) and issubclass(other, ComposeObject)):
            return NotImplemented
        return other | cls


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
        out: list[str] = []
        for v in x:
            if not isinstance(v, str):
                raise TypeError
            out.append(_norm_one(v, what=what))
        return tuple(out)
    except TypeError:
        raise ValueError(f"{what} must be a str or iterable[str], got {type(x).__name__}")


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

    def __getitem__(self, sel: object) -> Clause:
        keys = _to_tuple(sel, what="input key")
        if not keys:
            return Clause(MappingProxyType({self._p: ()}))
        _ensure_no_dupes(keys, what=f"input keys (param '{self._p}')")
        for k in keys:
            self._b._validate_input_key(self._p, k)
        return Clause(MappingProxyType({self._p: keys}))


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

    def __getitem__(self, sel: object) -> Clause:
        self._require_enabled()
        keys = _to_tuple(sel, what="input key")
        if not keys:
            return Clause(MappingProxyType({p: () for p in self._b.params}))
        _ensure_no_dupes(keys, what="input keys (same-across-all-params selection)")

        for k in keys:
            if k not in self._b._shared_input_keyset:
                hint = ", ".join(list(self._b._shared_input_keyset)[:10])
                raise ValueError(f"Unknown shared input key '{k}'. Shared keys include: {hint}")

        by = {p: keys for p in self._b.params}
        return Clause(MappingProxyType(by))


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


T = TypeVar("T")


class KeyMap(Generic[T]):
    def __init__(self, n_to_n_map: Mapping[Tuple[str, ...], T]):
        self.n_to_n_map = MappingProxyType(n_to_n_map)
        self.simple_map = MappingProxyType({
            k: g
            for ks, g in self.n_to_n_map.items()
            for k in ks
        })

    def __iter__(self):
        return iter(self.n_to_n_map.values())

    def __getitem__(self, k: str | Tuple[str, ...]) -> T:
        if isinstance(k, str):
            return self.simple_map[k]
        else:
            return self.n_to_n_map[k]

    def get(self, k, default=None):
        if isinstance(k, str):
            return self.simple_map.get(k, default)
        else:
            return self.n_to_n_map.get(k, default)

    def __contains__(self, k: str | Tuple[str, ...]) -> bool:
        if isinstance(k, str):
            return k in self.simple_map
        else:
            return k in self.n_to_n_map


@dataclasses.dataclass(frozen=True, slots=True)
class ActiveKeyMap(Generic[T]):
    simple_map: Mapping[str, T]

    def __init__(self, simple_map: Mapping[str, T]):
        object.__setattr__(self, "simple_map", MappingProxyType(dict(simple_map)))

    def __getitem__(self, k: str) -> T:
        return self.simple_map[k]

    def get(self, k: str, default=None):
        return self.simple_map.get(k, default)

    def __contains__(self, k: str) -> bool:
        return k in self.simple_map

    def __iter__(self) -> Iterator[T]:
        seen = set()
        for rel in self.simple_map.values():
            rid = id(rel)
            if rid in seen:
                continue
            seen.add(rid)
            yield rel


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
        self.param_proxies = {}

        self.out = OutputProxy(self)

        self._all_inputs_share_config = self._compute_all_inputs_share_config()
        self.keys = AllKeysAccessor(self)

        self._shared_input_keyset: Set[str] = set()
        if self._all_inputs_share_config:
            first_cfg = self.input_configs[self.params[0]]
            self._shared_input_keyset = set(first_cfg.keys().keys())

        self._relations: Dict[Tuple[str, ...], KeyRelation] = {}
        self._out_key_owner: Dict[str, Tuple[str, ...]] = {}

    def __getattr__(self, param):
        if param in self.input_configs:
            if param not in self.param_proxies:
                self.param_proxies[param] = ParamProxy(self, param)
            return self.param_proxies[param]

        available = ", ".join(repr(p) for p in self.params)
        raise AttributeError(
            f"{type(self).__name__} has no parameter or attribute {param!r}. "
            f"This may indicate an invalid input parameter name. "
            f"Available parameters: {available}"
        )

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
        if isinstance(out_keys, type(Ellipsis)):
            outs = tuple(self._output_keys())
        else:
            outs = _to_tuple(out_keys, what="output key")

        if not outs:
            raise ValueError("Output group must contain at least one key.")
        _ensure_no_dupes(outs, what="output keys (within the same output group)")

        for k in outs:
            self._validate_output_key(k)
        for k in outs:
            if k in self._out_key_owner:
                prev = self._out_key_owner[k]
                raise ValueError(f"Output key '{k}' is already used in output group {prev}")
        for k in outs:
            self._out_key_owner[k] = outs

        if isinstance(rhs, Clause):
            expr = ReqExpr((rhs,))
        elif isinstance(rhs, ReqExpr):
            expr = rhs
        else:
            raise ValueError("RHS must be a Clause or ReqExpr.")

        validated_clauses = []
        for clause in expr.clauses:
            inputs_mut: Dict[str, Tuple[str, ...]] = {}
            for p, ks in clause.by_param.items():
                if p not in self.input_configs:
                    raise ValueError(f"Unknown parameter '{p}' in requirements.")
                _ensure_no_dupes(ks, what=f"input keys (param '{p}')")
                for k in ks:
                    self._validate_input_key(p, k)
                inputs_mut[p] = ks
            validated_clauses.append(Clause(MappingProxyType(inputs_mut), meta=clause.meta))

        self._relations[outs] = KeyRelation(outputs=outs, clauses=tuple(validated_clauses))

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
