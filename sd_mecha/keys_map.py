import dataclasses
from collections import OrderedDict
from types import MappingProxyType
from typing import Any, Dict, Generic, Iterator, Mapping, Optional, Set, Tuple, TypeVar, Union
from sd_mecha.extensions import model_configs
from sd_mecha.extensions.model_configs import KeyMetadata, ModelConfig
from sd_mecha.typing_ import ClassObject


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

    def __and__(self, other: "Clause") -> "ReqExpr":
        if other is None:
            return ReqExpr((self,))
        if not isinstance(other, Clause):
            return NotImplemented

        merged: Dict[str, Tuple[str, ...]] = OrderedDict(self.by_param)
        for p, ks in other.by_param.items():
            prev = merged.get(p, ())
            merged[p] = prev + tuple(k for k in ks if k not in prev)

        meta = self._merge_clause_meta(self.meta, other.meta)
        return ReqExpr((Clause(MappingProxyType(merged), meta=meta),))

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
        if other is None:
            return self
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


@dataclasses.dataclass(frozen=True, slots=True)
class KeysAccessor:
    params: Tuple[str, ...]
    config: ModelConfig

    def __call__(self) -> Iterator[str]:
        yield from self.config.keys().keys()

    def items(self) -> Iterator[Tuple[str, KeyMetadata]]:
        yield from self.config.keys().items()

    def __getitem__(self, sel: object) -> ReqExpr:
        keys = _to_tuple(sel, what="input key")
        if not keys:
            return ReqExpr((Clause(MappingProxyType({param: () for param in self.params})),))
        self.validate_keys(keys)
        return ReqExpr((Clause(MappingProxyType({param: keys for param in self.params})),))

    def validate_keys(self, keys: Tuple[str, ...]):
        _ensure_no_dupes(keys, what=f"input keys ({self.config})")
        config_keys = self.config.keys()

        for key in keys:
            key = _norm_one(key, what="input key")
            if key not in config_keys:
                hint = ", ".join(list(config_keys)[:10])
                cfg_id = self.config.identifier
                raise ValueError(
                    f"{self.params}: unknown input key '{key}' (config={cfg_id}). "
                    f"Possible includes: {hint}"
                )


@dataclasses.dataclass(frozen=True, slots=True)
class OutputKeysAccessor:
    config: ModelConfig

    def __call__(self) -> Iterator[str]:
        yield from self.config.keys().keys()

    def items(self) -> Iterator[Tuple[str, KeyMetadata]]:
        yield from self.config.keys().items()


@dataclasses.dataclass(frozen=True, slots=True)
class ParamProxy:
    names: Tuple[str, ...]
    config: ModelConfig
    keys: KeysAccessor = dataclasses.field(init=False, hash=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "keys", KeysAccessor(self.names, self.config))


@dataclasses.dataclass(frozen=True, slots=True)
class OutputProxy:
    config: ModelConfig
    keys: OutputKeysAccessor = dataclasses.field(init=False, hash=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "keys", OutputKeysAccessor(self.config))


T = TypeVar("T")


class KeyMap(Generic[T]):
    def __init__(self, n_to_n_map: MappingProxyType[Tuple[str, ...], T]):
        self.n_to_n_map = n_to_n_map
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
    simple_map: MappingProxyType[str, T]

    def __post_init__(self):
        object.__setattr__(self, "simple_map", MappingProxyType(dict(self.simple_map)))

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


@dataclasses.dataclass(slots=True)
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

    __input_configs: MappingProxyType[str, ModelConfig]
    __output_config: ModelConfig

    __relations: Dict[Tuple[str, ...], KeyRelation] = dataclasses.field(default_factory=dict)
    __out_key_owner: Dict[str, Tuple[str, ...]] = dataclasses.field(default_factory=dict)
    __param_proxies_cache: Dict[str, ParamProxy] = dataclasses.field(default_factory=dict, init=False, hash=False, compare=False)
    __shared_key_accessor_cache: Optional[KeysAccessor] = dataclasses.field(default=None, init=False, hash=False, compare=False)
    __output_proxy_cache: Optional[OutputProxy] = dataclasses.field(default=None, init=False, hash=False, compare=False)
    __shared_config: Optional[ModelConfig] = dataclasses.field(default=None, init=False, hash=False, compare=False)

    def __post_init__(self):
        self.__shared_config = self.__compute_shared_input_config()

    def __getattr__(self, name) -> ParamProxy | KeysAccessor | OutputProxy:
        if name == "keys":
            if self.__shared_config is not None:
                if self.__shared_key_accessor_cache is None:
                    self.__shared_key_accessor_cache = KeysAccessor(tuple(self.__input_configs), self.__shared_config)
                return self.__shared_key_accessor_cache
            else:
                ids = {param: cfg.identifier for param, cfg in self.__input_configs.items()}
                raise AttributeError(
                    "b.keys is only available when all parameters share the same config. "
                    f"Got config identifiers per param: {ids}"
                )

        if name == "return":
            if self.__output_proxy_cache is None:
                self.__output_proxy_cache = OutputProxy(self.__output_config)
            return self.__output_proxy_cache

        if name in self.__input_configs:
            if name not in self.__param_proxies_cache:
                self.__param_proxies_cache[name] = ParamProxy(
                    (name,),
                    self.__input_configs[name]
                )
            return self.__param_proxies_cache[name]

        available = ", ".join(repr(p) for p in self.__input_configs)
        raise AttributeError(
            f"{type(self).__name__} has no parameter or attribute {name!r}. "
            f"This may indicate an invalid input parameter name. "
            f"Available parameters: {available}"
        )

    def __compute_shared_input_config(self) -> Optional[ModelConfig]:
        if not self.__input_configs:
            return model_configs.EMPTY
        cfgs = iter(self.__input_configs.values())
        first = next(cfgs)
        if all(cfg == first for cfg in cfgs):
            return first
        return None

    def _validate_output_key(self, key: str) -> None:
        k = _norm_one(key, what="output key")
        keys = self.__output_config.keys()
        if k not in keys:
            hint = ", ".join(list(keys)[:10])
            cfg_id = getattr(self.__output_config, "identifier", "<unknown>")
            raise ValueError(
                f"Unknown output key '{k}' (output_config={cfg_id}). Possible includes: {hint}"
            )

    def __setitem__(self, out_keys: object, rhs: object) -> None:
        if isinstance(out_keys, type(Ellipsis)):
            outs = tuple(self.__output_config.keys())
        else:
            outs = _to_tuple(out_keys, what="output key")

        if not outs:
            raise ValueError("Output group must contain at least one key.")
        _ensure_no_dupes(outs, what="output keys (within the same output group)")

        for k in outs:
            self._validate_output_key(k)
        for k in outs:
            if k in self.__out_key_owner:
                prev = self.__out_key_owner[k]
                raise ValueError(f"Output key '{k}' is already used in output group {prev}")
        for k in outs:
            self.__out_key_owner[k] = outs

        if isinstance(rhs, Clause):
            expr = ReqExpr((rhs,))
        elif isinstance(rhs, ReqExpr):
            expr = rhs
        else:
            raise ValueError("RHS must be a Clause or ReqExpr.")

        validated_clauses = []
        for clause in expr.clauses:
            inputs_mut: Dict[str, Tuple[str, ...]] = {}
            for p, keys in clause.by_param.items():
                if p not in self.__input_configs:
                    raise ValueError(f"Unknown parameter '{p}' in requirements.")
                self.__getattr__(p).keys.validate_keys(keys)
                inputs_mut[p] = keys
            validated_clauses.append(Clause(MappingProxyType(inputs_mut), meta=clause.meta))

        self.__relations[outs] = KeyRelation(outputs=outs, clauses=tuple(validated_clauses))

    def build(self) -> KeyMap:
        return KeyMap(MappingProxyType(self.__relations))
