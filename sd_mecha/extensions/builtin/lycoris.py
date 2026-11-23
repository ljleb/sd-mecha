from __future__ import annotations
import abc
import dataclasses
import warnings
import fuzzywuzzy.process
import torch
from collections import defaultdict
from typing import Callable, ClassVar, Iterable, Mapping, Dict, Optional, Tuple
from sd_mecha.extensions import merge_methods, model_configs
from .merge_methods.kronecker import extract_lokr
from sd_mecha.extensions.merge_methods import merge_method, MergeMethod, StateDict, Parameter, Return
from sd_mecha.extensions.model_configs import (
    ModelComponent, StateDictKey, ModelConfig, ModelConfigImpl,
    LazyModelConfigBase, KeyMetadata,
)
from sd_mecha.keys_map import KeyMapBuilder, ComposeObject, KeysAccessor, ReqExpr
from .merge_methods.svd import svd_lowrank
from sd_mecha.typing_ import ClassObject, subclasses


__all__ = [
    "lora",
    "lokr",
    "norm",
    "apply",
]


def _register_all_lycoris_configs():
    algorithms = subclasses(LycorisAlgorithm)
    for lyco_config_id, lyco_prefix in (
        ("lycoris", "lycoris"),
        ("kohya", "lora"),
    ):
        for base_config_id in (
            "sdxl-kohya",
            "sdxl-kohya_but_diffusers",
            "sd1-kohya",
        ):
            base_config = model_configs.resolve(base_config_id)
            lyco_config = LycorisModelConfig(
                base_config,
                lyco_config_id,
                lyco_prefix,
                algorithms,
            )
            model_configs.register_aux(lyco_config)

            define_application(lyco_config, algorithms)
            for algo in algorithms:
                algo.implement_extract_method(lyco_config, base_config)

            define_conversions(lyco_config, algorithms)


@merge_method(identifier="apply_lycoris", is_interface=True)
def apply(
    base: Parameter(torch.Tensor, "weight"),
    lyco: Parameter(torch.Tensor, "weight"),
) -> Return(torch.Tensor, "weight"):
    ...


def define_application(lyco_config: LycorisModelConfig, algorithms: Iterable[type[LycorisAlgorithm]]):
    lyco_config_id = lyco_config.identifier
    base_config = lyco_config.base_config

    @merge_method(
        identifier=f"apply_lycoris_{lyco_config_id}",
        implements=apply,
        globals=globals(),
        locals=locals(),
    )
    class ApplyLycorisToBase:
        @staticmethod
        def map_keys(b: KeyMapBuilder):
            for base_key in base_config.keys():
                lyco_inputs = None

                for algo in algorithms:
                    algo_inputs = algo.build_input_keys(b.lyco.keys, lyco_config, base_key)
                    if algo_inputs is None:
                        continue

                    lyco_inputs |= algo_inputs

                b[base_key] = (lyco_inputs | b.base.keys[()]) & b.base.keys[base_key]

        def __call__(
            self,
            base: Parameter(StateDict[torch.Tensor], "weight", base_config),
            lyco: Parameter(StateDict[torch.Tensor], "weight", lyco_config),
            **kwargs,
        ) -> Return(torch.Tensor, "weight", base_config):
            base_key = kwargs["key"]
            apply_fn = kwargs["key_relation"].meta
            if apply_fn is None:
                return base[base_key]

            key_prefix = next(iter(lyco.keys())).split(".")[0]
            sd_helper = StateDictKeyHelper(base, lyco, base_key, key_prefix)
            target_shape = base_config.keys()[base_key].shape
            return apply_fn(sd_helper, target_shape)


def define_conversions(lyco_config: LycorisModelConfig, algorithms: Iterable[type[LycorisAlgorithm]]):
    lyco_config_id = lyco_config.identifier
    base_config = lyco_config.base_config
    base_config_id = base_config.identifier

    @merge_method(
        identifier=f"convert_'{lyco_config_id}'_to_base",
        is_conversion=True,
        globals=globals(),
        locals=locals(),
    )
    class LycorisToBase:
        @staticmethod
        def map_keys(b: KeyMapBuilder):
            warnings.warn("converting lycoris to base is deprecated, consider using sd_mecha.apply_lycoris instead.")
            for base_key in base_config.keys():
                lyco_inputs = None

                for algo in algorithms:
                    if not algo.supports_delta_conversion:
                        continue

                    algo_inputs = algo.build_input_keys(b.keys, lyco_config, base_key)
                    if algo_inputs is None:
                        continue

                    lyco_inputs |= algo_inputs

                b[base_key] = (lyco_inputs | b.base.keys[()]) & b.base.keys[base_key]

        def __call__(
            self,
            lyco: Parameter(StateDict[torch.Tensor], "weight", lyco_config_id),
            **kwargs,
        ) -> Return(torch.Tensor, "delta", base_config_id):
            base_key = kwargs["key"]
            apply_fn = kwargs["key_relation"].meta

            base = {base_key: 0}
            key_prefix = next(iter(lyco.keys())).split(".")[0]
            sd_helper = StateDictKeyHelper(base, lyco, base_key, key_prefix)
            target_shape = base_config.keys()[base_key].shape
            return apply_fn(sd_helper, target_shape)


class LycorisModelConfig(LazyModelConfigBase):
    def __init__(self, base_config: ModelConfig, lycoris_identifier: str, prefix: str, algorithms: Iterable[type[LycorisAlgorithm]]):
        super().__init__()
        self.base_config = base_config
        self.lycoris_identifier = lycoris_identifier
        self.prefix = prefix
        self.algorithms = algorithms

    @property
    def identifier(self) -> str:
        return f"{self.base_config.identifier}_{self.lycoris_identifier}_lora"

    def create_config(self):
        identifier = self.identifier
        components = {
            k: ModelComponent(_to_lycoris_keys(component.keys(), self.algorithms, self.prefix))
            for k, component in self.base_config.components().items()
        }
        return ModelConfigImpl(identifier, components)

    def to_lycoris_keys(self, key: StateDictKey, algos: Iterable[type[LycorisAlgorithm]] = None) -> Dict[StateDictKey, KeyMetadata]:
        return _to_lycoris_keys({key: self.base_config.keys()[key]}, algos if algos is not None else self.algorithms, self.prefix)


def _to_lycoris_keys(
    base_keys: Mapping[StateDictKey, KeyMetadata],
    algorithms: Iterable[type[LycorisAlgorithm]],
    prefix: str,
) -> Dict[StateDictKey, KeyMetadata]:
    lyco_keys = {}

    for base_key, meta in base_keys.items():
        for algo in algorithms:
            if not algo.targets_key(base_key, meta) or not getattr(meta.dtype, "is_floating_point", True):
                continue

            algo_keys = algo.convert_key(base_key, meta)
            for algo_key, algo_val in algo_keys.items():
                lyco_keys[f"{prefix}_{algo_key}"] = algo_val

    return lyco_keys


_algos_by_name: Dict[str, type[LycorisAlgorithm]] = {}


@dataclasses.dataclass(frozen=True)
class LycorisAlgorithm(abc.ABC, metaclass=ClassObject):
    suffixes: ClassVar[Tuple[str, ...]]
    extract: ClassVar[Callable]
    extract_method: ClassVar[MergeMethod]
    supports_delta_conversion: ClassVar[bool] = False

    @classmethod
    def __init_subclass__(cls, **kwargs):
        global _algos_by_name
        cls.extract_method = merge_method(cls.extract, identifier=extraction_interface_name(cls.__name__), is_interface=True)
        _algos_by_name[cls.__name__] = cls

    @staticmethod
    def targets_key(key: str, meta: KeyMetadata):
        bias = key.endswith("bias")
        shape_at_least_2d = meta.shape is not None and len(meta.shape) >= 2
        return not bias and shape_at_least_2d

    @classmethod
    def convert_key(cls, base_key: str, meta: KeyMetadata) -> Dict[str, KeyMetadata]:
        res = {}

        stem = cls.stem(base_key)
        stem_aliases = tuple(cls.stem(alias) for alias in meta.aliases)
        for suffix in cls.suffixes:
            lyco_key = f"{stem}.{suffix}"
            lyco_aliases = tuple(f"{stem_alias}.{suffix}" for stem_alias in stem_aliases)
            res[lyco_key] = dataclasses.replace(
                meta,
                shape=[] if suffix == "alpha" else None,
                aliases=lyco_aliases,
                optional=True,
            )

        return res

    @classmethod
    def stem(cls, key: str) -> str:
        parts = key.split(".")
        if parts[-1] in ("weight", "bias"):
            parts = parts[:-1]
        stem = "_".join(parts)
        return stem

    @classmethod
    @abc.abstractmethod
    def build_input_keys(cls, builder: KeysAccessor, lyco_config: LycorisModelConfig, base_key: str) -> Optional[ReqExpr]:
        ...

    @classmethod
    @abc.abstractmethod
    def implement_extract_method(cls, lyco_config: LycorisModelConfig, base_config: ModelConfig):
        ...


def extraction_interface_name(algo_name: str):
    return f"extract_{algo_name}"


def extraction_implementation_name(algo_name: str, lyco_config: LycorisModelConfig):
    return f"{extraction_interface_name(algo_name)}_{lyco_config.identifier}"


def resolve(identifier: str) -> MergeMethod:
    if identifier in _algos_by_name:
        return merge_methods.resolve(extraction_interface_name(identifier))

    suggestions = fuzzywuzzy.process.extractOne(identifier, _algos_by_name.keys())
    postfix = ""
    if suggestions is not None:
        postfix = f". Nearest match is '{suggestions[0]}'"
    raise KeyError(f"unknown lycoris algorithm: {identifier}{postfix}")


class lora(LycorisAlgorithm):
    suffixes = (
        "lora_up.weight",
        "lora_mid.weight",
        "lora_down.weight",
        "alpha",
    )
    supports_delta_conversion = True

    @classmethod
    def build_input_keys(cls, keys: KeysAccessor, lyco_config: LycorisModelConfig, base_key: str) -> Optional[ReqExpr]:
        lyco_keys = lyco_config.to_lycoris_keys(base_key, algos=(cls,))
        if not lyco_keys:
            return None

        up, mid, down, alpha = lyco_keys
        return (
            keys[up, down] @ cls.apply &
            (keys[mid] @ cls.apply_mid | keys[()]) &
            (keys[alpha] @ cls.apply_alpha | keys[()])
        )

    class apply(ComposeObject):
        @classmethod
        def __call__(cls, state_dict: StateDictKeyHelper, target_shape: torch.Size) -> torch.Tensor:
            scale, down_weight = cls.get_scale_down(state_dict)
            delta = cls.build(state_dict, down_weight)

            delta = delta * scale
            delta = delta.reshape(target_shape)

            base = state_dict.get_base_tensor()
            return base + delta

        @staticmethod
        def get_scale_down(sd: StateDictKeyHelper) -> Tuple[torch.Tensor | float, torch.Tensor]:
            down = sd.get_lyco_tensor(lora.suffixes[2])
            alpha = 1
            return alpha, down

        @staticmethod
        def build(sd: StateDictKeyHelper, down: torch.Tensor) -> torch.Tensor:
            up = sd.get_lyco_tensor(lora.suffixes[0])
            delta = up.view(up.size(0), -1) @ down.view(down.size(0), -1)
            return delta

    class apply_mid(apply):
        @staticmethod
        def build(sd: StateDictKeyHelper, down: torch.Tensor) -> torch.Tensor:
            mid = sd.get_lyco_tensor(lora.suffixes[1])
            up = sd.get_lyco_tensor(lora.suffixes[0])
            wa = up.view(up.size(0), -1).transpose(0, 1)
            wb = down.view(down.size(0), -1)
            delta = rebuild_tucker(mid, wa, wb)
            return delta

    class apply_alpha(apply):
        @staticmethod
        def get_scale_down(sd: StateDictKeyHelper) -> Tuple[torch.Tensor | float, torch.Tensor]:
            alpha = sd.get_lyco_tensor(lora.suffixes[3])
            down = sd.get_lyco_tensor(lora.suffixes[2])
            return alpha / down.size(0), down

    @staticmethod
    def extract(
        base: Parameter(torch.Tensor, "delta"),
        rank: Parameter(int) = 8,
        use_approximate_basis: Parameter(bool) = True,
        approximate_basis_iters: Parameter(int) = 2,
        approximate_basis_seed: Parameter(int) = None,
    ) -> Return(torch.Tensor, "weight"):
        ...

    @classmethod
    def implement_extract_method(cls, lyco_config: LycorisModelConfig, base_config: ModelConfig):
        @merge_method(
            identifier=extraction_implementation_name(cls.__name__, lyco_config),
            implements=cls.extract_method,
            cache_factory=lambda: defaultdict(dict),
            globals=globals(),
            locals=locals(),
        )
        class BaseToLora(BaseToLycoris):
            @staticmethod
            def get_output_keys(base_key: str):
                keys = lyco_config.to_lycoris_keys(base_key, (cls,))
                if not keys:
                    return ()
                up, _, down, alpha = keys
                return up, down, alpha

            def __call__(
                self,
                base: Parameter(torch.Tensor, "delta", base_config),
                rank: Parameter(int, "param", base_config),
                use_approximate_basis: Parameter(bool, "param", base_config),
                approximate_basis_iters: Parameter(int, "param", base_config),
                approximate_basis_seed: Parameter(int, "param", base_config),
                **kwargs,
            ) -> Return(StateDict[torch.Tensor], "weight", lyco_config):
                (up_key, down_key, alpha_key) = kwargs["key_relation"].outputs

                original_shape = base.shape
                shape_vh = torch.Size((min(rank, original_shape[0]), *original_shape[1:]))
                shape_2d = torch.Size((original_shape[:1].numel(), original_shape[1:].numel()))

                if (cache := kwargs["cache"]) is not None:
                    cache = cache[kwargs["key"]]

                if (
                    cache is not None and
                    "s" in cache and cache["s"].numel() >= rank and
                    cache.get("iters", approximate_basis_iters) == approximate_basis_iters and
                    cache.get("seed", approximate_basis_seed) == approximate_basis_seed
                ):
                    u = cache["u"][..., :rank].to(base)
                    s = cache["s"][..., :rank].to(base)
                    vh = cache["vh"][..., :rank, :].to(base)
                else:
                    svd_driver = "gesvda" if base.is_cuda else None
                    if use_approximate_basis:
                        u, s, vh = svd_lowrank(base.reshape(shape_2d), rank=rank, iters=approximate_basis_iters, seed=approximate_basis_seed, driver=svd_driver)
                    else:
                        u, s, vh = torch.linalg.svd(base.reshape(shape_2d), full_matrices=False, driver=svd_driver)
                    if cache is not None:
                        cache["u"] = u.to(device="cpu", dtype=torch.float16)
                        cache["s"] = s.to(device="cpu", dtype=torch.float16)
                        cache["vh"] = vh.to(device="cpu", dtype=torch.float16)
                        if use_approximate_basis:
                            cache["iters"] = approximate_basis_iters
                            cache["seed"] = approximate_basis_seed
                        else:
                            cache.pop("iters", None)
                            cache.pop("seed", None)

                s = s[..., :rank].sqrt()
                u = u[..., :rank] * s.unsqueeze(-2)
                vh = s.unsqueeze(-1) * vh[..., :rank, :]

                return {
                    up_key: u,
                    down_key: vh.reshape(shape_vh),
                    alpha_key: torch.tensor(rank, device=base.device, dtype=base.dtype),
                }


class lokr(LycorisAlgorithm):
    suffixes = (
        "lokr_w1",
        "lokr_w1_a",
        "lokr_w1_b",
        "lokr_w2",
        "lokr_w2_a",
        "lokr_w2_b",
        "lokr_t2",
        "alpha",
    )
    supports_delta_conversion = True

    @classmethod
    def build_input_keys(cls, keys: KeysAccessor, lyco_config: LycorisModelConfig, base_key: str) -> Optional[ReqExpr]:
        lyco_keys = lyco_config.to_lycoris_keys(base_key, algos=(cls,))
        if not lyco_keys:
            return None

        w1, w1_a, w1_b, w2, w2_a, w2_b, t2, alpha = lyco_keys

        w1_inputs = keys[w1] @ cls.apply | keys[w1_a, w1_b] @ cls.apply_w1_factorized
        w2_inputs = keys[w2_a, w2_b, t2] @ cls.apply_w2_tucker | keys[w2] @ cls.apply | keys[w2_a, w2_b] @ cls.apply_w2_factorized
        alpha_input = keys[alpha] @ cls.compose_alpha | keys[()]

        return w1_inputs & w2_inputs & alpha_input

    class apply(ComposeObject):
        @classmethod
        def __call__(cls, state_dict: StateDictKeyHelper, target_shape: torch.Size) -> torch.Tensor:
            w1, lora_dim_1 = cls.get_w1(state_dict)
            w2, lora_dim_2 = cls.get_w2(state_dict)
            scale = cls.get_scale(state_dict, lora_dim_1 if lora_dim_1 is not None else lora_dim_2)

            while w1.dim() < w2.dim():
                w1 = w1.unsqueeze(-1)

            delta = torch.kron(w1, w2) * scale
            delta = delta.reshape(target_shape)

            base = state_dict.get_base_tensor()
            return base + delta

        @staticmethod
        def get_w1(sd: StateDictKeyHelper) -> Tuple[torch.Tensor, int | None]:
            w1 = sd.get_lyco_tensor(lokr.suffixes[0])
            return w1, None

        @staticmethod
        def get_w2(sd: StateDictKeyHelper) -> Tuple[torch.Tensor, int | None]:
            w2 = sd.get_lyco_tensor(lokr.suffixes[3])
            return w2, None

        @staticmethod
        def get_scale(sd: StateDictKeyHelper, lora_dim: int | None) -> torch.Tensor | float:
            return 1

    class apply_w1_factorized(apply):
        @staticmethod
        def get_w1(sd: StateDictKeyHelper) -> Tuple[torch.Tensor, int | None]:
            w1b = sd.get_lyco_tensor(lokr.suffixes[2])
            w1a = sd.get_lyco_tensor(lokr.suffixes[1])
            return w1a @ w1b, w1b.shape[0]

    class apply_w2_factorized(apply):
        @staticmethod
        def get_w2(sd: StateDictKeyHelper) -> Tuple[torch.Tensor, int | None]:
            w2b = sd.get_lyco_tensor(lokr.suffixes[5])
            w2a = sd.get_lyco_tensor(lokr.suffixes[4])
            w2_b_flat = w2b.flatten(1) if w2b.dim() > 1 else w2b
            return w2a @ w2_b_flat, w2b.shape[0]

    class apply_w2_tucker(apply):
        @staticmethod
        def get_w2(sd: StateDictKeyHelper) -> Tuple[torch.Tensor, int | None]:
            t2 = sd.get_lyco_tensor(lokr.suffixes[6])
            w2b = sd.get_lyco_tensor(lokr.suffixes[5])
            w2a = sd.get_lyco_tensor(lokr.suffixes[4])
            return rebuild_tucker(t2, w2a, w2b), w2b.shape[0]

    class compose_alpha(apply):
        @staticmethod
        def get_scale(sd: StateDictKeyHelper, lora_dim: int | None) -> torch.Tensor | float:
            alpha = sd.get_lyco_tensor(lokr.suffixes[7])
            if lora_dim is not None and alpha.isfinite():
                return alpha / lora_dim
            return 1

    @staticmethod
    def extract(
        base: Parameter(torch.Tensor, "delta"),
        dims: Parameter(torch.Tensor, "param") = ((1, 1), (1, 1)),
    ) -> Return(torch.Tensor, "weight"):
        ...

    @classmethod
    def implement_extract_method(cls, lyco_config: LycorisModelConfig, base_config: ModelConfig):
        @merge_method(
            identifier=extraction_implementation_name(cls.__name__, lyco_config),
            implements=cls.extract_method,
            cache_factory=lambda: defaultdict(dict),
            globals=globals(),
            locals=locals(),
        )
        class BaseToLokr(BaseToLycoris):
            @staticmethod
            def get_output_keys(base_key: str):
                keys = lyco_config.to_lycoris_keys(base_key, (cls,))
                if not keys:
                    return ()
                w1, _, _, w2, _, _, _, _ = keys
                return w1, w2

            def __call__(
                self,
                base: Parameter(torch.Tensor, "delta", base_config),
                dims: Parameter(torch.Tensor, "param", base_config),
                **kwargs,
            ) -> Return(StateDict[torch.Tensor], "weight", lyco_config):
                w1_key, w2_key = kwargs["key_relation"].outputs

                cache = kwargs.get("cache")
                if cache is not None:
                    cache = cache[kwargs["key"]]

                w1, w2 = extract_lokr(
                    base,
                    dims,
                    cache=cache,
                )

                return {
                    w1_key: w1,
                    w2_key: w2,
                }


class norm(LycorisAlgorithm):
    suffixes = (
        "w_norm",
        "b_norm",
    )
    supports_delta_conversion = True

    @staticmethod
    def targets_key(key: str, meta: KeyMetadata):
        bias = key.endswith("bias")
        shape_is_1d = meta.shape is not None and len(meta.shape) == 1
        return bias or shape_is_1d

    @classmethod
    def convert_key(cls, base_key: str, meta: KeyMetadata) -> Dict[str, KeyMetadata]:
        stem = cls.stem(base_key)
        stem_aliases = tuple(cls.stem(alias) for alias in meta.aliases)
        if base_key.endswith("bias"):
            suffix = cls.suffixes[1]
        else:
            suffix = cls.suffixes[0]

        lyco_key = f"{stem}.{suffix}"
        lyco_aliases = tuple(f"{stem_alias}.{suffix}" for stem_alias in stem_aliases)
        res = {
            lyco_key: dataclasses.replace(
                meta,
                shape=None,
                aliases=lyco_aliases,
                optional=True,
            )
        }
        return res

    @classmethod
    def build_input_keys(cls, keys: KeysAccessor, lyco_config: LycorisModelConfig, base_key: str) -> Optional[ReqExpr]:
        lyco_keys = lyco_config.to_lycoris_keys(base_key, algos=(cls,))
        if not lyco_keys:
            return None

        norm, = lyco_keys
        inputs = keys[norm]
        if base_key.endswith("bias"):
            inputs @= cls.apply_bias
        else:
            inputs @= cls.apply_weight

        return inputs

    class apply_weight(ComposeObject):
        @classmethod
        def __call__(cls, state_dict: StateDictKeyHelper, target_shape: torch.Size) -> torch.Tensor:
            return norm.apply(state_dict, target_shape, norm.suffixes[0])

    class apply_bias(ComposeObject):
        @classmethod
        def __call__(cls, state_dict: StateDictKeyHelper, target_shape: torch.Size) -> torch.Tensor:
            return norm.apply(state_dict, target_shape, norm.suffixes[1])

        @classmethod
        def get_suffix(cls) -> str:
            return norm.suffixes[1]

    @staticmethod
    def apply(state_dict: StateDictKeyHelper, target_shape: torch.Size, suffix: str):
        delta = state_dict.get_lyco_tensor(suffix)
        delta = delta.reshape(target_shape)
        base = state_dict.get_base_tensor()
        return base + delta

    @staticmethod
    def extract(
        diff: Parameter(torch.Tensor, "delta"),
    ) -> Return(torch.Tensor, "weight"):
        ...

    @classmethod
    def implement_extract_method(cls, lyco_config: LycorisModelConfig, base_config: ModelConfig):
        @merge_method(
            identifier=extraction_implementation_name(cls.__name__, lyco_config),
            implements=cls.extract_method,
            globals=globals(),
            locals=locals(),
        )
        class BaseToNorm(BaseToLycoris):
            @staticmethod
            def get_output_keys(base_key: str):
                lyco_keys = lyco_config.to_lycoris_keys(base_key, (cls,))
                if not lyco_keys:
                    return ()
                norm, = lyco_keys
                return norm,

            def __call__(
                self,
                diff: Parameter(torch.Tensor, "delta", base_config),
                **kwargs,
            ) -> Return(torch.Tensor, "weight", lyco_config):
                return diff


@dataclasses.dataclass(frozen=True, slots=True)
class BaseToLycoris(abc.ABC):
    @classmethod
    def map_keys(cls, b: KeyMapBuilder):
        for base_key in b.base.keys():
            if output_keys := cls.get_output_keys(base_key):
                b[output_keys] = b.keys[base_key]

    @staticmethod
    @abc.abstractmethod
    def get_output_keys(base_key: str) -> Iterable[str]:
        ...


class StateDictKeyHelper:
    def __init__(
        self,
        base_sd: Mapping[str, torch.Tensor | int],
        lyco_sd: Mapping[str, torch.Tensor],
        base_key: str,
        lyco_key_prefix: str,
    ):
        self.base_sd = base_sd
        self.lyco_sd = lyco_sd
        self.base_key = base_key
        self.lyco_key_prefix = lyco_key_prefix

    def get_lyco_tensor(self, lyco_suffix: str):
        return self.lyco_sd[f"{self.lyco_key_prefix}.{lyco_suffix}"]

    def get_base_tensor(self) -> torch.Tensor | int:
        return self.base_sd[self.base_key]


def rebuild_tucker(t, wa, wb):
    rebuild2 = torch.einsum("i j ..., i p, j r -> p r ...", t, wa, wb)
    return rebuild2


_register_all_lycoris_configs()
