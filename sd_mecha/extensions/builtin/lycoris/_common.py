from __future__ import annotations

import abc
import dataclasses
import warnings
from typing import Callable, ClassVar, Dict, Iterable, Mapping, Optional, Tuple

import torch
from lycoris.functional.general import add_scaled

from sd_mecha.extensions.merge_methods import MergeMethod, merge_method
from sd_mecha.extensions.model_configs import KeyMetadata, LazyModelConfigBase, ModelComponent, ModelConfig, ModelConfigImpl, StateDictKey
from sd_mecha.keys_map import KeyMapBuilder, KeysAccessor, ReqExpr
from sd_mecha.typing_ import ClassObject


class LycorisModelConfig(LazyModelConfigBase):
    """Build a LyCORIS adapter config from the keys of its base model."""

    def __init__(self, base_config: ModelConfig, algorithms: Iterable[type[LycorisAlgorithm]]):
        super().__init__()
        self.base_config = base_config
        self.algorithms = tuple(algorithms)

    @property
    def identifier(self) -> str:
        return f"{self.base_config.identifier}_lycoris_lora"

    def create_config(self):
        identifier = self.identifier
        components = {
            k: ModelComponent(_to_lycoris_keys(component.keys(), self.algorithms))
            for k, component in self.base_config.components().items()
        }
        return ModelConfigImpl(identifier, components)

    def to_lycoris_keys(self, key: StateDictKey, algos: Iterable[type[LycorisAlgorithm]] = None) -> Dict[StateDictKey, KeyMetadata]:
        """Return the canonical adapter keys and input aliases for one base key."""
        return _to_lycoris_keys(
            {key: self.base_config.keys()[key]}, algos if algos is not None else self.algorithms,
        )


def _to_lycoris_keys(
    base_keys: Mapping[StateDictKey, KeyMetadata],
    algorithms: Iterable[type[LycorisAlgorithm]],
) -> Dict[StateDictKey, KeyMetadata]:
    lyco_keys = {}
    prefixes = ("lycoris", "lora")

    for base_key, meta in base_keys.items():
        if meta.generated:
            continue
        for algo in algorithms:
            if not algo.targets_key(base_key, meta) or not getattr(meta.dtype, "is_floating_point", True):
                continue

            algo_keys = algo.convert_key(base_key, meta)
            for key, metadata in algo_keys.items():
                canonical = f"lycoris_{key}"
                stem, _, suffix = key.partition(".")
                alternatives = algo.suffix_aliases.get(suffix, ())
                aliases = list(metadata.aliases)
                aliases.extend(f"{p}_{stem}.{spelling}" for p in prefixes for spelling in (suffix, *alternatives))
                for base_spelling in (base_key, *meta.aliases):
                    module = base_spelling.removesuffix(".weight").removesuffix(".bias")
                    flat = algo.stem(base_spelling)
                    for spelling in (suffix, *alternatives):
                        aliases.extend((f"{flat}.{spelling}", f"{module}.{spelling}"))
                        aliases.extend(f"{p}_{flat}.{spelling}" for p in prefixes)

                lyco_keys[canonical] = dataclasses.replace(
                    metadata, aliases=tuple(dict.fromkeys(alias for alias in aliases if alias != canonical)),
                )

    return lyco_keys


_algos_by_name: Dict[str, type[LycorisAlgorithm]] = {}


@dataclasses.dataclass(frozen=True)
class LycorisAlgorithm(abc.ABC, metaclass=ClassObject):
    """Declare the keys and tensor operations for one adapter algorithm.

    Subclasses declare the suffixes and input aliases they read for each base key.
    `build_input_keys` picks the tensors and the function that reconstructs their update.
    Subclasses that implement extraction generate adapter tensors from a weight difference or a pair of weights.
    """

    suffixes: ClassVar[Tuple[str, ...]]
    suffix_aliases: ClassVar[Mapping[str, tuple[str, ...]]] = {}
    extract: ClassVar[Optional[Callable]] = None
    extract_method: ClassVar[Optional[MergeMethod]] = None
    supports_delta_conversion: ClassVar[bool] = False

    @classmethod
    def __init_subclass__(cls, **kwargs):
        if cls.extract is not None:
            cls.extract_method = merge_method(cls.extract, identifier=extraction_interface_name(cls.__name__), is_interface=True)
            _algos_by_name[cls.__name__] = cls

    @staticmethod
    def targets_key(key: str, meta: KeyMetadata):
        bias = key.endswith("bias")
        shape_at_least_2d = meta.shape is not None and len(meta.shape) >= 2 and meta.shape.numel() > 0
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
        """Describe the adapter tensors and reconstruction function for one output key."""
        ...

    @classmethod
    def implement_extract_method(cls, lyco_config: LycorisModelConfig, base_config: ModelConfig):
        raise NotImplementedError(f"{cls.__name__} does not implement extraction.")


def extraction_interface_name(algo_name: str):
    return f"extract_{algo_name}"


def extraction_implementation_name(algo_name: str, lyco_config: LycorisModelConfig):
    return f"{extraction_interface_name(algo_name)}_{lyco_config.identifier}"


@dataclasses.dataclass(frozen=True, slots=True)
class BaseToLycoris(abc.ABC):
    """Map a base-model key to the adapter tensors extracted for that key."""

    @classmethod
    def map_keys(cls, b: KeyMapBuilder):
        cls._map_first_input_keys(b, b.keys())

    @classmethod
    def _map_first_input_keys(cls, b: KeyMapBuilder, base_keys: Iterable[str]):
        for base_key in base_keys:
            if output_keys := cls.get_output_keys(base_key):
                b[output_keys] = b.keys[base_key]

    @staticmethod
    @abc.abstractmethod
    def get_output_keys(base_key: str) -> Iterable[str]:
        ...


def tucker_fallback_reason(core: torch.Tensor, up_rank: int, down_rank: int) -> str | None:
    """Describe a Tucker layout that needs the eager implementation."""
    if up_rank != down_rank or core.shape[:2] != (up_rank, down_rank):
        return "LyCORIS Tucker factors with unequal ranks"
    if up_rank > 64:
        return "LyCORIS Tucker factors with rank above 64"
    return None


def _weight_backend(*weights: torch.Tensor | None, fallback_reason: str | None = None) -> str | None:
    """Use eager PyTorch when a dtype or layout exceeds the fused kernel's limits."""
    fused_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if any(weight is not None and weight.dtype not in fused_dtypes for weight in weights):
        return "torch"
    if fallback_reason is not None:
        if any(weight is not None and weight.is_cuda for weight in weights):
            warnings.warn(f"Using eager PyTorch for {fallback_reason}.", stacklevel=2)
        return "torch"
    return None


_unavailable_acceleration: set[torch.device] = set()


def run_weight_function(function, *weights: torch.Tensor | None, fallback_reason: str | None = None, **kwargs):
    """Run a LyCORIS weight operation with eager PyTorch as a fallback."""
    device = next((weight.device for weight in weights if weight is not None and weight.is_cuda), None)
    if device in _unavailable_acceleration:
        return function(*weights, backend="torch", **kwargs)
    backend = _weight_backend(*weights, fallback_reason=fallback_reason)
    try:
        return function(*weights, backend=backend, **kwargs)
    except (FileNotFoundError, ImportError, RuntimeError) as error:
        missing_dependency = isinstance(error, (FileNotFoundError, ImportError)) or str(error).startswith((
            "Failed to find C compiler.", "Failed to find C++ compiler.",
        ))
        if device is None or backend == "torch" or not missing_dependency:
            raise
        _unavailable_acceleration.add(device)
        warnings.warn(
            f"LyCORIS could not start its accelerated backend on {device}. "
            f"Using eager PyTorch on this device for this process. {error}",
            stacklevel=2,
        )
        return function(*weights, backend="torch", **kwargs)


class StateDictKeyHelper:
    """Read the adapter tensors planned for one base key and apply their delta."""

    def __init__(
        self,
        base_sd: Mapping[str, torch.Tensor] | None,
        lyco_sd: Mapping[str, torch.Tensor],
        base_key: str,
    ):
        self.base_sd = base_sd
        self.base_key = base_key
        self.lyco_sd = lyco_sd
        self.lyco_key_prefix = next(iter(lyco_sd)).partition(".")[0]

    def get_lyco_tensor(self, lyco_suffix: str):
        return self.lyco_sd[f"{self.lyco_key_prefix}.{lyco_suffix}"]

    def apply_delta(self, delta: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        delta = delta.reshape(target_shape)
        if self.base_sd is None:
            return delta
        base = self.base_sd[self.base_key]
        if base.is_cuda:
            return run_weight_function(add_scaled, base, delta)
        return base + delta
