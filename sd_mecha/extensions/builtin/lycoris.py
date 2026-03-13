import abc
import dataclasses
import torch
from typing import ClassVar, Iterable, Mapping, Dict, Tuple
from sd_mecha.extensions import model_configs
from .merge_methods.kronecker import kron_dims_from_ratio
from sd_mecha.extensions.merge_methods import merge_method, StateDict, Parameter, Return
from sd_mecha.extensions.model_configs import (
    ModelComponent, StateDictKey, ModelConfig, ModelConfigImpl,
    LazyModelConfigBase, KeyMetadata,
)
from sd_mecha.streaming import StateDictKeyError
from sd_mecha.keys_map import KeyMapBuilder, ComposeObject


def _register_all_lycoris_configs():
    algorithms = LycorisAlgorithm.all()
    for lyco_id, lyco_prefix in (
        ("lycoris", "lycoris"),
        ("kohya", "lora"),
    ):
        algo_interfaces = {}
        for algo in algorithms:
            algo_interfaces[algo.name] = algo.define_extract_method(lyco_id)

        for base_config_id in (
            "sdxl-kohya",
            "sdxl-kohya_but_diffusers",
            "sd1-kohya",
        ):
            base_config = model_configs.resolve(base_config_id)
            lyco_config = LycorisModelConfig(
                base_config,
                lyco_id,
                lyco_prefix,
                algorithms,
            )
            model_configs.register_aux(lyco_config)
            define_conversions(lyco_config, algorithms)
            for algo in algorithms:
                algo.implement_extract_method(algo_interfaces[algo.name], lyco_config, base_config)


def define_conversions(lyco_config, algorithms: Iterable["LycorisAlgorithm"]):
    lyco_config_id = lyco_config.identifier
    base_config = lyco_config.base_config
    base_config_id = base_config.identifier

    @merge_method(identifier=f"convert_'{lyco_config_id}'_to_base", is_conversion=True)
    class LycorisToBase:
        @staticmethod
        def map_keys(b: KeyMapBuilder):
            for base_key in base_config.keys():
                inputs = None

                for algo in algorithms:
                    algo_inputs = algo.relation_inputs(b, lyco_config, base_key)
                    if algo_inputs is None:
                        continue

                    inputs |= algo_inputs

                if inputs is not None:
                    b[base_key] = inputs

        def __call__(
            self,
            lora: Parameter(StateDict[torch.Tensor], "weight", lyco_config_id),
            **kwargs,
        ) -> Return(torch.Tensor, "delta", base_config_id):
            (base_key,), input_keys = key_relation = kwargs["key_relation"]
            target_shape = base_config.keys()[base_key].shape
            compose_fn = key_relation.meta

            key_prefix = input_keys["lora"][0].split(".")[0]
            sd_helper = StateDictKeyHelper(lora, key_prefix)
            try:
                return compose_fn(sd_helper, target_shape)
            except StateDictKeyError:
                pass

            raise StateDictKeyError(base_key)


class StateDictKeyHelper:
    def __init__(self, state_dict: Mapping[str, torch.Tensor], key_prefix):
        self.state_dict = state_dict
        self.key_prefix = key_prefix

    def get_tensor(self, name, raise_on_missing=True):
        try:
            return self.state_dict[f"{self.key_prefix}.{name}"]
        except StateDictKeyError as e:
            if raise_on_missing:
                raise
            else:
                return None


def rebuild_tucker(t, wa, wb):
    rebuild2 = torch.einsum("i j ..., i p, j r -> p r ...", t, wa, wb)
    return rebuild2


class LycorisModelConfig(LazyModelConfigBase):
    def __init__(self, base_config: ModelConfig, lycoris_identifier: str, prefix: str, algorithms: Iterable["LycorisAlgorithm"]):
        super().__init__()
        self.base_config = base_config
        self.lycoris_identifier = lycoris_identifier
        self.prefix = prefix
        self.algorithms = algorithms
        self.lycoris_to_base_keys = {
            lycoris_key: key
            for key, meta in base_config.keys().items()
            for lycoris_key in _to_lycoris_keys({key: dataclasses.replace(meta, shape=None)}, algorithms, self.prefix)
        }

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

    def to_lycoris_keys(self, key: StateDictKey, algos: Iterable["LycorisAlgorithm"] = None) -> Mapping[StateDictKey, KeyMetadata]:
        return _to_lycoris_keys({key: self.base_config.keys()[key]}, algos if algos is not None else self.algorithms, self.prefix)


def _to_lycoris_keys(
    base_keys: Mapping[StateDictKey, KeyMetadata],
    algorithms: Iterable["LycorisAlgorithm"],
    prefix: str,
) -> Dict[StateDictKey, KeyMetadata]:
    lycoris_keys = {}

    for key, meta in base_keys.items():
        for algo in algorithms:
            if key.endswith("bias") or not getattr(meta.dtype, "is_floating_point", True) or (
                meta.shape is not None and len(meta.shape) < 2
            ):
                continue

            parts = key.split(".")
            if parts[-1] == "weight":
                parts = parts[:-1]
            stem = "_".join(parts)

            for suffix in algo.suffixes:
                lycoris_key = f"{prefix}_{stem}.{suffix}"
                lycoris_keys[lycoris_key] = dataclasses.replace(
                    meta,
                    shape=[] if suffix == "alpha" else None,
                    optional=True,
                )

    return lycoris_keys


@dataclasses.dataclass(frozen=True)
class LycorisAlgorithm(abc.ABC):
    name: ClassVar[str]
    suffixes: ClassVar[Tuple[str, ...]]

    _registry: ClassVar[list[type["LycorisAlgorithm"]]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is LycorisAlgorithm:
            return
        if not getattr(cls, "name", None):
            raise TypeError(f"{cls.__name__} must define classvar 'name'")
        if not getattr(cls, "suffixes", None):
            raise TypeError(f"{cls.__name__} must define classvar 'suffixes'")
        LycorisAlgorithm._registry.append(cls)

    @classmethod
    def all(cls) -> tuple["LycorisAlgorithm", ...]:
        return tuple(algo_cls() for algo_cls in cls._registry)

    @abc.abstractmethod
    def relation_inputs(self, b, lyco_config, base_key):
        ...

    @abc.abstractmethod
    def define_extract_method(self, lyco_id: str):
        ...

    @abc.abstractmethod
    def implement_extract_method(self, lyco_interface, lyco_config, base_config):
        ...

    def key_by_suffix(self, lyco_keys: Mapping[str, object]) -> Tuple[str, ...]:
        out = {}
        for full_key in lyco_keys:
            _, suffix = full_key.split(".", 1)
            out[suffix] = full_key
        return tuple(out.get(suffix) for suffix in self.suffixes)


class LoraAlgorithm(LycorisAlgorithm):
    name = "lora"
    suffixes = (
        "lora_up.weight",
        "lora_mid.weight",
        "lora_down.weight",
        "alpha",
    )

    def relation_inputs(self, b: KeyMapBuilder, lyco_config: LycorisModelConfig, base_key: str):
        lyco_keys = lyco_config.to_lycoris_keys(base_key, algos=(self,))
        if not lyco_keys:
            return None

        up, mid, down, alpha = lyco_keys
        if not (up and down):
            return None

        return (
            b.keys[(up, down)] @ self.compose &
            (b.keys[mid] @ self.compose_mid | b.keys[()] if mid else b.keys[()]) &
            (b.keys[alpha] @ self.compose_alpha | b.keys[()] if alpha else b.keys[()])
        )

    class compose(ComposeObject):
        @classmethod
        def __call__(cls, state_dict: StateDictKeyHelper, target_shape: torch.Size) -> torch.Tensor:
            scale, down_weight = cls.get_scale_down(state_dict)
            delta = cls.build(state_dict, down_weight)

            delta *= scale
            delta = delta.reshape(target_shape)
            return delta

        @staticmethod
        def get_scale_down(sd: StateDictKeyHelper) -> Tuple[torch.Tensor | float, torch.Tensor]:
            down = sd.get_tensor("lora_down.weight")
            alpha = 1
            return alpha, down

        @staticmethod
        def build(sd: StateDictKeyHelper, down: torch.Tensor) -> torch.Tensor:
            up = sd.get_tensor("lora_up.weight")
            delta = up.view(up.size(0), -1) @ down.view(down.size(0), -1)
            return delta

    class compose_mid(compose):
        @staticmethod
        def build(sd: StateDictKeyHelper, down: torch.Tensor) -> torch.Tensor:
            mid = sd.get_tensor("lora_mid.weight")
            up = sd.get_tensor("lora_up.weight")
            wa = up.view(up.size(0), -1).transpose(0, 1)
            wb = down.view(down.size(0), -1)
            delta = rebuild_tucker(mid, wa, wb)
            return delta

    class compose_alpha(compose):
        @staticmethod
        def get_scale_down(sd: StateDictKeyHelper) -> Tuple[torch.Tensor | float, torch.Tensor]:
            alpha = sd.get_tensor("alpha")
            down = sd.get_tensor("lora_down.weight")
            return alpha / down.size(0), down

    def define_extract_method(self, lyco_id: str):
        @merge_method(identifier=f"extract_{lyco_id}_lora", is_interface=True)
        def extract_lora(
            base: Parameter(StateDict[torch.Tensor], "delta"),
            rank: Parameter(StateDict[int]) = 8,
        ) -> Return(StateDict[torch.Tensor], "weight"):
            ...
        return extract_lora

    def implement_extract_method(self, lyco_interface, lyco_config, base_config):
        algo = self
        lyco_config_id = lyco_config.identifier
        base_config_id = base_config.identifier

        @merge_method(
            identifier=f"extract_{algo.name}_'{lyco_config_id}'",
            implements=lyco_interface,
        )
        class BaseToLora:
            @classmethod
            def map_keys(cls, b):
                for base_key in base_config.keys():
                    if output_keys := cls.get_output_keys(base_key):
                        b[output_keys] = b.keys[base_key]

            @staticmethod
            def get_output_keys(base_key: str):
                keys = lyco_config.to_lycoris_keys(base_key, (algo,))
                if not keys:
                    return ()
                up, _, down, alpha = keys
                return up, down, alpha

            def __call__(
                self,
                base: Parameter(StateDict[torch.Tensor], "delta", base_config_id),
                rank: Parameter(StateDict[int], model_config=base_config_id),
                **kwargs,
            ) -> Return(StateDict[torch.Tensor], "weight", lyco_config_id):
                (up_key, down_key, alpha_key), input_keys = kwargs["key_relation"]
                base_key = input_keys["base"][0]

                base_value = base[base_key]
                rank_value = rank[base_key]
                original_shape = base_value.shape
                shape_vh = torch.Size((min(rank_value, original_shape[0]), *original_shape[1:]))
                shape_2d = torch.Size((original_shape[0], original_shape[1:].numel()))

                svd_driver = "gesvd" if base_value.is_cuda else None
                u, s, vh = torch.linalg.svd(base_value.reshape(shape_2d), full_matrices=False, driver=svd_driver)
                s = s[..., :rank_value].sqrt()
                u = u[..., :rank_value] * s.unsqueeze(-2)
                vh = s.unsqueeze(-1) * vh[..., :rank_value, :]

                return {
                    up_key: u,
                    down_key: vh.reshape(shape_vh),
                    alpha_key: torch.tensor(rank_value, device=base_value.device, dtype=base_value.dtype),
                }


class LokrAlgorithm(LycorisAlgorithm):
    name = "lokr"
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

    def relation_inputs(self, b: KeyMapBuilder, lyco_config: LycorisModelConfig, base_key: str):
        lyco_keys = lyco_config.to_lycoris_keys(base_key, algos=(self,))
        if not lyco_keys:
            return None

        w1, w1_a, w1_b, w2, w2_a, w2_b, t2, alpha = lyco_keys

        w1_inputs = None
        if w1:
            w1_inputs = b.keys[w1] @ self.compose
        if w1_a and w1_b:
            alt = b.keys[(w1_a, w1_b)] @ self.compose_w1_factorized
            w1_inputs |= alt
        if w1_inputs is None:
            return None

        w2_inputs = None
        if w2:
            w2_inputs = b.keys[w2] @ self.compose
        if w2_a and w2_b:
            alt = b.keys[(w2_a, w2_b)] @ self.compose_w2_factorized
            w2_inputs |= alt
            if t2:
                w2_inputs = (b.keys[(w2_a, w2_b, t2)] @ self.compose_w2_tucker) | w2_inputs
        if w2_inputs is None:
            return None

        alpha_input = (b.keys[alpha] @ self.compose_alpha) | b.keys[()] if alpha else b.keys[()]

        return w1_inputs & w2_inputs & alpha_input

    class compose(ComposeObject):
        @classmethod
        def __call__(cls, state_dict: StateDictKeyHelper, target_shape: torch.Size) -> torch.Tensor:
            w1, lora_dim_1 = cls.get_w1(state_dict)
            w2, lora_dim_2 = cls.get_w2(state_dict)
            scale = cls.get_scale(state_dict, lora_dim_1 if lora_dim_1 is not None else lora_dim_2)

            while w1.dim() < w2.dim():
                w1 = w1.unsqueeze(-1)

            delta = torch.kron(w1, w2) * scale
            return delta.reshape(target_shape)

        @staticmethod
        def get_w1(sd: StateDictKeyHelper) -> Tuple[torch.Tensor, int | None]:
            w1 = sd.get_tensor("lokr_w1")
            return w1, None

        @staticmethod
        def get_w2(sd: StateDictKeyHelper) -> Tuple[torch.Tensor, int | None]:
            w2 = sd.get_tensor("lokr_w2")
            return w2, None

        @staticmethod
        def get_scale(sd: StateDictKeyHelper, lora_dim: int | None) -> torch.Tensor | float:
            return 1

    class compose_w1_factorized(compose):
        @staticmethod
        def get_w1(sd: StateDictKeyHelper) -> Tuple[torch.Tensor, int | None]:
            w1a = sd.get_tensor("lokr_w1_a")
            w1b = sd.get_tensor("lokr_w1_b")
            return w1a @ w1b, w1b.shape[0]

    class compose_w2_factorized(compose):
        @staticmethod
        def get_w2(sd: StateDictKeyHelper) -> Tuple[torch.Tensor, int | None]:
            w2a = sd.get_tensor("lokr_w2_a")
            w2b = sd.get_tensor("lokr_w2_b")
            w2_b_flat = w2b.flatten(1) if w2b.dim() > 1 else w2b
            return w2a @ w2_b_flat, w2b.shape[0]

    class compose_w2_tucker(compose):
        @staticmethod
        def get_w2(sd: StateDictKeyHelper) -> Tuple[torch.Tensor, int | None]:
            w2a = sd.get_tensor("lokr_w2_a")
            w2b = sd.get_tensor("lokr_w2_b")
            t2 = sd.get_tensor("lokr_t2")
            return rebuild_tucker(t2, w2a, w2b), w2b.shape[0]

    class compose_alpha(compose):
        @staticmethod
        def get_scale(sd: StateDictKeyHelper, lora_dim: int | None) -> torch.Tensor | float:
            alpha = sd.get_tensor("alpha")
            if lora_dim is not None and alpha.isfinite():
                return alpha / lora_dim
            return 1

    def define_extract_method(self, lyco_id: str):
        @merge_method(identifier=f"extract_{lyco_id}_lokr", is_interface=True)
        def extract_lokr(
            base: Parameter(StateDict[torch.Tensor], "delta"),
            kronecker_ratio: Parameter(StateDict[float]) = 0.5,
        ) -> Return(StateDict[torch.Tensor], "weight"):
            ...
        return extract_lokr

    def implement_extract_method(self, lyco_interface, lyco_config, base_config):
        algo_name = self.name
        lyco_config_id = lyco_config.identifier
        base_config_id = base_config.identifier

        @merge_method(
            identifier=f"extract_{algo_name}_'{lyco_config_id}'",
            implements=lyco_interface,
        )
        class BaseToLokr:
            @classmethod
            def map_keys(cls, b):
                for base_key in base_config.keys():
                    if output_keys := cls.get_output_keys(base_key):
                        b[output_keys] = b.keys[base_key]

            @staticmethod
            def get_output_keys(base_key: str):
                keys = lyco_config.to_lycoris_keys(base_key, ("lokr",))
                if not keys:
                    return ()
                w1, _, _, w2, _, _, _, _ = keys
                return w1, w2

            def __call__(
                self,
                base: Parameter(StateDict[torch.Tensor], "delta", base_config_id),
                kronecker_ratio: Parameter(StateDict[float], model_config=base_config_id),
                **kwargs,
            ) -> Return(StateDict[torch.Tensor], "weight", lyco_config_id):
                (w1_key, w2_key), input_keys = kwargs["key_relation"]
                base_key = input_keys["base"][0]

                base_value = base[base_key]
                kronecker_ratio_value = kronecker_ratio[base_key]
                shape_original = base_value.shape
                m1, m2, n1, n2 = kron_dims_from_ratio(shape_original, kronecker_ratio_value)
                shape_w1 = torch.Size((m1, n1))
                shape_w2 = torch.Size((m2, n2, *shape_original[2:]))
                p2 = shape_original[2:].numel()

                value_2d = base_value.reshape(m1, m2, n1, n2 * p2).permute(0, 2, 1, 3).reshape(
                    shape_w1.numel(), shape_w2.numel()
                )

                svd_driver = "gesvd" if base_value.is_cuda else None
                u, s, vh = torch.linalg.svd(value_2d, full_matrices=False, driver=svd_driver)
                s = s[..., 0].sqrt()
                u = u[..., 0] * s
                vh = s * vh[..., 0, :]

                return {
                    w1_key: u.reshape(shape_w1),
                    w2_key: vh.reshape(shape_w2),
                }


_register_all_lycoris_configs()
