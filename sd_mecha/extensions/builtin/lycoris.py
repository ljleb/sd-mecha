import abc
import dataclasses
import inspect
import torch
from typing import Iterable, Mapping, Dict
from sd_mecha.extensions import model_configs
from .merge_methods.kronecker import kron_dims_from_ratio
from sd_mecha.extensions.merge_methods import merge_method, StateDict, Parameter, Return
from sd_mecha.extensions.model_configs import StateDictKey, ModelConfig, ModelConfigImpl, LazyModelConfigBase, KeyMetadata
from sd_mecha.streaming import StateDictKeyError


def _register_all_lycoris_configs():
    for lyco_id, lyco_prefix in (
        ("lycoris", "lycoris"),
        ("kohya", "lora"),
    ):
        @merge_method(identifier=f"extract_{lyco_id}_lora", is_interface=True)
        def extract_lora(
            base: Parameter(StateDict[torch.Tensor], "delta"),
            rank: Parameter(StateDict[int]) = 8,
        ) -> Return(StateDict[torch.Tensor], "weight"):
            ...

        @merge_method(identifier=f"extract_{lyco_id}_lokr", is_interface=True)
        def extract_lokr(
            base: Parameter(StateDict[torch.Tensor], "delta"),
            kronecker_ratio: Parameter(StateDict[float]) = 0.5,
        ) -> Return(StateDict[torch.Tensor], "weight"):
            ...

        lyco_interfaces = {
            "lora": extract_lora,
            "lokr": extract_lokr,
        }

        for base_config_id in (
            "sdxl-kohya",
            "sdxl-kohya_but_diffusers",
            "sd1-kohya",
        ):
            base_config = model_configs.resolve(base_config_id)
            lyco_config = LycorisModelConfig(base_config, lyco_id, lyco_prefix, list(lycoris_algorithms))
            model_configs.register_aux(lyco_config)
            define_conversions(lyco_config, lyco_interfaces)


def define_conversions(lyco_config, lyco_interfaces):
    lyco_config_id = lyco_config.identifier
    base_config = lyco_config.base_config
    base_config_id = base_config.identifier

    @merge_method(identifier=f"convert_'{lyco_config_id}'_to_base", is_conversion=True)
    class LycorisToBase:
        @staticmethod
        def map_keys(b):
            for base_key in base_config.keys():
                input_keys = tuple(lyco_config.to_lycoris_keys(base_key))
                if not input_keys:
                    continue
                b[base_key] = b.keys[input_keys]

        def __call__(
            self,
            lora: Parameter(StateDict[torch.Tensor], "weight", lyco_config_id),
            **kwargs,
        ) -> Return(torch.Tensor, "delta", base_config_id):
            (base_key,), input_keys = kwargs["key_relation"]
            target_shape = base_config.keys()[base_key].shape

            key_prefix = input_keys["lora"][0].split(".")[0]
            sd_helper = StateDictKeyHelper(lora, key_prefix)
            for compose_fn in (compose_lora, compose_lokr):
                try:
                    return compose_fn(sd_helper, target_shape)
                except StateDictKeyError:
                    pass

            raise StateDictKeyError(base_key)

    class BaseToLycoris(abc.ABC):
        @classmethod
        def map_keys(cls, b):
            for base_key in base_config.keys():
                if output_keys := cls.get_output_keys(base_key):
                    b[output_keys] = b.keys[base_key]

        @staticmethod
        @abc.abstractmethod
        def get_output_keys(base_key: str):
            ...

    @merge_method(identifier=f"extract_lora_'{lyco_config_id}'", implements=lyco_interfaces["lora"])
    class BaseToLora(BaseToLycoris):
        @staticmethod
        def get_output_keys(base_key: str):
            keys = lyco_config.to_lycoris_keys(base_key, ("lora",))
            if keys:
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
            shape_vh = torch.Size((rank_value, *original_shape[1:]))
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

    @merge_method(identifier=f"extract_lokr_'{lyco_config_id}'", implements=lyco_interfaces["lokr"])
    class BaseToLokr(BaseToLycoris):
        @staticmethod
        def get_output_keys(base_key: str):
            keys = lyco_config.to_lycoris_keys(base_key, ("lokr",))
            if not keys:
                return keys
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

            value_2d = base_value.reshape(m1, m2, n1, n2*p2).permute(0, 2, 1, 3).reshape(shape_w1.numel(), shape_w2.numel())

            svd_driver = "gesvd" if base_value.is_cuda else None
            u, s, vh = torch.linalg.svd(value_2d, full_matrices=False, driver=svd_driver)
            s = s[..., 0].sqrt()
            u = u[..., 0] * s
            vh = s * vh[..., 0, :]

            return {
                w1_key: u.reshape(shape_w1),
                w2_key: vh.reshape(shape_w2),
            }


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


def compose_lora(state_dict: StateDictKeyHelper, target_shape: torch.Size):
    # fetching these 4 keys in lexicographic order is very important
    # any other order would raise the number of cache misses in the input safetensors when streaming keys
    #   which in turn would slow down merging significantly
    alpha = state_dict.get_tensor("alpha")
    down_weight = state_dict.get_tensor("lora_down.weight")
    mid_weight = state_dict.get_tensor("lora_mid.weight", raise_on_missing=False)
    up_weight = state_dict.get_tensor("lora_up.weight")
    dim = down_weight.size(0)

    if mid_weight is not None:
        wa = up_weight.view(up_weight.size(0), -1).transpose(0, 1)
        wb = down_weight.view(down_weight.size(0), -1)
        delta = rebuild_tucker(mid_weight, wa, wb)
    else:
        delta = up_weight.view(up_weight.size(0), -1) @ down_weight.view(down_weight.size(0), -1)

    delta = delta * (alpha / dim)
    delta = delta.reshape(target_shape)
    return delta


def compose_lokr(state_dict: StateDictKeyHelper, target_shape: torch.Size):
    lora_dim = None

    w1 = state_dict.get_tensor("lokr_w1", raise_on_missing=False)
    if w1 is None:
        w1a = state_dict.get_tensor("lokr_w1_a")
        w1b = state_dict.get_tensor("lokr_w1_b")
        w1 = w1a @ w1b
        lora_dim = w1b.shape[0]

    w2 = state_dict.get_tensor("lokr_w2", raise_on_missing=False)
    if w2 is None:
        w2a = state_dict.get_tensor("lokr_w2_a")
        w2b = state_dict.get_tensor("lokr_w2_b")
        t2 = state_dict.get_tensor("lokr_t2", raise_on_missing=False)

        if t2 is not None:
            w2 = rebuild_tucker(t2, w2a, w2b)
        else:
            # flatten for matmul if conv
            w2_b_flat = w2b.flatten(1) if w2b.dim() > 1 else w2b
            w2 = w2a @ w2_b_flat

        lora_dim = w2b.shape[0]

    while w1.dim() < w2.dim():
        w1 = w1.unsqueeze(-1)

    delta = torch.kron(w1, w2)

    alpha = state_dict.get_tensor("alpha", raise_on_missing=False)
    if alpha is not None and alpha.isfinite() and lora_dim is not None:
        delta *= alpha / lora_dim

    delta = delta.reshape(target_shape)
    return delta


def rebuild_tucker(t, wa, wb):
    rebuild2 = torch.einsum("i j ..., i p, j r -> p r ...", t, wa, wb)
    return rebuild2


class LycorisModelConfig(LazyModelConfigBase):
    def __init__(self, base_config: ModelConfig, lycoris_identifier: str, prefix: str, algorithms: Iterable[str]):
        super().__init__()
        self.base_config = base_config
        self.lycoris_identifier = lycoris_identifier
        self.prefix = prefix
        self.algorithms = list(sorted(algorithms))
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
            k: _to_lycoris_keys(component.keys(), self.algorithms, self.prefix)
            for k, component in self.base_config.components().items()
        }
        return ModelConfigImpl(identifier, components)

    def to_lycoris_keys(self, key: StateDictKey, algos: Iterable[str] = None) -> Mapping[StateDictKey, KeyMetadata]:
        return _to_lycoris_keys({key: dataclasses.replace(self.base_config.keys()[key], shape=None)}, algos if algos is not None else self.algorithms, self.prefix)


def _to_lycoris_keys(
    base_keys: Mapping[StateDictKey, KeyMetadata],
    algorithms: Iterable[str],
    prefix: str,
) -> Dict[StateDictKey, KeyMetadata]:
    lycoris_keys = {}

    for algorithm in algorithms:
        for key, meta in base_keys.items():
            if key.endswith("bias") or not getattr(meta.dtype, "is_floating_point", True) or meta.optional:
                continue

            key = key.split('.')
            if key[-1] == "weight":
                key = key[:-1]
            key = "_".join(key)

            for suffix in lycoris_algorithms[algorithm]:
                lycoris_key = f"{prefix}_{key}.{suffix}"
                lycoris_keys[lycoris_key] = dataclasses.replace(meta, shape=[], optional=True)

    return lycoris_keys


lycoris_algorithms = {
    "lora": ("lora_up.weight", "lora_mid.weight", "lora_down.weight", "alpha"),
    "lokr": ("lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2", "lokr_w2_a", "lokr_w2_b", "lokr_t2", "alpha"),
}


_register_all_lycoris_configs()
