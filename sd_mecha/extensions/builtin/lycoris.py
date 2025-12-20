import abc
import dataclasses
import inspect
import torch
from typing import Iterable, Mapping, Dict
from sd_mecha.extensions import model_configs
from sd_mecha.extensions.builtin.merge_methods.kronecker import kron_dims_from_ratio
from sd_mecha.extensions.merge_methods import merge_method, StateDict, Parameter, Return
from sd_mecha.extensions.model_configs import StateDictKey, ModelConfig, ModelConfigImpl, LazyModelConfigBase, KeyMetadata
from sd_mecha.streaming import StateDictKeyError


def _register_all_lycoris_configs():
    for base_config_id in (
        "sdxl-kohya",
        "sdxl-kohya_but_diffusers",
        "sd1-kohya",
    ):
        base_config = model_configs.resolve(base_config_id)
        for lyco_config in (
            LycorisModelConfig(base_config, "lycoris", "lycoris", list(lycoris_algorithms)),
            LycorisModelConfig(base_config, "kohya", "lora", list(lycoris_algorithms)),
        ):
            model_configs.register_aux(lyco_config)
            lyco_to_base, base_to_algos = define_conversions(lyco_config)


def define_conversions(lyco_config):
    lyco_config_id = lyco_config.identifier
    base_config = lyco_config.base_config
    base_config_id = base_config.identifier

    @merge_method(identifier=f"convert_'{lyco_config_id}'_to_base", is_conversion=True)
    class DiffusersLycoToBase:
        @staticmethod
        def input_keys_for_output(base_key: str, *_args, **_kwargs):
            return list(lyco_config.to_lycoris_keys(base_key))

        def __call__(
            self,
            lora: Parameter(StateDict[torch.Tensor], "weight", lyco_config_id),
            **kwargs,
        ) -> Return(torch.Tensor, "delta", base_config_id):
            key = kwargs["key"]
            lycoris_keys = lyco_config.to_lycoris_keys(key)
            all_base_keys = lyco_config.base_config.keys()
            if not lycoris_keys or key not in all_base_keys:
                raise StateDictKeyError(key)

            target_shape = all_base_keys[key].shape

            key_prefix = next(iter(lycoris_keys)).split(".")[0]
            sd_helper = StateDictKeyHelper(lora, key_prefix)
            for compose_fn in (compose_lora, compose_lokr):
                try:
                    return compose_fn(sd_helper, target_shape)
                except StateDictKeyError:
                    pass

            raise StateDictKeyError(key)

    class BaseToDiffusersLycoris(abc.ABC):
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            fn = getattr(cls, "__call__", None)
            if fn is None:
                return

            spec = inspect.getfullargspec(fn)
            pos = 1 if inspect.ismethod(fn) else 0
            assert spec.args[pos] == "base"

        @classmethod
        def output_groups(cls):
            return [
                cls.get_output_keys(key)
                for key in base_config.keys()
            ]

        @staticmethod
        @abc.abstractmethod
        def get_output_keys(base_key: str):
            ...

        @staticmethod
        def input_keys_for_output(lyco_key: str, arg_name: str, *_args, **_kwargs):
            if arg_name != "base":
                return ()

            base_key = lyco_config.lycoris_to_base_keys.get(lyco_key)
            return (base_key,) if base_key is not None else ()

        @classmethod
        def base_key_for_output(cls, lyco_key: str):
            base_keys = cls.input_keys_for_output(lyco_key, "base")
            if not base_keys:
                raise StateDictKeyError(lyco_key)
            return base_keys[0]

    @merge_method(identifier=f"extract_lora_'{lyco_config_id}'")
    class BaseToDiffusersLora(BaseToDiffusersLycoris):
        @staticmethod
        def get_output_keys(base_key: str):
            keys = lyco_config.to_lycoris_keys(base_key, ("lora",))
            if not keys:
                return keys
            up, _, down, alpha = keys
            return up, down, alpha

        def __call__(
            self,
            base: Parameter(StateDict[torch.Tensor], "delta", base_config_id),
            rank: Parameter(int) = 8,
            **kwargs,
        ) -> Return(StateDict[torch.Tensor], "weight", lyco_config_id):
            lora_key = kwargs["key"]
            base_key = self.base_key_for_output(lora_key)
            lora_keys = lyco_config.to_lycoris_keys(base_key, ("lora",))
            if lora_key not in lora_keys:
                raise StateDictKeyError(lora_key)

            up_key, down_key, alpha_key = lora_keys
            base_value = base[base_key]
            original_shape = base_value.shape
            shape_vh = torch.Size((rank, *original_shape[1:]))
            shape_2d = torch.Size((original_shape[0], original_shape[1:].numel()))

            svd_driver = "gesvd" if base_value.is_cuda else None
            u, s, vh = torch.linalg.svd(base[base_key].reshape(shape_2d), full_matrices=False, driver=svd_driver)
            s = s[..., :rank].sqrt()
            u = u[..., :rank] * s.unsqueeze(-2)
            vh = s.unsqueeze(-1) * vh[..., :rank, :]

            return {
                up_key: u,
                down_key: vh.reshape(shape_vh),
                alpha_key: torch.tensor(rank, device=base_value.device, dtype=base_value.dtype),
            }

    @merge_method(identifier=f"extract_lokr_'{lyco_config_id}'")
    class BaseToDiffusersLokr(BaseToDiffusersLycoris):
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
            kronecker_ratio: Parameter(float) = 0.5,
            **kwargs,
        ) -> Return(StateDict[torch.Tensor], "weight", lyco_config_id):
            lokr_key = kwargs["key"]
            base_key, = self.base_key_for_output(lokr_key)
            lokr_keys = lyco_config.to_lycoris_keys(base_key, ("lokr",))
            if lokr_key not in lokr_keys:
                raise StateDictKeyError(lokr_key)

            w1_key, w2_key = lokr_keys
            base_value = base[base_key]
            shape_original = base_value.shape
            m1, m2, n1, n2 = kron_dims_from_ratio(shape_original, kronecker_ratio)
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

    return DiffusersLycoToBase, {
        "lora": BaseToDiffusersLora,
        "lokr": BaseToDiffusersLokr,
    }


class StateDictKeyHelper:
    def __init__(self, state_dict: Mapping[str, torch.Tensor], key_prefix):
        self.state_dict = state_dict
        self.key_prefix = key_prefix

    def get_tensor(self, name, raise_on_missing=True):
        try:
            return self.state_dict[f"{self.key_prefix}.{name}"]
        except StateDictKeyError:
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
            for key in base_config.keys()
            for lycoris_keys in _to_lycoris_keys({key: KeyMetadata(None, None)}, algorithms, self.prefix)
            for lycoris_key in lycoris_keys
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
        return _to_lycoris_keys({key: KeyMetadata(None, None)}, algos if algos is not None else self.algorithms, self.prefix)


def _to_lycoris_keys(
    base_keys: Mapping[StateDictKey, KeyMetadata],
    algorithms: Iterable[str],
    prefix: str,
) -> Dict[StateDictKey, KeyMetadata]:
    lycoris_keys = {}

    for algorithm in algorithms:
        for key, meta in base_keys.items():
            if key.endswith("bias") or not getattr(meta.metadata().dtype, "is_floating_point", True):
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
