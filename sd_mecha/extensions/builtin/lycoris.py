import dataclasses
import torch
from typing import Iterable, Mapping, Dict
from sd_mecha.extensions import model_configs
from sd_mecha.extensions.merge_methods import merge_method, StateDict, Parameter, Return
from sd_mecha.extensions.model_configs import StateDictKey, ModelConfig, ModelConfigImpl, LazyModelConfigBase, KeyMetadata
from sd_mecha.streaming import StateDictKeyError


def _register_all_lycoris_configs():
    base_configs = model_configs.get_all_base()
    for base_config in base_configs:
        for lyco_config in (
            LycorisModelConfig(base_config, "lycoris", "lycoris", list(lycoris_algorithms)),
            LycorisModelConfig(base_config, "kohya", "lora", list(lycoris_algorithms)),
        ):
            model_configs.register_aux(lyco_config)
            lora_config_id = lyco_config.identifier
            base_config_id = lyco_config.base_config.identifier

            @merge_method(identifier=f"convert_'{lora_config_id}'_to_base", is_conversion=True)
            def diffusers_lora_to_base(
                lora: Parameter(StateDict[torch.Tensor], "weight", lora_config_id),
                **kwargs,
            ) -> Return(torch.Tensor, "delta", base_config_id):
                key = kwargs["key"]
                lycoris_keys = lyco_config.to_lycoris_keys(key)
                all_base_keys = lora.model_config.base_config.keys()
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

    def to_lycoris_keys(self, key: StateDictKey) -> Mapping[StateDictKey, KeyMetadata]:
        return _to_lycoris_keys({key: KeyMetadata(None, None)}, self.algorithms, self.prefix)


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
