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
                if not lycoris_keys:
                    raise StateDictKeyError(key)

                lycoris_key = next(iter(lycoris_keys))
                return compose_lora_up_down(lora, lycoris_key.split(".")[0])


def compose_lora_up_down(state_dict: Mapping[str, torch.Tensor], key: str):
    # fetching these 3 keys in lexicographic order is very important
    # any other order would raise the number of cache misses in the input safetensors when streaming keys
    #  which in turn would slow down merging significantly
    alpha = state_dict[f"{key}.alpha"]
    down_weight = state_dict[f"{key}.lora_down.weight"]
    up_weight = state_dict[f"{key}.lora_up.weight"]
    dim = down_weight.size(0)

    if up_weight.numel() <= down_weight.numel():
        up_weight = up_weight * (alpha / dim)
    else:
        down_weight = down_weight * (alpha / dim)

    if len(down_weight.size()) == 2:  # linear
        res = up_weight @ down_weight
    elif down_weight.size()[2:4] == (1, 1):  # conv2d 1x1
        res = (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
    else:  # conv2d 3x3
        res = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
    return res


class LycorisModelConfig(LazyModelConfigBase):
    def __init__(self, base_config: ModelConfig, lycoris_identifier: str, prefix: str, algorithms: Iterable[str] | str):
        super().__init__()
        self.base_config = base_config
        self.lycoris_identifier = lycoris_identifier
        self.prefix = prefix
        self.algorithms = list(sorted(algorithms)) if not isinstance(algorithms, str) else [algorithms]

    @property
    def identifier(self) -> str:
        return f"{self.base_config.identifier}_{self.lycoris_identifier}_{'_'.join(self.algorithms)}"

    def create_config(self):
        if "lora" not in self.algorithms or len(self.algorithms) != 1:
            raise ValueError(f"unknown lycoris algorithms {self.algorithms}")

        identifier = f"{self.base_config.identifier}_{self.lycoris_identifier}_{'_'.join(self.algorithms)}"
        components = {
            k: _to_lycoris_keys(component.keys, self.algorithms, self.prefix)
            for k, component in self.base_config.components.items()
        }
        return ModelConfigImpl(identifier, components)

    def to_lycoris_keys(self, key: StateDictKey) -> Mapping[StateDictKey, KeyMetadata]:
        return _to_lycoris_keys({key: KeyMetadata(None, None)}, self.algorithms, self.prefix)


def _to_lycoris_keys(
    keys: Mapping[StateDictKey, KeyMetadata],
    algorithms: Iterable[str],
    prefix: str,
) -> Dict[StateDictKey, KeyMetadata]:
    lycoris_keys = {}

    for algorithm in algorithms:
        for key, meta in keys.items():
            if key.endswith("bias") or not getattr(meta.metadata.dtype, "is_floating_point", True):
                continue

            key = key.split('.')
            if key[-1] == "weight":
                key = key[:-1]
            key = "_".join(key)

            for suffix in lycoris_algorithms[algorithm]:
                lycoris_key = f"{prefix}_{key}.{suffix}"
                lycoris_keys[lycoris_key] = dataclasses.replace(meta, shape=[])

    return lycoris_keys


lycoris_algorithms = {
    "lora": ("lora_up.weight", "lora_down.weight", "alpha"),
}


_register_all_lycoris_configs()
