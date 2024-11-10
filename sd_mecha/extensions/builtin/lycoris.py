import dataclasses
import inspect
from types import MethodType

import torch
from typing import Iterable, Mapping, Dict
from sd_mecha import extensions
from sd_mecha.extensions.merge_space import MergeSpace
from sd_mecha.extensions.merge_method import convert_to_recipe, config_converter, StateDict
from sd_mecha.extensions.model_config import StateDictKey, ModelConfig, ModelConfigImpl, LazyModelConfigBase
from sd_mecha.streaming import TensorMetadata, StateDictKeyError


def _register_all_lycoris_configs():
    base_configs = extensions.model_config.get_all_base()
    for base_config in base_configs:
        for lyco_config in (
            LycorisLazyModelConfig(base_config, "lycoris", "lycoris", list(lycoris_algorithms)),
            LycorisLazyModelConfig(base_config, "kohya", "lora", list(lycoris_algorithms)),
        ):
            extensions.model_config.register_aux(lyco_config)
            lora_config_id = lyco_config.identifier
            base_config_id = lyco_config.base_config.identifier

            @config_converter
            @convert_to_recipe(identifier=f"convert_'{lora_config_id.replace('-', '_')}'_to_base")
            def diffusers_lora_to_base(
                lora: StateDict | ModelConfig[lora_config_id] | MergeSpace["weight"],
                **kwargs,
            ) -> torch.Tensor | ModelConfig[base_config_id] | MergeSpace["delta"]:
                key = kwargs["key"]
                lycoris_keys = lyco_config.to_lycoris_keys(key)
                if not lycoris_keys:
                    raise StateDictKeyError(key)

                lycoris_key = next(iter(lycoris_keys))
                return compose_lora_up_down(lora, lycoris_key.split(".")[0])


def compose_lora_up_down(state_dict: Mapping[str, torch.Tensor], key: str):
    up_weight = state_dict[f"{key}.lora_up.weight"]
    down_weight = state_dict[f"{key}.lora_down.weight"]
    alpha = state_dict[f"{key}.alpha"]
    dim = down_weight.size()[0]

    if len(down_weight.size()) == 2:  # linear
        res = up_weight @ down_weight
    elif down_weight.size()[2:4] == (1, 1):  # conv2d 1x1
        res = (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
    else:  # conv2d 3x3
        res = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
    return res * (alpha / dim)


class LycorisLazyModelConfig(LazyModelConfigBase):
    def __init__(self, base_config: ModelConfig, lycoris_identifier: str, prefix: str, algorithms: Iterable[str]):
        super().__init__()
        self.base_config = base_config
        self.lycoris_identifier = lycoris_identifier
        self.prefix = prefix
        self.algorithms = algorithms
        self.multiple_text_encoders = None

    @property
    def identifier(self) -> str:
        return f"{self.base_config.identifier}_{self.lycoris_identifier}_{'_'.join(self.algorithms)}"

    def create_config(self):
        underlying_config = to_lycoris_config(self.base_config, self.lycoris_identifier, self.prefix, self.algorithms)
        self.multiple_text_encoders = any(key.startswith("text_encoder_2") for key in self.base_config.compute_keys())
        return underlying_config

    def to_lycoris_keys(self, key: StateDictKey) -> Mapping[StateDictKey, TensorMetadata]:
        return _to_lycoris_keys({key: TensorMetadata(None, None)}, self.algorithms, self.lycoris_identifier, self.prefix, self.multiple_text_encoders)


def to_lycoris_config(base_config: ModelConfig, lycoris_identifier: str, prefix: str, algorithms: Iterable[str]) -> ModelConfig:
    if isinstance(algorithms, str):
        algorithms = [algorithms]
    algorithms = list(sorted(algorithms))

    multiple_text_encoders = any(key.startswith("text_encoder_2") for key in base_config.compute_keys())

    if "lora" not in algorithms or len(algorithms) != 1:
        raise ValueError(f"unknown lycoris algorithms {algorithms}")

    return ModelConfigImpl(
        identifier=f"{base_config.identifier}_{lycoris_identifier}_{'_'.join(algorithms)}",
        orphan_keys_to_copy=_to_lycoris_keys(base_config.orphan_keys_to_copy, algorithms, lycoris_identifier, prefix, multiple_text_encoders),
        components={
            component_id: dataclasses.replace(
                component,
                orphan_keys_to_copy=_to_lycoris_keys(component.orphan_keys_to_copy, algorithms, lycoris_identifier, prefix, multiple_text_encoders),
                blocks={
                    block_id: dataclasses.replace(
                        block,
                        keys_to_merge=_to_lycoris_keys(block.keys_to_merge, algorithms, lycoris_identifier, prefix, multiple_text_encoders),
                        keys_to_copy=_to_lycoris_keys(block.keys_to_copy, algorithms, lycoris_identifier, prefix, multiple_text_encoders),
                    )
                    for block_id, block in component.blocks.items()
                },
            )
            for component_id, component in base_config.components.items()
        },
    )


def _to_lycoris_keys(
    keys: Mapping[StateDictKey, TensorMetadata],
    algorithms: Iterable[str],
    lycoris_identifier: str,
    prefix: str,
    multiple_text_encoders: bool,
) -> Dict[StateDictKey, TensorMetadata]:
    lycoris_keys = {}

    for algorithm in algorithms:
        for key, meta in keys.items():
            if key.endswith("bias"):
                continue

            key = key.split('.')
            if key[-1] == "weight":
                key = key[:-1]
            if lycoris_identifier == "kohya":
                if key[0] == "text_encoder":
                    key[0] = "te1" if multiple_text_encoders else "te"
                if key[0] == "text_encoder_2":
                    key[0] = "te2"
            key = "_".join(key)

            for suffix in lycoris_algorithms[algorithm]:
                lycoris_key = f"{prefix}_{key}.{suffix}"
                lycoris_keys[lycoris_key] = dataclasses.replace(meta, shape=None)

    return lycoris_keys


lycoris_algorithms = {
    "lora": ("lora_up.weight", "lora_down.weight", "alpha"),
}


_register_all_lycoris_configs()
