import dataclasses
import functools
from typing import Iterable, Mapping, Dict
from sd_mecha.extensions.model_config import ModelConfig, get_all, StateDictKey, register_model_config
from sd_mecha.streaming import TensorMetadata


# run once
@functools.cache
def _register_all_lycoris():
    base_configs = get_all()
    for base_config in base_configs:
        register_model_config(LazyLycorisModelConfig(base_config, "lycoris", "lycoris", list(lycoris_algorithms)))
        register_model_config(LazyLycorisModelConfig(base_config, "kohya", "lora", ["lora"]))


def to_lycoris_config(base_config: ModelConfig, lycoris_identifier: str, prefix: str, algorithms: Iterable[str]) -> ModelConfig:
    if isinstance(algorithms, str):
        algorithms = [algorithms]
    algorithms = list(sorted(algorithms))

    if "lora" not in algorithms or len(algorithms) != 1:
        raise ValueError(f"unknown lycoris algorithms {algorithms}")

    return ModelConfig(
        identifier=f"{base_config.identifier}_{lycoris_identifier}_{'_'.join(algorithms)}",
        merge_space="delta",
        orphan_keys_to_copy=_to_lycoris_keys(base_config.orphan_keys_to_copy, algorithms, prefix),
        components={
            component_id: dataclasses.replace(
                component,
                orphan_keys_to_copy=_to_lycoris_keys(component.orphan_keys_to_copy, algorithms, prefix),
                blocks={
                    block_id: dataclasses.replace(
                        block,
                        keys_to_merge=_to_lycoris_keys(block.keys_to_merge, algorithms, prefix),
                        keys_to_copy=_to_lycoris_keys(block.keys_to_copy, algorithms, prefix),
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
    prefix: str,
) -> Dict[StateDictKey, TensorMetadata]:
    lycoris_keys = {}

    for algorithm in algorithms:
        for key, meta in keys.items():
            if key.endswith("bias"):
                continue

            key = key.split('.')
            if key[-1] == "weight":
                key = key[:-1]
            key = "_".join(key)

            for suffix in lycoris_algorithms[algorithm]:
                lycoris_key = f"{prefix}_{key}.{suffix}"
                lycoris_keys[lycoris_key] = dataclasses.replace(meta, shape=None)

    return lycoris_keys


lycoris_algorithms = {
    "lora": ("lora_up.weight", "lora_down.weight", "alpha"),
}


class LazyLycorisModelConfig:
    def __init__(self, base_config: ModelConfig, lycoris_identifier: str, prefix: str, algorithms: Iterable[str]):
        self.base_config = base_config
        self.lycoris_identifier = lycoris_identifier
        self.prefix = prefix
        self.algorithms = algorithms
        self.underlying_config = None

    @property
    def identifier(self) -> str:
        return f"{self.base_config.identifier}_{self.prefix}_{'_'.join(self.algorithms)}"

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]

        self._ensure_config()
        return getattr(self.underlying_config, item)

    def _ensure_config(self):
        if self.underlying_config is not None:
            return

        self.underlying_config = to_lycoris_config(self.base_config, self.lycoris_identifier, self.prefix, self.algorithms)


_register_all_lycoris()
