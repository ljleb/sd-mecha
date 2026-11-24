from __future__ import annotations

import dataclasses
from typing import Dict, Optional

import torch

from sd_mecha.extensions.model_configs import KeyMetadata
from sd_mecha.keys_map import ComposeObject, KeysAccessor, ReqExpr

from ._common import LycorisAlgorithm, LycorisModelConfig, StateDictKeyHelper


class full(LycorisAlgorithm):
    """Apply complete weight and bias differences stored as `diff` and `diff_b`."""

    suffixes = ("diff", "diff_b")
    supports_delta_conversion = True

    @staticmethod
    def targets_key(key: str, meta: KeyMetadata):
        return meta.shape is not None and meta.shape.numel() > 0

    @classmethod
    def convert_key(cls, base_key: str, meta: KeyMetadata) -> Dict[str, KeyMetadata]:
        suffix = cls.suffixes[base_key.endswith(".bias")]
        aliases = tuple(f"{cls.stem(alias)}.{suffix}" for alias in meta.aliases)
        return {f"{cls.stem(base_key)}.{suffix}": dataclasses.replace(meta, aliases=aliases, optional=True)}

    @classmethod
    def build_input_keys(cls, keys: KeysAccessor, lyco_config: LycorisModelConfig, base_key: str) -> Optional[ReqExpr]:
        lyco_keys = lyco_config.to_lycoris_keys(base_key, algos=(cls,))
        if lyco_keys:
            apply_delta = cls.apply_bias if base_key.endswith(".bias") else cls.apply_weight
            return keys[next(iter(lyco_keys))] @ apply_delta
        return None

    class apply_weight(ComposeObject):
        @classmethod
        def __call__(cls, state_dict: StateDictKeyHelper, target_shape: torch.Size) -> torch.Tensor:
            return state_dict.apply_delta(state_dict.get_lyco_tensor("diff"), target_shape)

    class apply_bias(ComposeObject):
        @classmethod
        def __call__(cls, state_dict: StateDictKeyHelper, target_shape: torch.Size) -> torch.Tensor:
            return state_dict.apply_delta(state_dict.get_lyco_tensor("diff_b"), target_shape)
