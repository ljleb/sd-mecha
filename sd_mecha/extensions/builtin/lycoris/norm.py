from __future__ import annotations

import dataclasses
from typing import Dict, Optional

import torch

from sd_mecha.extensions.merge_methods import Parameter, Return, merge_method
from sd_mecha.extensions.model_configs import KeyMetadata, ModelConfig
from sd_mecha.keys_map import ComposeObject, KeysAccessor, ReqExpr

from ._common import LycorisAlgorithm, LycorisModelConfig, StateDictKeyHelper, BaseToLycoris, extraction_implementation_name


class norm(LycorisAlgorithm):
    """Store additive deltas for one-dimensional weights and biases."""

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

    @staticmethod
    def apply(state_dict: StateDictKeyHelper, target_shape: torch.Size, suffix: str):
        delta = state_dict.get_lyco_tensor(suffix)
        return state_dict.apply_delta(delta, target_shape)

    @staticmethod
    def extract(
        diff: Parameter(torch.Tensor, "delta"),
    ) -> Return(torch.Tensor, "weight"):
        """Store one-dimensional weight and bias differences as a LyCORIS Norm adapter.

        Args:
            diff: A recipe containing the weight and bias differences to extract.

        Returns:
            A LyCORIS recipe with the selected tensors stored as `w_norm` or `b_norm`.
        """
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
