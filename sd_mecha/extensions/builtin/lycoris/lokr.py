from __future__ import annotations

from collections import defaultdict
from typing import Optional

import torch
from lycoris.functional.lokr import kron_weight

from sd_mecha.extensions.merge_methods import Parameter, Return, StateDict, merge_method
from sd_mecha.extensions.model_configs import ModelConfig
from sd_mecha.keys_map import ComposeObject, KeysAccessor, ReqExpr

from ..merge_methods.kronecker import extract_lokr
from ._common import (
    BaseToLycoris, LycorisAlgorithm, LycorisModelConfig, StateDictKeyHelper,
    extraction_implementation_name, run_weight_function,
)


class lokr(LycorisAlgorithm):
    """Reconstruct an update as the Kronecker product of two dense or factorized weights."""

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
    supports_delta_conversion = True

    @classmethod
    def build_input_keys(cls, keys: KeysAccessor, lyco_config: LycorisModelConfig, base_key: str) -> Optional[ReqExpr]:
        lyco_keys = lyco_config.to_lycoris_keys(base_key, algos=(cls,))
        if not lyco_keys:
            return None

        w1, w1_a, w1_b, w2, w2_a, w2_b, t2, alpha = lyco_keys

        w1_inputs = keys[w1] @ cls.apply | keys[w1_a, w1_b] @ cls.apply_w1_factorized
        w2_inputs = keys[w2_a, w2_b, t2] @ cls.apply_w2_tucker | keys[w2] @ cls.apply | keys[w2_a, w2_b] @ cls.apply_w2_factorized
        alpha_input = keys[alpha] @ cls.compose_alpha | keys[()]

        return w1_inputs & w2_inputs & alpha_input

    class apply(ComposeObject):
        @classmethod
        def __call__(cls, state_dict: StateDictKeyHelper, target_shape: torch.Size) -> torch.Tensor:
            w1, w1a, w1b = cls.get_w1(state_dict)
            w2, w2a, w2b, t2 = cls.get_w2(state_dict)
            factor = w1b if w1b is not None else w2b
            rank = factor.shape[0] if factor is not None else None
            scale = cls.get_scale(state_dict, rank)
            delta = run_weight_function(kron_weight, w1, w1a, w1b, w2, w2a, w2b, t2, scale=scale)
            return state_dict.apply_delta(delta, target_shape)

        @staticmethod
        def get_w1(sd: StateDictKeyHelper):
            w1 = sd.get_lyco_tensor(lokr.suffixes[0])
            return w1, None, None

        @staticmethod
        def get_w2(sd: StateDictKeyHelper):
            w2 = sd.get_lyco_tensor(lokr.suffixes[3])
            return w2, None, None, None

        @staticmethod
        def get_scale(sd: StateDictKeyHelper, lora_dim: int | None) -> torch.Tensor | float:
            return 1

    class apply_w1_factorized(apply):
        @staticmethod
        def get_w1(sd: StateDictKeyHelper):
            w1a = sd.get_lyco_tensor(lokr.suffixes[1])
            w1b = sd.get_lyco_tensor(lokr.suffixes[2])
            return None, w1a, w1b

    class apply_w2_factorized(apply):
        @staticmethod
        def get_w2(sd: StateDictKeyHelper):
            w2a = sd.get_lyco_tensor(lokr.suffixes[4])
            w2b = sd.get_lyco_tensor(lokr.suffixes[5]).contiguous()
            return None, w2a, w2b, None

    class apply_w2_tucker(apply):
        @staticmethod
        def get_w2(sd: StateDictKeyHelper):
            w2a = sd.get_lyco_tensor(lokr.suffixes[4])
            w2b = sd.get_lyco_tensor(lokr.suffixes[5])
            t2 = sd.get_lyco_tensor(lokr.suffixes[6])
            return None, w2a, w2b, t2

    class compose_alpha(apply):
        @staticmethod
        def get_scale(sd: StateDictKeyHelper, lora_dim: int | None) -> torch.Tensor | float:
            alpha = sd.get_lyco_tensor(lokr.suffixes[7])
            if lora_dim is not None and alpha.isfinite():
                return alpha / lora_dim
            return 1

    @staticmethod
    def extract(
        base: Parameter(torch.Tensor, "delta"),
        dims: Parameter(torch.Tensor, "param") = None,
    ) -> Return(torch.Tensor, "weight"):
        """Extract a LoKr adapter that approximates a weight difference with a Kronecker product.

        Args:
            base: The weight difference to approximate.
            dims: A 2 by 2 tensor of factor sizes. The rows split the output and input axes,
                respectively, between the two factors. Use `lokr_dims` or `lokr_dims_lycoris`
                to choose sizes. None favors similarly sized factors.

        Returns:
            A LyCORIS recipe that stores the two Kronecker factors and their alpha.
        """
        ...

    @classmethod
    def implement_extract_method(cls, lyco_config: LycorisModelConfig, base_config: ModelConfig):
        @merge_method(
            identifier=extraction_implementation_name(cls.__name__, lyco_config),
            implements=cls.extract_method,
            cache_factory=lambda: defaultdict(dict),
            globals=globals(),
            locals=locals(),
        )
        class BaseToLokr(BaseToLycoris):
            @staticmethod
            def get_output_keys(base_key: str):
                keys = lyco_config.to_lycoris_keys(base_key, (cls,))
                if not keys:
                    return ()
                w1, _, _, w2, _, _, _, alpha = keys
                return w1, w2, alpha

            def __call__(
                self,
                base: Parameter(torch.Tensor, "delta", base_config),
                dims: Parameter(torch.Tensor, "param", base_config),
                **kwargs,
            ) -> Return(StateDict[torch.Tensor], "weight", lyco_config):
                w1_key, w2_key, alpha_key = kwargs["key_relation"].outputs

                cache = kwargs.get("cache")
                if cache is not None:
                    cache = cache[w1_key]

                w1, w2 = extract_lokr(
                    base,
                    dims,
                    cache=cache,
                )

                return {
                    w1_key: w1,
                    w2_key: w2,
                    alpha_key: base.new_tensor(1.0),
                }
