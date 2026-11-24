from __future__ import annotations

import warnings
from typing import Optional

import torch
from lycoris.functional.general import rebuild_tucker
from lycoris.functional.loha import diff_weight

from sd_mecha.keys_map import ComposeObject, KeysAccessor, ReqExpr

from ._common import LycorisAlgorithm, LycorisModelConfig, StateDictKeyHelper, run_weight_function, tucker_fallback_reason


def _factor_shape(up, down, core):
    if core is None:
        return up.shape[0], down.shape[1]
    return up.shape[1], down.shape[1], *core.shape[2:]


def _rebuild_delta(w1a, w1b, w2a, w2b, t1, t2, scale):
    if _factor_shape(w1a, w1b, t1) != _factor_shape(w2a, w2b, t2):
        # LyCORIS reshapes to the first factor's size after multiplying the factors.
        if t1 is None:
            delta = (w1a @ w1b) * (w2a @ w2b)
        else:
            delta = rebuild_tucker(t1, w1a, w1b) * rebuild_tucker(t2, w2a, w2b)
        warnings.warn("Using PyTorch broadcasting for LoHa factors with different output shapes.", stacklevel=2)
        return delta * scale

    rank = w1b.shape[0]
    up_rank_axis = 1 if t1 is None else 0
    fallback_reason = None
    if any(size != rank for size in (w1a.shape[up_rank_axis], w2a.shape[up_rank_axis], w2b.shape[0])):
        fallback_reason = "LoHa factors with different ranks"
    elif t1 is not None:
        fallback_reason = tucker_fallback_reason(t1, rank, rank) or tucker_fallback_reason(t2, rank, rank)
    return run_weight_function(diff_weight, w1b, w1a, w2b, w2a, t1, t2, gamma=scale, fallback_reason=fallback_reason)


class loha(LycorisAlgorithm):
    """Reconstruct an update by multiplying two low-rank weights element by element."""

    suffixes = ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "hada_t1", "hada_t2", "alpha")
    supports_delta_conversion = True

    @classmethod
    def build_input_keys(cls, keys: KeysAccessor, lyco_config: LycorisModelConfig, base_key: str) -> Optional[ReqExpr]:
        lyco_keys = lyco_config.to_lycoris_keys(base_key, algos=(cls,))
        if not lyco_keys:
            return None

        w1a, w1b, w2a, w2b, t1, t2, alpha = lyco_keys
        factors = keys[w1a, w1b, w2a, w2b]
        tucker = (factors & keys[t1, t2]) @ cls.apply_tucker
        return (tucker | factors @ cls.apply) & (keys[alpha] @ cls.with_alpha | keys[()])

    class apply(ComposeObject):
        @classmethod
        def __call__(cls, state_dict: StateDictKeyHelper, target_shape: torch.Size) -> torch.Tensor:
            w1a, w1b, w2a, w2b = (state_dict.get_lyco_tensor(suffix).flatten(1) for suffix in loha.suffixes[:4])
            t1, t2 = cls.get_cores(state_dict)
            scale = cls.scale(state_dict, w1b.shape[0])
            delta = _rebuild_delta(w1a, w1b, w2a, w2b, t1, t2, scale)
            return state_dict.apply_delta(delta, target_shape)

        @staticmethod
        def get_cores(state_dict):
            return None, None

        @staticmethod
        def scale(state_dict, rank):
            return 1

    class apply_tucker(apply):
        @staticmethod
        def get_cores(state_dict):
            return state_dict.get_lyco_tensor("hada_t1"), state_dict.get_lyco_tensor("hada_t2")

    class with_alpha(apply):
        @staticmethod
        def scale(state_dict, rank):
            alpha = state_dict.get_lyco_tensor("alpha")
            return alpha / rank if alpha.isfinite() and alpha != 0 else 1
