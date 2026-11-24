from __future__ import annotations

from typing import Optional, Tuple

import torch
from lycoris.functional.locon import diff_weight

from sd_mecha.extensions.merge_methods import Parameter, Return, StateDict, merge_method
from sd_mecha.extensions.model_configs import ModelConfig
from sd_mecha.keys_map import ComposeObject, KeysAccessor, ReqExpr

from ..merge_methods._tensor_cache import create_tensor_cache, tensor_cache_for_key
from ..merge_methods.svd import _truncated_svd
from ._common import (
    BaseToLycoris, LycorisAlgorithm, LycorisModelConfig, StateDictKeyHelper,
    extraction_implementation_name, run_weight_function, tucker_fallback_reason,
)


class lora(LycorisAlgorithm):
    """Reconstruct a low-rank update from up and down factors and an optional Tucker core."""

    suffixes = (
        "lora_up.weight",
        "lora_mid.weight",
        "lora_down.weight",
        "alpha",
    )
    suffix_aliases = {
        "lora_up.weight": ("lora_B.weight", "lora_B.default.weight", "lora_B", "lora.up.weight"),
        "lora_down.weight": ("lora_A.weight", "lora_A.default.weight", "lora_A", "lora.down.weight"),
    }
    supports_delta_conversion = True

    @classmethod
    def build_input_keys(cls, keys: KeysAccessor, lyco_config: LycorisModelConfig, base_key: str) -> Optional[ReqExpr]:
        lyco_keys = lyco_config.to_lycoris_keys(base_key, algos=(cls,))
        if not lyco_keys:
            return None

        up, mid, down, alpha = lyco_keys
        return (
            keys[up, down] @ cls.apply &
            (keys[mid] @ cls.apply_mid | keys[()]) &
            (keys[alpha] @ cls.apply_alpha | keys[()])
        )

    class apply(ComposeObject):
        @classmethod
        def __call__(cls, state_dict: StateDictKeyHelper, target_shape: torch.Size) -> torch.Tensor:
            scale, down = cls.get_scale_down(state_dict)
            up = state_dict.get_lyco_tensor(lora.suffixes[0])
            up = up.reshape(up.shape[0], -1)
            mid = cls.get_mid(state_dict)
            if down.ndim == 1:
                down = down.unsqueeze(-1)
            fallback_reason = None
            if mid is not None:
                down = down.flatten(1)
                fallback_reason = tucker_fallback_reason(mid, up.shape[1], down.shape[0])
            delta = run_weight_function(diff_weight, down, up, mid, gamma=scale, fallback_reason=fallback_reason)
            return state_dict.apply_delta(delta, target_shape)

        @staticmethod
        def get_scale_down(sd: StateDictKeyHelper) -> Tuple[torch.Tensor | float, torch.Tensor]:
            down = sd.get_lyco_tensor(lora.suffixes[2])
            alpha = 1
            return alpha, down

        @staticmethod
        def get_mid(sd: StateDictKeyHelper) -> torch.Tensor | None:
            return None

    class apply_mid(apply):
        @staticmethod
        def get_mid(sd: StateDictKeyHelper) -> torch.Tensor:
            return sd.get_lyco_tensor(lora.suffixes[1])

    class apply_alpha(apply):
        @staticmethod
        def get_scale_down(sd: StateDictKeyHelper) -> Tuple[torch.Tensor | float, torch.Tensor]:
            alpha = sd.get_lyco_tensor(lora.suffixes[3])
            down = sd.get_lyco_tensor(lora.suffixes[2])
            return alpha / down.size(0), down

    @staticmethod
    def extract(
        base: Parameter(torch.Tensor, "delta"),
        rank: Parameter(int) = 8,
        use_approximate_basis: Parameter(bool) = True,
        approximate_basis_iters: Parameter(int) = 2,
        approximate_basis_seed: Parameter(int) = None,
    ) -> Return(torch.Tensor, "weight"):
        """Extract a LoRA adapter from a weight difference using a truncated SVD.

        Args:
            base: The weight difference to approximate.
            rank: The number of components to keep. Mecha caps it at the smaller matrix dimension.
            use_approximate_basis: Use a randomized SVD to estimate the components.
            approximate_basis_iters: The number of power iterations in the randomized SVD.
            approximate_basis_seed: The random seed for the basis estimate. None uses the current random state.

        Returns:
            A LyCORIS recipe that stores the up and down factors and their alpha.
        """
        ...

    @classmethod
    def implement_extract_method(cls, lyco_config: LycorisModelConfig, base_config: ModelConfig):
        @merge_method(
            identifier=extraction_implementation_name(cls.__name__, lyco_config),
            implements=cls.extract_method,
            cache_factory=create_tensor_cache,
            globals=globals(),
            locals=locals(),
        )
        class BaseToLora(BaseToLycoris):
            @staticmethod
            def get_output_keys(base_key: str):
                keys = lyco_config.to_lycoris_keys(base_key, (cls,))
                if not keys:
                    return ()
                up, _, down, alpha = keys
                return up, down, alpha

            def __call__(
                self,
                base: Parameter(torch.Tensor, "delta", base_config),
                rank: Parameter(int, "param", base_config),
                use_approximate_basis: Parameter(bool, "param", base_config),
                approximate_basis_iters: Parameter(int, "param", base_config),
                approximate_basis_seed: Parameter(int, "param", base_config),
                **kwargs,
            ) -> Return(StateDict[torch.Tensor], "weight", lyco_config):
                (up_key, down_key, alpha_key) = kwargs["key_relation"].outputs

                base_2d = base.flatten(1)
                rank = min(max(rank, 1), *base_2d.shape)

                cache, cache_device, cache_dtype = tensor_cache_for_key(kwargs.get("cache"), up_key)

                u, s, vh = _truncated_svd(
                    base_2d, rank, use_approximate_basis, approximate_basis_iters, approximate_basis_seed, cache,
                    cache_device=cache_device, cache_dtype=cache_dtype,
                )
                s = s.sqrt()
                up = (u * s.unsqueeze(-2)).reshape(base.shape[0], rank, *((1,) * (base.ndim - 2)))
                down = (s.unsqueeze(-1) * vh).reshape(rank, *base.shape[1:])

                return {
                    up_key: up,
                    down_key: down,
                    alpha_key: torch.tensor(rank, device=base.device, dtype=base.dtype),
                }
