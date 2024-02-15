import logging
import pathlib

import torch

from sd_mecha.sd_meh import utils as sd_meh_utils
from sd_mecha.sd_meh.streaming import OutSafetensorDict, InSafetensorDict
from typing import Optional, Dict


class MergeScheduler:
    def __init__(
        self, *,
        base_dir: Optional[pathlib.Path | str] = None,
        threads: int = 1,
        default_device: str = "cpu",
        default_dtype: Optional[torch.dtype] = torch.float16,
        default_work_device: Optional[str] = None,
        default_work_dtype: Optional[torch.dtype] = None,
        cache: Optional[dict] = None
    ):
        self.__base_dir = base_dir if base_dir is not None else base_dir
        if isinstance(self.__base_dir, str):
            self.__base_dir = pathlib.Path(self.__base_dir)
        self.__base_dir = self.__base_dir.absolute()

        self.__threads = threads
        self.__default_device = default_device
        self.__default_dtype = default_dtype
        self.__default_work_device = default_work_device if default_work_device is not None else default_device
        self.__default_work_dtype = default_work_dtype if default_work_dtype is not None else default_dtype
        self.__cache = cache

    def load_state_dict(self, state_dict: str | pathlib.Path | InSafetensorDict) -> InSafetensorDict:
        if isinstance(state_dict, InSafetensorDict):
            return state_dict
        if not isinstance(state_dict, pathlib.Path):
            state_dict = pathlib.Path(state_dict)
        if not state_dict.is_absolute():
            state_dict = self.__base_dir / state_dict
        if not state_dict.suffix:
            state_dict = state_dict.with_suffix(".safetensors")

        return InSafetensorDict(state_dict)

    def symbolic_merge(self, key, merge_method, inputs, alpha, beta, device, dtype, work_device, work_dtype):
        if self.__cache is not None and key not in self.__cache:
            self.__cache[key] = {}

        return merge_method(
            inputs,
            get_hyper_parameters(key, merge_method, alpha, beta),
            device if device is not None else self.__default_device,
            work_device if work_device is not None else self.__default_work_device,
            dtype if dtype is not None else self.__default_dtype,
            work_dtype if work_dtype is not None else self.__default_work_dtype,
            self.__cache[key] if self.__cache is not None else None,
        )

    def clip_weights(self, model, a, b):
        maximums = torch.maximum(a, b)
        minimums = torch.minimum(a, b)
        return torch.minimum(torch.maximum(model, minimums), maximums)

    def merge_and_save(
        self, recipe, *,
        output_path: Optional[pathlib.Path | str] = None,
    ):
        if not isinstance(output_path, pathlib.Path):
            output_path = pathlib.Path(output_path)
        if not output_path.is_absolute():
            output_path = self.__base_dir / output_path
        if not output_path.suffix:
            output_path = output_path.with_suffix(".safetensors")
        logging.info(f"Saving to {output_path}")

        arbitrary_input_model = recipe.get_arbitrary_input_model(self)
        output = OutSafetensorDict(output_path, arbitrary_input_model.header)
        for key in arbitrary_input_model.keys():
            if is_passthrough_key(key, arbitrary_input_model.header[key]["shape"]):
                output[key] = arbitrary_input_model[key]
            elif is_merge_key(key):
                output[key] = recipe.visit(key, self)
        output.finalize()


def is_passthrough_key(key: str, shape: list):
    is_ema = "ema." in key
    is_vae = key.startswith("first_stage_model.")
    return not is_ema and (is_vae or shape == [1000])


def is_merge_key(key: str):
    is_ema = "ema." in key
    is_unet = key.startswith("model.diffusion_model.")
    is_text_encoder = key.startswith("cond_stage_model.")
    return not is_ema and (is_unet or is_text_encoder)


def get_hyper_parameters(key: str, merge_method, alpha, beta) -> Dict[str, float]:
    return {
        "alpha": alpha,
        **({"beta": beta} if merge_method.requests_beta() else {}),
    }
