import logging
import pathlib
from concurrent.futures import ThreadPoolExecutor

import torch
from tqdm import tqdm

from sd_mecha.streaming import OutSafetensorDict, InSafetensorDict
from typing import Optional, Dict


class MergeScheduler:
    def __init__(
        self, *,
        base_dir: Optional[pathlib.Path | str] = None,
        threads: int = 1,
        default_device: str = "cpu",
        default_dtype: Optional[torch.dtype] = torch.float16,
        cache: Optional[dict] = None
    ):
        self.__base_dir = base_dir if base_dir is not None else base_dir
        if isinstance(self.__base_dir, str):
            self.__base_dir = pathlib.Path(self.__base_dir)
        self.__base_dir = self.__base_dir.absolute()

        self.__threads = threads
        self.__default_device = default_device
        self.__default_dtype = default_dtype
        self.__cache = cache

    def load_state_dict(self, state_dict: str | pathlib.Path | InSafetensorDict, device: Optional[str]) -> InSafetensorDict:
        if isinstance(state_dict, InSafetensorDict):
            return state_dict
        if not isinstance(state_dict, pathlib.Path):
            state_dict = pathlib.Path(state_dict)
        if not state_dict.is_absolute():
            state_dict = self.__base_dir / state_dict
        if not state_dict.suffix:
            state_dict = state_dict.with_suffix(".safetensors")

        return InSafetensorDict(state_dict, device if device is not None else self.__default_device)

    def symbolic_merge(self, key, merge_method, inputs, alpha, beta, device, dtype):
        if self.__cache is not None and key not in self.__cache:
            self.__cache[key] = {}

        return merge_method(
            inputs,
            get_hyper_parameters(key, merge_method, alpha, beta),
            device if device is not None else self.__default_device,
            dtype if dtype is not None else self.__default_dtype,
            self.__cache[key] if self.__cache is not None else None,
        )

    def merge_and_save(
        self, recipe, *,
        output_path: Optional[pathlib.Path | str] = None,
        threads: int = 1,
    ):
        if not isinstance(output_path, pathlib.Path):
            output_path = pathlib.Path(output_path)
        if not output_path.is_absolute():
            output_path = self.__base_dir / output_path
        if not output_path.suffix:
            output_path = output_path.with_suffix(".safetensors")
        logging.info(f"Saving to {output_path}")

        input_dicts = recipe.get_input_dicts(self)
        arbitrary_input_dict = input_dicts[0]

        output = OutSafetensorDict(output_path, arbitrary_input_dict.header)
        progress = tqdm(total=len(arbitrary_input_dict.keys()), desc="Merging recipe")

        def _merge_and_save(key: str):
            progress.set_postfix({"key": key, "shape": arbitrary_input_dict.header[key]["shape"]})
            output[key] = recipe.visit(key, self)
            progress.update()

        def _forward_and_save(key: str):
            progress.set_postfix({"key": key})
            output[key] = arbitrary_input_dict[key]
            progress.update()

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for key in arbitrary_input_dict.keys():
                if is_passthrough_key(key, arbitrary_input_dict.header[key]["shape"]):
                    futures.append(executor.submit(_forward_and_save, key))
                elif is_merge_key(key):
                    futures.append(executor.submit(_merge_and_save, key))
                else:
                    progress.update()

        for res in futures:
            res.result()

        output.finalize()


def is_passthrough_key(key: str, shape: list):
    is_vae = key.startswith("first_stage_model.")
    is_time_embed = key.startswith("model.diffusion_model.time_embed.")
    is_position_ids = key == "cond_stage_model.transformer.text_model.embeddings.position_ids"
    return is_vae or is_time_embed or is_position_ids or shape == [1000]


def is_merge_key(key: str):
    is_unet = key.startswith("model.diffusion_model.")
    is_text_encoder = key.startswith("cond_stage_model.")
    return is_unet or is_text_encoder


def get_hyper_parameters(key: str, merge_method, alpha, beta) -> Dict[str, float]:
    hyper_parameters = {}
    if merge_method.requests_alpha():
        hyper_parameters["alpha"] = alpha
    if merge_method.requests_beta():
        hyper_parameters["beta"] = beta
    return hyper_parameters
