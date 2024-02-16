import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Dict, Optional, Tuple, Callable

import safetensors.torch
import torch
from tensordict import TensorDict
from sd_mecha.sd_meh.streaming import InSafetensorDict
from tqdm import tqdm

from sd_mecha.sd_meh.rebasin import (
    apply_permutation,
    sdunet_permutation_spec,
    step_weights_and_bases,
    update_model_a,
    weight_matching,
)
from sd_mecha.sd_meh.utils import (
    KEY_POSITION_IDS, MAX_TOKENS, NAI_KEYS, NUM_TOTAL_BLOCKS, NUM_INPUT_BLOCKS, NUM_MID_BLOCK,
)
logging.getLogger("sd_meh").addHandler(logging.NullHandler())


def fix_clip(model: TensorDict) -> TensorDict:
    if KEY_POSITION_IDS in model.keys():
        model[KEY_POSITION_IDS] = torch.tensor(
            [list(range(MAX_TOKENS))],
            dtype=torch.int64,
            device=model[KEY_POSITION_IDS].device,
        )

    return model


def fix_key(model: TensorDict, key: str) -> TensorDict:
    for nk in NAI_KEYS:
        if key.startswith(nk):
            model[key.replace(nk, NAI_KEYS[nk])] = model[key]
            del model[key]

    return model


# https://github.com/j4ded/sdweb-merge-block-weighted-gui/blob/master/scripts/mbw/merge_block_weighted.py#L115
def fix_model(model: TensorDict) -> TensorDict:
    for k in model.keys():
        model = fix_key(model, k)
    return fix_clip(model)


InMemoryDict = InSafetensorDict | TensorDict
InputDict = os.PathLike | str | InMemoryDict


def restore_sd_model(original_model: TensorDict, merged_model: TensorDict) -> TensorDict:
    for k in original_model:
        if k not in merged_model:
            merged_model[k] = original_model[k]
    return merged_model


def log_vram(txt=""):
    alloc = torch.cuda.memory_allocated(0)
    logging.debug(f"{txt} VRAM: {alloc*1e-9:5.3f}GB")


def merge_models(
    thetas: Dict[str, TensorDict],
    weights: dict,
    bases: dict,
    merge_method: Callable,
    dtype: torch.dtype,
    work_dtype: torch.dtype,
    weights_clip: bool,
    device: str,
    work_device: str,
    threads: int,
    cache: Optional[dict],
) -> TensorDict:
    logging.info(f"start merging with method {merge_method.__name__} on device {work_device} and dtype {work_dtype}")
    merged = simple_merge(
        thetas,
        weights,
        bases,
        merge_method,
        dtype=dtype,
        work_dtype=work_dtype,
        weights_clip=weights_clip,
        device=device,
        work_device=work_device,
        threads=threads,
        cache=cache,
    )

    return merged


def simple_merge(
    thetas: Dict[str, TensorDict],
    weights: dict,
    bases: dict,
    merge_method: Callable,
    dtype: torch.dtype,
    work_dtype: torch.dtype,
    weights_clip: bool,
    device: str,
    work_device: str,
    threads: int,
    cache: Optional[dict],
) -> TensorDict:
    futures = []
    with tqdm(thetas["a"].keys(), desc="stage 1") as progress:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            for key in thetas["a"].keys():
                future = executor.submit(
                    simple_merge_key,
                    progress,
                    key,
                    thetas,
                    weights,
                    bases,
                    merge_method,
                    dtype,
                    work_dtype,
                    weights_clip,
                    device,
                    work_device,
                    cache,
                )
                futures.append(future)

        for res in futures:
            res.result()

    log_vram("after stage 1")

    for key in tqdm(thetas["b"].keys(), desc="stage 2"):
        if KEY_POSITION_IDS in key:
            continue
        if "model" in key and key not in thetas["a"].keys():
            thetas["a"].update({key: thetas["b"][key].to(dtype)})

    log_vram("after stage 2")

    return fix_model(thetas["a"])


def simple_merge_key(progress, key, thetas, *args, **kwargs):
    with merge_key_context(key, thetas, *args, **kwargs) as result:
        if result is not None:
            thetas["a"].update({key: result.detach().clone()})

        progress.update()


def merge_key(
    key: str,
    thetas: Dict[str, TensorDict],
    weights: dict,
    bases: dict,
    merge_method: Callable,
    dtype: torch.dtype,
    work_dtype: torch.dtype,
    weights_clip: bool,
    device: str,
    work_device: str,
    cache: Optional[dict],
) -> Optional[Tuple[str, Dict]]:
    if KEY_POSITION_IDS in key:
        return

    for theta in thetas.values():
        if key not in theta.keys():
            return

    if "model" in key:
        current_bases = bases

        if "model.diffusion_model." in key:
            weight_index = -1

            re_inp = re.compile(r"\.input_blocks\.(\d+)\.")  # 12
            re_mid = re.compile(r"\.middle_block\.(\d+)\.")  # 1
            re_out = re.compile(r"\.output_blocks\.(\d+)\.")  # 12

            if "time_embed" in key:
                weight_index = 0  # before input blocks
            elif ".out." in key:
                weight_index = NUM_TOTAL_BLOCKS - 1  # after output blocks
            elif m := re_inp.search(key):
                weight_index = int(m.groups()[0])
            elif re_mid.search(key):
                weight_index = NUM_INPUT_BLOCKS
            elif m := re_out.search(key):
                weight_index = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + int(m.groups()[0])

            if weight_index >= NUM_TOTAL_BLOCKS:
                raise ValueError(f"illegal block index {key}")

            if weight_index >= 0:
                current_bases = {k: w[weight_index] for k, w in weights.items()}

        merged_key = merge_method(current_bases, thetas, key, device, work_device, dtype, work_dtype, cache)

        if weights_clip:
            merged_key = clip_weights_key(thetas, merged_key, key)

        return merged_key


def clip_weights(thetas, merged):
    for k in thetas["a"].keys():
        if k in thetas["b"].keys():
            merged.update({k: clip_weights_key(thetas, merged[k], k)})
    return merged


def clip_weights_key(thetas, merged_weights, key):
    t0 = thetas["a"][key]
    t1 = thetas["b"][key]
    maximums = torch.maximum(t0, t1)
    minimums = torch.minimum(t0, t1)
    return torch.minimum(torch.maximum(merged_weights, minimums), maximums)


@contextmanager
def merge_key_context(*args, **kwargs):
    result = merge_key(*args, **kwargs)
    try:
        yield result
    finally:
        if result is not None:
            del result


def save_model(model, output_file, file_format) -> None:
    logging.info(f"Saving {output_file}")
    if file_format == "safetensors":
        safetensors.torch.save_file(
            model if type(model) == dict else model.to_dict(),
            f"{output_file}.safetensors",
            metadata={"format": "pt"},
        )
    else:
        torch.save({"state_dict": model}, f"{output_file}.ckpt")
