import pathlib

import safetensors.torch
import torch

from sd_mecha.sd_meh import utils as sd_meh_utils, merge as sd_meh_merge
from typing import Optional, Tuple


class MergeScheduler:
    def __init__(
        self, *,
        base_dir: Optional[pathlib.Path | str] = None,
        precision: int = 16,
        prune: bool = False,
        threads: int = 1,
        device: str = "cpu",
        work_device: Optional[str] = None,
    ):
        self.__base_dir = base_dir if base_dir is not None else base_dir
        if isinstance(self.__base_dir, str):
            self.__base_dir = pathlib.Path(self.__base_dir)
        self.__base_dir = self.__base_dir.absolute()

        self.__precision = precision
        self.__prune = prune
        self.__threads = threads
        self.__default_device = device
        self.__default_work_device = work_device

    def symbolic_merge(self, merge_method, a, b, c, alpha, beta, rebasin_iters, device, work_device, prune, threads):
        models = models_dict(a, b, c)
        weights, bases = weights_and_bases(merge_method, alpha, beta)

        return sd_meh_merge.merge_models(
            models,
            weights,
            bases,
            merge_method,
            self.__precision,
            False,
            bool(rebasin_iters),
            rebasin_iters if rebasin_iters is not None else 0,
            device if device is not None else self.__default_device,
            work_device if work_device is not None else self.__default_work_device,
            prune if prune is not None else self.__prune,
            threads if threads is not None else self.__threads,
        )

    def clip_weights(self, model, a, b, device):
        models = models_dict(a, b, None)
        return sd_meh_merge.clip_weights(models, model)

    def load_state_dict(self, path, device):
        if not isinstance(path, (str, pathlib.Path)):
            return path
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        if not path.suffix:
            path = path.with_suffix(".safetensors")
        if not path.is_absolute():
            path = self.__base_dir / path
        return sd_meh_merge.load_sd_model(path, device if device is not None else self.__default_device)

    def merge_and_save(self, merge_tree, *, output_path: Optional[pathlib.Path | str] = None):
        merged = merge_tree.visit(self)

        if not isinstance(output_path, pathlib.Path):
            output_path = pathlib.Path(output_path)
        if not output_path.is_absolute():
            output_path = self.__base_dir / output_path
        if not output_path.suffix:
            output_path = output_path.with_suffix(".safetensors")

        if output_path.suffix == ".safetensors":
            safetensors.torch.save_file(
                merged.to_dict(),
                f"{output_path}",
                metadata={"format": "pt"},
            )
        else:
            torch.save(
                {"state_dict": merged},
                f"{output_path.with_suffix('.ckpt')}",
            )


def models_dict(a, b, c=None) -> dict:
    models = {
        "model_a": a,
        "model_b": b,
    }
    if c is not None:
        models["model_c"] = c
    return models


def weights_and_bases(merge_method, alpha, beta=None) -> Tuple[dict, dict]:
    return sd_meh_utils.weights_and_bases(
        merge_method,
        None,
        alpha,
        None,
        None,
        beta,
        None,
        None,
        None,
        None,
        None,
    )
