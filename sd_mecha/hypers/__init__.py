from typing import Dict
import fuzzywuzzy.process
from .sd15 import SD15_HYPERS, sd15_txt_blocks, sd15_txt_classes, sd15_unet_blocks, sd15_unet_classes
from .sdxl import SDXL_HYPERS, sdxl_txt_blocks, sdxl_txt_classes, sdxl_txt_g14_classes, sdxl_txt_g14_blocks, sdxl_unet_blocks, sdxl_unet_classes
Hyper = float | Dict[str, float]


def get_hyper(hyper: Hyper, key: str) -> float:
    if isinstance(hyper, float) or hyper is None:
        return hyper
    elif isinstance(hyper, dict):
        all_hypers = SDXL_HYPERS if any(key.startswith(("g14_txt_", "sdxl_unet_")) for key in hyper.keys()) else SD15_HYPERS

        hypers = []
        default = 0.0
        for key_identifier, weight in hyper.items():
            partial_key = all_hypers[key_identifier]
            if partial_key[0] != "." and key.startswith(partial_key) or partial_key in key:
                hypers.append(weight)
            elif key_identifier.endswith("_default"):
                default = weight
        if hypers:
            return sum(hypers) / len(hypers)
        return default
    else:
        raise TypeError(f"Hyperparameter must be a float or a dictionary, not {type(hyper)}")


def validate_hyper(hyper: Hyper) -> Hyper:
    if isinstance(hyper, dict):
        all_hypers = SDXL_HYPERS if any(key.startswith(("g14_txt_", "sdxl_unet_")) for key in hyper.keys()) else SD15_HYPERS

        for key in hyper.keys():
            if key not in all_hypers and not key.endswith("_default"):
                suggestion = fuzzywuzzy.process.extractOne(key, all_hypers.keys())[0]
                raise ValueError(f"Unsupported dictionary key '{key}'. Nearest match is '{suggestion}'.")
    elif isinstance(hyper, float) or hyper is None:
        return hyper
    else:
        raise TypeError(f"Hyperparameter must be a float or a dictionary, not {type(hyper)}")
