import json
import pathlib
import torch
from sd_mecha.merge_space import MergeSpace
from sd_mecha.extensions.model_type import register_model_type
from typing import Mapping


@register_model_type(merge_space=MergeSpace.BASE)
def base(state_dict: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    return state_dict[key]


@register_model_type
def lora(state_dict: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    lora_key = SD1_MODEL_TO_LORA_KEYS[key]
    up_weight = state_dict[f"{lora_key}.lora_up.weight"].to(torch.float64)
    down_weight = state_dict[f"{lora_key}.lora_down.weight"].to(torch.float64)
    alpha = state_dict[f"{lora_key}.alpha"].to(torch.float64)
    dim = down_weight.size()[0]

    if len(down_weight.size()) == 2:  # linear
        res = up_weight @ down_weight
    elif down_weight.size()[2:4] == (1, 1):  # conv2d 1x1
        res = (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
    else:  # conv2d 3x3
        res = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
    return res * (alpha / dim)


with open(pathlib.Path(__file__).parent / "lora" / "sd1_ldm_to_lora.json", 'r') as f:
    SD1_MODEL_TO_LORA_KEYS = json.load(f)
