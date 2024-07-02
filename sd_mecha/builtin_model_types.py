import json
import pathlib
import torch
from sd_mecha.merge_space import MergeSpace
from sd_mecha.extensions.model_type import register_model_type
from typing import Mapping


@register_model_type(merge_space=MergeSpace.BASE, needs_header_conversion=False)
def base(state_dict: Mapping[str, torch.Tensor], key: str, **kwargs) -> torch.Tensor:
    return state_dict[key]


@register_model_type(merge_space=MergeSpace.DELTA, key_suffixes=[".lora_up.weight", ".lora_down.weight", ".alpha"], strict_suffixes=True)
def lora(state_dict: Mapping[str, torch.Tensor], key: str, **kwargs) -> torch.Tensor:
    if key.endswith(".bias"):
        raise KeyError(key)

    if key.startswith("cond_stage_model.transformer."):
        lora_key = "lora_te_" + "_".join(key.split(".")[2:-1])
    else:
        lora_key = SD1_MODEL_TO_LORA_KEYS[key]

    return compose_lora_up_down(state_dict, lora_key)


with open(pathlib.Path(__file__).parent / "lora" / "sd1_ldm_to_lora.json", 'r') as f:
    SD1_MODEL_TO_LORA_KEYS = json.load(f)


@register_model_type(merge_space=MergeSpace.DELTA, model_archs="sdxl", key_suffixes=[".lora_up.weight", ".lora_down.weight", ".alpha"], strict_suffixes=True)
def lora(state_dict: Mapping[str, torch.Tensor], key: str, **kwargs) -> torch.Tensor:
    if key.endswith((".bias", "_bias")):
        raise KeyError(key)

    if key.startswith("model.diffusion_model."):
        lora_key = "lora_unet_" + "_".join(key.split(".")[2:-1])
        return compose_lora_up_down(state_dict, lora_key)
    elif key.startswith("conditioner.embedders.0.transformer."):
        lora_key = "lora_te1_" + "_".join(key.split(".")[4:-1])
        return compose_lora_up_down(state_dict, lora_key)
    elif key.startswith("conditioner.embedders.1.model.transformer.resblocks."):
        # [6:] instead of [6:-1] because `key` can end with either ".weight" or "_weight"
        lora_key = "lora_te2_text_model_encoder_layers_" + "_".join(key.split(".")[6:])
        lora_key = lora_key.replace("_attn_", "_self_attn_")
        lora_key = lora_key.replace("_mlp_c_fc_", "_mlp_fc1_")
        lora_key = lora_key.replace("_mlp_c_proj_", "_mlp_fc2_")
        lora_key = lora_key.replace("_weight", "")

        if lora_key.endswith("_in_proj"):
            lora_key = lora_key.replace("_in_proj", "")
            lora_keys = [
                f"{lora_key}_{k}_proj"
                for k in ("q", "k", "v")
            ]
            return torch.vstack([
                compose_lora_up_down(state_dict, k)
                for k in lora_keys
            ])

        return compose_lora_up_down(state_dict, lora_key)
    else:
        raise KeyError(key)


def compose_lora_up_down(state_dict: Mapping[str, torch.Tensor], key: str):
    up_weight = state_dict[f"{key}.lora_up.weight"].to(torch.float64)
    down_weight = state_dict[f"{key}.lora_down.weight"].to(torch.float64)
    alpha = state_dict[f"{key}.alpha"].to(torch.float64)
    dim = down_weight.size()[0]

    if len(down_weight.size()) == 2:  # linear
        res = up_weight @ down_weight
    elif down_weight.size()[2:4] == (1, 1):  # conv2d 1x1
        res = (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
    else:  # conv2d 3x3
        res = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
    return res * (alpha / dim)
