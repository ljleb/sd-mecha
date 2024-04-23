import json
import pathlib
import lycoris
import torch
from sd_mecha.merge_space import MergeSpace
from sd_mecha.extensions.model_type import register_model_type
from typing import Mapping, Dict


@register_model_type(merge_space=MergeSpace.BASE, needs_header_conversion=False)
def base(state_dict: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    return state_dict[key]


@register_model_type(merge_space=MergeSpace.DELTA)
def lora(state_dict: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    if key.endswith(".bias"):
        raise KeyError(key)

    if key.startswith("cond_stage_model.transformer."):
        lora_key = "lora_te_" + "_".join(key.split(".")[2:-1])
    else:
        lora_key = SD1_MODEL_TO_LORA_KEYS[key]

    return compose_lora_up_down(state_dict, lora_key)


with open(pathlib.Path(__file__).parent / "lora" / "sd1_ldm_to_lora.json", 'r') as f:
    SD1_MODEL_TO_LORA_KEYS = json.load(f)


@register_model_type(merge_space=MergeSpace.DELTA, model_archs="sdxl")
def lora(state_dict: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
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




def merge(tes, unet, lyco_state_dict, scale: float = 1.0, device="cpu"):
    UNET_TARGET_REPLACE_MODULE = [
        "Linear",
        "Conv2d",
        "LayerNorm",
        "GroupNorm",
        "GroupNorm32",
    ]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = [
        "Embedding",
        "Linear",
        "Conv2d",
        "LayerNorm",
        "GroupNorm",
        "GroupNorm32",
        "Embedding",
    ]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    merged = 0

    def merge_state_dict(
        prefix,
        root_module: torch.nn.Module,
        lyco_state_dict: Dict[str, torch.Tensor],
        target_replace_modules,
    ):
        nonlocal merged
        for child_name, child_module in lycoris.utils.tqdm(
            list(root_module.named_modules()), desc=f"Merging {prefix}"
        ):
            if child_module.__class__.__name__ in target_replace_modules:
                lora_name = prefix + "." + child_name
                lora_name = lora_name.replace(".", "_")

                result, result_b = lycoris.utils.rebuild_weight(
                    *lycoris.utils.get_module(lyco_state_dict, lora_name),
                    getattr(child_module, "weight"),
                    getattr(child_module, "bias", None),
                    scale,
                )
                if result is not None:
                    key_dict.pop(lora_name)
                    merged += 1
                    child_module.requires_grad_(False)
                    child_module.weight.copy_(result)
                if result_b is not None:
                    child_module.bias.copy_(result_b)

    key_dict = {}
    for k, v in lycoris.utils.tqdm(list(lyco_state_dict.items()), desc="Converting Dtype and Device"):
        module, weight_key = k.split(".", 1)
        convert_key = lycoris.utils.convert_diffusers_name_to_compvis(module)
        if convert_key != module and len(tes) > 1:
            # kohya's format for sdxl is as same as SGM, not diffusers
            del lyco_state_dict[k]
            key_dict[convert_key] = key_dict.get(convert_key, []) + [k]
            k = f"{convert_key}.{weight_key}"
        else:
            key_dict[module] = key_dict.get(module, []) + [k]
        if device == "cpu":
            lyco_state_dict[k] = v.float().cpu()
        else:
            lyco_state_dict[k] = v.to(
                device, dtype=tes[0].parameters().__next__().dtype
            )

    for idx, te in enumerate(tes):
        if len(tes) > 1:
            prefix = LORA_PREFIX_TEXT_ENCODER + str(idx + 1)
        else:
            prefix = LORA_PREFIX_TEXT_ENCODER
        merge_state_dict(
            prefix,
            te.to(device),
            lyco_state_dict,
            TEXT_ENCODER_TARGET_REPLACE_MODULE,
        )
    merge_state_dict(
        LORA_PREFIX_UNET,
        unet.to(device),
        lyco_state_dict,
        UNET_TARGET_REPLACE_MODULE,
    )
    print(f"Unused state dict key: {key_dict}")
    print(f"{merged} modules were merged")
