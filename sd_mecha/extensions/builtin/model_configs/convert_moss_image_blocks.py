import re
from typing import TypeVar

from sd_mecha.extensions.merge_methods import Parameter, Return, StateDict, merge_method

from .convert_moss_image import comfyui_config, openmoss_config


T = TypeVar("T")


def _qwen_block_id(key):
    if match := re.match(r"model\.language_model\.layers\.(\d+)\.", key):
        return f"QWEN_TEXT{int(match[1]):02d}"
    if match := re.match(r"model\.visual\.blocks\.(\d+)\.", key):
        return f"QWEN_VISION{int(match[1]):02d}"
    if key.startswith("model.visual.merger."):
        return "QWEN_VISION_OUT"
    if key.startswith("model.visual."):
        return "QWEN_VISION_IN"
    if key.startswith("mtp."):
        return "QWEN_MTP"
    if key.startswith("model.language_model.embed_tokens."):
        return "QWEN_TEXT_IN"
    return "QWEN_TEXT_OUT"


def _diffuser_block_id(key):
    if match := re.match(r"layers\.(\d+)\.", key):
        return f"DIFFUSER{int(match[1]):02d}"
    if match := re.match(r"text_embedder\.blocks\.(\d+)\.", key):
        return f"TEXT_EMBEDDER{int(match[1]):02d}"
    if key.startswith("text_embedder."):
        return "TEXT_EMBEDDER_IN"
    if key.startswith(("final_linear_video.", "final_norm_video.", "video_head_modulation.")):
        return "DIFFUSER_OUT"
    return "DIFFUSER_IN"


def _get_block_ids(config):
    components = config.components()
    return {
        **{key: _diffuser_block_id(key) for key in components["diffuser"].keys()},
        **{key: _qwen_block_id(key) for key in components["qwen3_5_9b"].keys()},
        **dict.fromkeys(components["vae"].keys(), "VAE"),
    }


_openmoss_block_ids = _get_block_ids(openmoss_config)
_comfyui_block_ids = _get_block_ids(comfyui_config)


@merge_method(is_conversion=True, reuse_outputs=False)
class convert_moss_image_blocks_to_openmoss:
    @staticmethod
    def map_keys(b):
        for key, block_id in _openmoss_block_ids.items():
            b[key] = b.keys[block_id]

    def __call__(
        self,
        blocks: Parameter(StateDict[T], model_config="moss_image-blocks"),
        **kwargs,
    ) -> Return(T, model_config=openmoss_config):
        (block_id,) = kwargs["key_relation"].inputs["blocks"]
        return blocks[block_id]


@merge_method(is_conversion=True, reuse_outputs=False)
class convert_moss_image_blocks_to_comfyui:
    @staticmethod
    def map_keys(b):
        for key, block_id in _comfyui_block_ids.items():
            b[key] = b.keys[block_id]

    def __call__(
        self,
        blocks: Parameter(StateDict[T], model_config="moss_image-blocks"),
        **kwargs,
    ) -> Return(T, model_config=comfyui_config):
        (block_id,) = kwargs["key_relation"].inputs["blocks"]
        return blocks[block_id]
