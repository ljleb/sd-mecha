import re
from typing import TypeVar

from sd_mecha.extensions import model_configs
from sd_mecha.extensions.merge_methods import Parameter, Return, StateDict, merge_method


T = TypeVar("T")
anima_config = model_configs.resolve("anima-diffusion_pipe")


def _block_id(key, component):
    if component == "qwen3_06b":
        if match := re.match(r"text_encoders\.qwen3_06b\.transformer\.model\.layers\.(\d+)\.", key):
            return f"QWEN_TEXT{int(match[1]):02d}"
        if key.endswith(".embed_tokens.weight"):
            return "QWEN_TEXT_IN"
        return "QWEN_TEXT_OUT"
    if match := re.match(r"net\.blocks\.(\d+)\.", key):
        return f"DIFFUSER{int(match[1]):02d}"
    if match := re.match(r"net\.llm_adapter\.blocks\.(\d+)\.", key):
        return f"LLM_ADAPTER{int(match[1]):02d}"
    if key.startswith(("net.llm_adapter.embed.", "net.llm_adapter.in_proj.")):
        return "LLM_ADAPTER_IN"
    if key.startswith("net.llm_adapter."):
        return "LLM_ADAPTER_OUT"
    if key.startswith("net.final_layer."):
        return "DIFFUSER_OUT"
    return "DIFFUSER_IN"


_block_ids = {
    key: _block_id(key, component)
    for component, tensors in anima_config.components().items()
    for key in tensors.keys()
}


@merge_method(is_conversion=True, reuse_outputs=False)
class convert_anima_blocks_to_diffusion_pipe:
    @staticmethod
    def map_keys(b):
        for key, block_id in _block_ids.items():
            b[key] = b.keys[block_id]

    def __call__(
        self,
        blocks: Parameter(StateDict[T], model_config="anima-blocks"),
        **kwargs,
    ) -> Return(T, model_config=anima_config):
        (block_id,) = kwargs["key_relation"].inputs["blocks"]
        return blocks[block_id]
