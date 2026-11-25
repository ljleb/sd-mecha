import pytest
import torch

import sd_mecha
from sd_mecha.extensions import model_configs


def merge_with_block_weights(config, block_config, block_ids_by_key):
    block_ids = dict.fromkeys(block_ids_by_key.values())
    weights = {block_id: index / 32 for index, block_id in enumerate(block_ids, start=1)}
    a = sd_mecha.model({key: torch.zeros(2) for key in block_ids_by_key}, config)
    b = sd_mecha.model({key: torch.ones(2) for key in block_ids_by_key}, config)
    alpha = sd_mecha.convert(sd_mecha.literal(weights, block_config), a)
    alpha = sd_mecha.deserialize(sd_mecha.serialize(alpha))
    result = sd_mecha.merge(sd_mecha.weighted_sum(a, b, alpha), threads=0, output_dtype=torch.float32)
    for key, block_id in block_ids_by_key.items():
        torch.testing.assert_close(result[key], torch.full((2,), weights[block_id]))


def test_anima_block_weights():
    merge_with_block_weights("anima-diffusion_pipe", "anima-blocks", {
        "net.t_embedder.1.linear_1.weight": "DIFFUSER_IN",
        "net.blocks.0.self_attn.q_proj.weight": "DIFFUSER00",
        "net.blocks.0.mlp.layer1.weight": "DIFFUSER00",
        "net.blocks.27.self_attn.q_proj.weight": "DIFFUSER27",
        "net.final_layer.linear.weight": "DIFFUSER_OUT",
        "net.llm_adapter.embed.weight": "LLM_ADAPTER_IN",
        "net.llm_adapter.blocks.0.self_attn.q_proj.weight": "LLM_ADAPTER00",
        "net.llm_adapter.blocks.5.self_attn.q_proj.weight": "LLM_ADAPTER05",
        "net.llm_adapter.out_proj.weight": "LLM_ADAPTER_OUT",
        "text_encoders.qwen3_06b.transformer.model.embed_tokens.weight": "QWEN_TEXT_IN",
        "text_encoders.qwen3_06b.transformer.model.layers.0.self_attn.q_proj.weight": "QWEN_TEXT00",
        "text_encoders.qwen3_06b.transformer.model.layers.27.self_attn.q_proj.weight": "QWEN_TEXT27",
        "text_encoders.qwen3_06b.transformer.model.norm.weight": "QWEN_TEXT_OUT",
    })


@pytest.mark.parametrize("layout,projections", [("openmoss", ("q", "k", "v")), ("comfyui", ("qkv",))])
def test_moss_image_block_weights(layout, projections):
    config = model_configs.resolve(f"moss_image-{layout}")
    merge_with_block_weights(config, "moss_image-blocks", {
        **{f"layers.0.proj.{projection}.weight": "DIFFUSER00" for projection in projections},
        "layers.41.norm1.weight": "DIFFUSER41",
        "shared_block_modulation.weight": "DIFFUSER_IN",
        "final_linear_video.weight": "DIFFUSER_OUT",
        "video_head_modulation.weight": "DIFFUSER_OUT",
        "text_embedder.proj.weight": "TEXT_EMBEDDER_IN",
        "text_embedder.blocks.0.qkv.weight": "TEXT_EMBEDDER00",
        "text_embedder.blocks.1.qkv.weight": "TEXT_EMBEDDER01",
        "model.language_model.layers.0.input_layernorm.weight": "QWEN_TEXT00",
        "model.language_model.layers.31.input_layernorm.weight": "QWEN_TEXT31",
        "model.visual.patch_embed.proj.weight": "QWEN_VISION_IN",
        "model.visual.blocks.0.attn.qkv.weight": "QWEN_VISION00",
        "model.visual.blocks.26.attn.qkv.weight": "QWEN_VISION26",
        "model.visual.merger.linear_fc2.weight": "QWEN_VISION_OUT",
        "mtp.fc.weight": "QWEN_MTP",
        "mtp.layers.0.input_layernorm.weight": "QWEN_MTP",
        "mtp.norm.weight": "QWEN_MTP",
        **dict.fromkeys(config.components()["vae"].keys(), "VAE"),
    })
