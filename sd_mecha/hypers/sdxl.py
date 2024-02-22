from typing import Dict, Optional


CLIP_L14_HYEPRS = {
    f"l14_txt_{k}": v
    for k, v in ({
        f"block_{k}": f"conditioner.embedders.0.transformer.text_model.{v}"
        for k, v in ({
            f"in{i:02}": f"encoder.layers.{i}"
            for i in range(12)
        } | {
            "embed": "embeddings.",
            "final": "final_layer_norm.",
        }).items()
    } | {
        f"class_{k}": v
        for k, v in ({
            "pos_embed": ".embeddings.position_embedding.",
            "token_embed": ".embeddings.token_embedding.",
            "final_norm": ".final_layer_norm.",
            "layer_norm1": ".layer_norm1.",
            "layer_norm2": ".layer_norm2.",
            "mlp_fc1": ".mlp.fc1.",
            "mlp_fc2": ".mlp.fc2.",
            "q": ".self_attn.q_proj.",
            "k": ".self_attn.k_proj.",
            "v": ".self_attn.v_proj.",
            "out": ".self_attn.out_proj.",
        }).items()
    }).items()
}


CLIP_G14_HYPERS = {
    f"g14_txt_{k}": v
    for k, v in ({
        f"block_{k}": f"conditioner.embedders.1.model.{v}"
        for k, v in ({
            f"in{i:02}": f"transformer.resblocks.{i}"
            for i in range(32)
        } | {
            "ln_final": "ln_final.",
            "logit_scale": "logit_scale.",
            "pos_embed": "positional_embedding.",
            "token_embed": "token_embedding.",
            "text_proj": "text_projection.",
        }).items()
    } | {
        f"class_{k}": v
        for k, v in ({
            "ln_final": "1.model.ln_final.",
            "logit_scale": "1.model.logit_scale.",
            "positional_embedding": "1.model.positional_embedding.",
            "text_projection": "1.model.text_projection.",
            "token_embedding": "1.model.token_embedding.",
            # no "." for "attn_in_proj" is intentional, the parts are ".in_proj_bias" and ".in_proj_weight"
            "attn_in_proj": "attn.in_proj_",
            "attn_out_proj": "attn.out_proj.",
            "ln_1": "ln_1.",
            "ln_2": "ln_2.",
            "mlp_c_fc": "mlp.c_fc.",
            "mlp_c_proj": "mlp.c_proj.",
        }).items()
    }).items()
}


SDXL_HYPERS = {
    f"sdxl_unet_{k}": v
    for k, v in ({
        f"block_{k}": f"model.diffusion_model.{v}"
        for k, v in ({
            f"in{i:02}": f"input_blocks.{i}."
            for i in range(9)
        } | {
            "mid00": "middle_block.",
        } | {
            f"out{i:02}": f"output_blocks.{i}."
            for i in range(9)
        } | {
            "out": "out.",
            "time_embed": "time_embed.",
        }).items()
    } | {
        f"class_{k}": v
        for k, v in ({
            "in0": "model.diffusion_model.input_blocks.0.0.",
            "op": ".op.",
            "proj_in": ".proj_in.",
            "proj_out": ".proj_out.",
        } | {
            f"trans_attn{i}_{p}": f".attn{i}.to_{p}."
            for i in range(1, 3)
            for p in {"q", "k", "v", "out"}
        } | {
            f"trans_norm{i}": f".norm{i}."
            for i in range(1, 4)
        } | {
            "trans_ff_net0_proj": ".ff.net.0.proj.",
            "trans_ff_net2": ".ff.net.2.",
            "emb_layers1": ".emb_layers.1.",
            "in_layers0": ".in_layers.0.",
            "in_layers2": ".in_layers.2.",
            "out_layers0": ".out_layers.0.",
            "out_layers3": ".out_layers.3.",
            "norm": ".norm.",
            "skip_connection": ".skip_connection.",
            "conv": ".conv.",
            "out0": ".out.0.",
            "out2": ".out.2.",
            "time_embed0": ".time_embed.0",
            "time_embed2": ".time_embed.2",
        }).items()
    }).items()
} | CLIP_L14_HYEPRS | CLIP_G14_HYPERS
Hyper = float | Dict[str, float]


def sdxl_unet_blocks(
    in00: Optional[float] = None,
    in01: Optional[float] = None,
    in02: Optional[float] = None,
    in03: Optional[float] = None,
    in04: Optional[float] = None,
    in05: Optional[float] = None,
    in06: Optional[float] = None,
    in07: Optional[float] = None,
    in08: Optional[float] = None,
    mid00: Optional[float] = None,
    out00: Optional[float] = None,
    out01: Optional[float] = None,
    out02: Optional[float] = None,
    out03: Optional[float] = None,
    out04: Optional[float] = None,
    out05: Optional[float] = None,
    out06: Optional[float] = None,
    out07: Optional[float] = None,
    out08: Optional[float] = None,
    default: float = 0.0,
) -> dict:
    (
        out,
        time_embed,
    ) = (
        out08,
        calculate_time_embed_from_blocks(locals())
    )
    return {
        f"sdxl_unet_block_{k}": v if v is not None else default
        for k, v in locals().items() if v is not None
    }


def calculate_time_embed_from_blocks(blocks: dict) -> float:
    blocks_without_time = {"in00", "in03", "in06", "default"}
    return sum(
        v if v is not None else blocks["default"]
        for k, v in blocks.items()
        if k not in blocks_without_time
    ) / (len(blocks) - len(blocks_without_time))



def sdxl_unet_classes(
    default: float = 0.0, *,
    in0: Optional[float] = None,
    op: Optional[float] = None,
    proj_in: Optional[float] = None,
    proj_out: Optional[float] = None,
    trans_attn1_q: Optional[float] = None,
    trans_attn1_k: Optional[float] = None,
    trans_attn1_v: Optional[float] = None,
    trans_attn1_out: Optional[float] = None,
    trans_attn2_q: Optional[float] = None,
    trans_attn2_k: Optional[float] = None,
    trans_attn2_v: Optional[float] = None,
    trans_attn2_out: Optional[float] = None,
    trans_norm1: Optional[float] = None,
    trans_norm2: Optional[float] = None,
    trans_norm3: Optional[float] = None,
    trans_ff_net0_proj: Optional[float] = None,
    trans_ff_net2: Optional[float] = None,
    emb_layers1: Optional[float] = None,
    in_layers0: Optional[float] = None,
    in_layers2: Optional[float] = None,
    out_layers0: Optional[float] = None,
    out_layers3: Optional[float] = None,
    norm: Optional[float] = None,
    skip_connection: Optional[float] = None,
    conv: Optional[float] = None,
    out0: Optional[float] = None,
    out2: Optional[float] = None,
    time_embed0: Optional[float] = None,
    time_embed2: Optional[float] = None,
):
    return {
        f"sdxl_unet_class_{k}": v if v is not None else default
        for k, v in locals().items()
    }


def sdxl_txt_blocks(
    in00: Optional[float] = None,
    in01: Optional[float] = None,
    in02: Optional[float] = None,
    in03: Optional[float] = None,
    in04: Optional[float] = None,
    in05: Optional[float] = None,
    in06: Optional[float] = None,
    in07: Optional[float] = None,
    in08: Optional[float] = None,
    default: float = 0.0,
):
    embed = in00
    final = in08
    return {
        f"l14_txt_block_{k}": v if v is not None else default
        for k, v in locals().items()
    }


def sdxl_txt_classes(
    default: float = 0.0, *,
    pos_embed: Optional[float] = None,
    token_embed: Optional[float] = None,
    final_norm: Optional[float] = None,
    layer_norm1: Optional[float] = None,
    layer_norm2: Optional[float] = None,
    mlp_fc1: Optional[float] = None,
    mlp_fc2: Optional[float] = None,
    q: Optional[float] = None,
    k: Optional[float] = None,
    v: Optional[float] = None,
    out: Optional[float] = None,
):
    return {
        f"l14_txt_class_{k}": v if v is not None else default
        for k, v in locals().items()
    }


def sdxl_txt_g14_blocks(
    in00: Optional[float] = None,
    in01: Optional[float] = None,
    in02: Optional[float] = None,
    in03: Optional[float] = None,
    in04: Optional[float] = None,
    in05: Optional[float] = None,
    in06: Optional[float] = None,
    in07: Optional[float] = None,
    in08: Optional[float] = None,
    in09: Optional[float] = None,
    in10: Optional[float] = None,
    in11: Optional[float] = None,
    in12: Optional[float] = None,
    in13: Optional[float] = None,
    in14: Optional[float] = None,
    in15: Optional[float] = None,
    in16: Optional[float] = None,
    in17: Optional[float] = None,
    in18: Optional[float] = None,
    in19: Optional[float] = None,
    in20: Optional[float] = None,
    in21: Optional[float] = None,
    in22: Optional[float] = None,
    in23: Optional[float] = None,
    in24: Optional[float] = None,
    in25: Optional[float] = None,
    in26: Optional[float] = None,
    in27: Optional[float] = None,
    in28: Optional[float] = None,
    in29: Optional[float] = None,
    in30: Optional[float] = None,
    in31: Optional[float] = None,
    default: float = 0.0,
):
    pos_embed = in00
    text_proj = in00
    token_embed = in00
    ln_final = in08
    logit_scale = in08
    return {
        f"g14_txt_block_{k}": v if v is not None else default
        for k, v in locals().items()
    }


def sdxl_txt_g14_classes(
    default: Optional[float] = 0.0, *,
    pos_embed: Optional[float] = None,
    text_proj: Optional[float] = None,
    token_embed: Optional[float] = None,
    ln_final: Optional[float] = None,
    logit_scale: Optional[float] = None,
    attn_in_proj:  Optional[float] = None,
    attn_out_proj: Optional[float] = None,
    ln_1: Optional[float] = None,
    ln_2: Optional[float] = None,
    mlp_c_fc: Optional[float] = None,
    mlp_c_proj: Optional[float] = None,
):
    return {
        f"g14_txt_class_{k}": v if v is not None else default
        for k, v in locals().items()
    }
