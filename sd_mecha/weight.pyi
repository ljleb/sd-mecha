from typing import overload, Optional, Dict


ModelParameter = float | Dict[str, float]


def get_weight(parameter: ModelParameter, key: str) -> float:
    pass


def validate_model_parameter(parameter: ModelParameter) -> ModelParameter:
    pass


@overload
def unet15_blocks(value: float):
    pass


@overload
def unet15_blocks(
    in00: float,
    in01: float,
    in02: float,
    in03: float,
    in04: float,
    in05: float,
    in06: float,
    in07: float,
    in08: float,
    in09: float,
    in10: float,
    in11: float,
    mid00: float,
    out00: float,
    out01: float,
    out02: float,
    out03: float,
    out04: float,
    out05: float,
    out06: float,
    out07: float,
    out08: float,
    out09: float,
    out10: float,
    out11: float,
) -> dict:
    pass


@overload
def unet15_blocks(
    *,
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
    out09: Optional[float] = None,
    out10: Optional[float] = None,
    out11: Optional[float] = None,
    default: float = 0.0,
) -> dict:
    pass


def unet15_classes(
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
):
    pass


def txt15_blocks(
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
    default: float = 0.0,
):
    pass


def txt15_classes(
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
    pass
