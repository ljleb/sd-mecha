import enum
from typing import Dict, Tuple
import fuzzywuzzy.process


SD15_HYPER_PARAMETERS = {
    f"in{i:02}" for i in range(12)
} | {
    f"out{i:02}" for i in range(12)
} | {
    "mid00",
    "op",
    "proj_in",
    "proj_out",
} | {
    f"trans_attn{i}_{p}"
    for i in range(1, 3)
    for p in {"q", "k", "v", "out"}
} | {
    f"trans_norm{i}"
    for i in range(1, 4)
} | {
    "trans_ff_net0_proj",
    "trans_ff_net2",
    "emb_layers1",
    "in_layers0",
    "in_layers2",
    "out_layers0",
    "out_layers3",
    "norm",
    "skip_connection",
    "conv",
    "out0",
    "out2",
}
SingleComponentParameter = float | Dict[str, float]
HyperParameter = SingleComponentParameter | Tuple[SingleComponentParameter, SingleComponentParameter]


def validate_hyper_parameter(alpha: HyperParameter, nested: bool = False):
    if isinstance(alpha, dict):
        for key in alpha.keys():
            if key not in SD15_HYPER_PARAMETERS:
                suggestion = fuzzywuzzy.process.extractOne(key, SD15_HYPER_PARAMETERS)[0]
                raise ValueError(f"Unsupported dictionary key '{key}'. Did you mean '{suggestion}'?")
    elif isinstance(alpha, tuple):
        if nested:
            raise ValueError("Nested tuples are not supported for hyperparameters.")
        if len(alpha) != 2:
            raise ValueError("Tuples must contain exactly two elements: a text encoder hyperparameter and a UNet hyperparameter.")
        for v in alpha:
            validate_hyper_parameter(v, nested=True)
    elif not isinstance(alpha, float):
        raise TypeError("Hyperparameter must be a float, dictionary, or tuple.")


def generate_blocks_enum(name, count, is_unet=True):
    return enum.Enum(name, {
        f"IN{i:02}": enum.auto()
        for i in range(count)
    } | ({
        'MID00': enum.auto()
    } if is_unet else {}) | {
        f"OUT{i:02}": enum.auto()
        for i in range(count)
    } if is_unet else {})


SD15UnetBlockName = generate_blocks_enum('SD15UnetBlockName', 12)
SD15TextEncoderBlockName = generate_blocks_enum('SD15TextEncoderBlockName', 12, is_unet=False)

SDXLUnetBlockName = generate_blocks_enum('SDXLUnetBlockName', 9)


class SD15UnetClassName(enum.Enum):
    IN = enum.auto()
    OP = enum.auto()
    PROJ_IN = enum.auto()
    PROJ_OUT = enum.auto()
    TRANSFORMER_ATTN1_Q = enum.auto()
    TRANSFORMER_ATTN1_K = enum.auto()
    TRANSFORMER_ATTN1_V = enum.auto()
    TRANSFORMER_ATTN1_OUT = enum.auto()
    TRANSFORMER_ATTN2_Q = enum.auto()
    TRANSFORMER_ATTN2_K = enum.auto()
    TRANSFORMER_ATTN2_V = enum.auto()
    TRANSFORMER_ATTN2_OUT = enum.auto()
    TRANSFORMER_NORM1 = enum.auto()
    TRANSFORMER_NORM2 = enum.auto()
    TRANSFORMER_NORM3 = enum.auto()
    TRANSFORMER_FF_NET0_PROJ = enum.auto()
    TRANSFORMER_FF_NET2 = enum.auto()
    EMB_LAYERS1 = enum.auto()
    IN_LAYERS0 = enum.auto()
    IN_LAYERS2 = enum.auto()
    OUT_LAYERS0 = enum.auto()
    OUT_LAYERS3 = enum.auto()
    NORM = enum.auto()
    SKIP_CONNECTION = enum.auto()
    CONV = enum.auto()
    OUT0 = enum.auto()
    OUT2 = enum.auto()


class SD15TextEncoderClassName(enum.Enum):
    POSITION_EMBEDDING = enum.auto()
    TOKEN_EMBEDDING = enum.auto()
    FINAL_NORM = enum.auto()
    LAYER_NORM1 = enum.auto()
    LAYER_NORM2 = enum.auto()
    MLP_FC1 = enum.auto()
    MLP_FC2 = enum.auto()
    Q = enum.auto()
    K = enum.auto()
    V = enum.auto()
    OUT = enum.auto()


class SdComponent(enum.Enum):
    UNET = enum.auto()
    TEXT_ENCODER = enum.auto()
    ALL = enum.auto()


class SdVersion(enum.Enum):
    SDXL = enum.auto()
    SD15 = enum.auto()
    ANY = enum.auto()
