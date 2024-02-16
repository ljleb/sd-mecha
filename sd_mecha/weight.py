from typing import Dict, Optional
import fuzzywuzzy.process


SD15_HYPER_PARAMETERS = {
    f"unet_{k}": v
    for k, v in ({
        f"block_{k}": f"model.diffusion_model.{v}"
        for k, v in ({
            f"in{i:02}": f"input_blocks.{i}."
            for i in range(12)
        } | {
            "mid00": "middle_block.",
        } | {
            f"out{i:02}": f"output_blocks.{i}."
            for i in range(12)
        } | {
            "out": "out.",
        }).items()
    } | {
        f"class_{k}": v
        for k, v in ({
            "in0": "model.diffusion_model.input_blocks.0.0.",
            "op": ".op.",
            "proj_in": ".proj_in.",
            "proj_out": ".proj_out.",
        } | {
            f"trans_attn{i}_{p}": f".transformer_blocks.0.attn{i}.to_{p}."
            for i in range(1, 3)
            for p in {"q", "k", "v", "out"}
        } | {
            f"trans_norm{i}": f".transformer_blocks.0.norm{i}."
            for i in range(1, 4)
        } | {
            "trans_ff_net0_proj": ".transformer_blocks.0.ff.net.0.proj.",
            "trans_ff_net2": ".transformer_blocks.0.ff.net.2.",
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
        }).items()
    }).items()
} | {
    f"txt_{k}": v
    for k, v in ({
        f"block_{k}": f"cond_stage_model.transformer.text_model.{v}"
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
ModelParameter = float | Dict[str, float]


def get_weight(parameter: ModelParameter, key: str) -> float:
    if isinstance(parameter, float):
        return parameter
    elif isinstance(parameter, dict):
        weights = []
        for key_identifier, weight in parameter.items():
            partial_key = SD15_HYPER_PARAMETERS[key_identifier]
            if partial_key[0] != "." and key.startswith(partial_key) or partial_key in key:
                weights.append(weight)
        if weights:
            return sum(weights) / len(weights)
    return 0.0


def validate_model_parameter(parameter: ModelParameter) -> ModelParameter:
    if isinstance(parameter, dict):
        for key in parameter.keys():
            if key not in SD15_HYPER_PARAMETERS:
                suggestion = fuzzywuzzy.process.extractOne(key, SD15_HYPER_PARAMETERS)[0]
                raise ValueError(f"Unsupported dictionary key '{key}'. Nearest match is '{suggestion}'.")
    elif not isinstance(parameter, float):
        raise TypeError("Hyperparameter must be a float or a dictionary.")
    return parameter


def unet15_blocks(*args, default: float = 0.0, **kwargs):
    key_identifiers = [
        k
        for k in SD15_HYPER_PARAMETERS.keys()
        if k.startswith("unet_block_")
    ]
    max_positional_args = len(key_identifiers) - 1
    res = {}
    if len(args) == 1:
        if kwargs:
            raise TypeError(f"{unet15_blocks.__name__}() takes 0 keyword arguments with 1 positional argument but {len(kwargs)} were given")
        res.update({
            k: args[0]
            for k in key_identifiers
        })
    elif args and len(args) != max_positional_args:
        raise TypeError(f"{unet15_blocks.__name__}() either takes 0, 1 or {max_positional_args} positional arguments but {len(args)} were given")
    else:
        res.update(dict(zip(key_identifiers, args + (default,) * (len(key_identifiers) - len(args)))))
        res.update(kwargs)
        res["unet_block_out"] = res["unet_block_out11"]
    return res


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
    return {
        f"unet_class_{k}": v if v is not None else default
        for k, v in locals().items()
        if k != "default"
    }


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
    embed = in00
    final = in11
    return {
        f"txt_block_{k}": v if v is not None else default
        for k, v in locals().items()
        if k != "default"
    }


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
    return {
        f"txt_class_{k}": v if v is not None else default
        for k, v in locals().items()
        if k != "default"
    }
