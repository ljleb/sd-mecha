import argparse
import re
from dataclasses import dataclass

import sd_mecha
from sd_mecha.extensions.builtin import merge_methods as mm


@dataclass
class Paths:
	sdxl_base: str
	kohaku_base: str
	model_a: str
	model_b: str
	output: str


@dataclass
class Weights:
	"""Weights controlling bridge and final."""
	lambda_k: float = 0.35			# Strength of Kohaku -> SDXL bridge
	lambda_b: float = 0.80			# Strength of B delta along with Kohaku branch
	lambda_a: float = 0.55			# Contribution of Model A delta
	lambda_bk: float = 0.65			# Contribution of bridged model B delta


@dataclass
class NormCfg:
	"""Normalization and stabilization configuration."""
	# Per-tensor RMS matching
	target_ratio: float = 1.0
	clip_multiple: float = 2.5
	# OPTIONAL: SLERP stabilization
	slerp_t: float = 0.50


@dataclass
class Gates:
	"""Module-wise gates to dampen instability-prone tensors."""
	qk: float = 0.45				# Attention Q/K projections
	qkv_out: float = 0.60			# V and attention_out projections
	ffn_conv: float = 0.75			# FFN and conv-like modules
	time_embed_out: float = 0.35	# Time embedding and UNet output heads

def _patterns() -> dict[str, re.Pattern]:
    """Compile regex patterns used to gate module subsets.

    Adjust these patterns if your state dict keys differ.
    """
    return {
        "qk": re.compile(r"(attn.*to_q|attn.*to_k|to_q\.weight|to_k\.weight|to_q\.bias|to_k\.bias)"),
        "qkv_out": re.compile(r"(attn.*to_v|attn.*to_out|to_v\.weight|to_out\.weight|to_v\.bias|to_out\.bias)"),
        "ffn_conv": re.compile(r"(conv\.|down_blocks|up_blocks|mid_block|mlp|ff|proj|fc)"),
        "time_out": re.compile(r"(time_embed|out\.weight|out\.bias)"),
    }


def apply_gates(
    node,
    *,
    qk: float,
    qkv_out: float,
    ffn_conv: float,
    time_embed_out: float,
):
    """Apply multiplicative gates to subsets of tensors using regex masks.

    This function composes masked scaling nodes that multiply delta contributions
    for specific key patterns. It uses the custom merge method `scale_by_mask`
    added under sd_mecha.extensions.builtin.merge_methods.

    Args:
        node: The input delta node to be gated.
        qk: Multiplier for Q/K projections.
        qkv_out: Multiplier for V and attention out projections.
        ffn_conv: Multiplier for FFN/conv-like modules.
        time_embed_out: Multiplier for time embedding and UNet outputs.

    Returns:
        A new node with gates applied.
    """
    pats = _patterns()
    gated = node
    gated = mm.scale_by_mask(gated, pattern=pats["qk"].pattern, multiplier=qk)
    gated = mm.scale_by_mask(gated, pattern=pats["qkv_out"].pattern, multiplier=qkv_out)
    gated = mm.scale_by_mask(gated, pattern=pats["ffn_conv"].pattern, multiplier=ffn_conv)
    gated = mm.scale_by_mask(gated, pattern=pats["time_out"].pattern, multiplier=time_embed_out)
    return gated


def build_recipe(
    paths: Paths,
    weights: Weights = Weights(),
    norm: NormCfg = NormCfg(),
    gates: Gates = Gates(),
):
    """Construct the sd_mecha recipe graph implementing anchored tri-delta.

    Graph (conceptual):
        W0 = SDXL base
        Wk = Kohaku-XL base
        Wa = Model A
        Wb = Model B

        ΔA   = Wa - W0
        ΔK   = Wk - W0
        ΔB|K = Wb - Wk

        ΔBK  = λK * ΔK + λB * ΔB|K

        Normalize per tensor (RMS to W0 with clipping)
        Gate sensitive modules

        Δmix = αA * ΔA_norm + αBK * ΔBK_norm
        W*   = W0 + Δmix
        Optional SLERP toward W0

    Returns:
        An sd_mecha recipe node suitable for sd_mecha.merge.
    """
    # Load models
    W0 = sd_mecha.model(paths.sdxl_base)
    Wk = sd_mecha.model(paths.kohaku_base)
    Wa = sd_mecha.model(paths.model_a)
    Wb = sd_mecha.model(paths.model_b)

    # Deltas
    dA = mm.subtract(Wa, W0)       # Wa - W0
    dK = mm.subtract(Wk, W0)       # Wk - W0
    dB_K = mm.subtract(Wb, Wk)     # Wb - Wk

    # Bridged B path back to SDXL basis
    dK_sc = mm.scale(dK, factor=weights.lambda_k)
    dB_K_sc = mm.scale(dB_K, factor=weights.lambda_b)
    dBK = mm.add(dK_sc, dB_K_sc)

    # Per-tensor RMS normalization against base (W0 weights) with clipping
    dA_norm = mm.scale_to_match_rms(dA, W0, target_ratio=norm.target_ratio, clip_multiple=norm.clip_multiple)
    dBK_norm = mm.scale_to_match_rms(dBK, W0, target_ratio=norm.target_ratio, clip_multiple=norm.clip_multiple)

    # Module-wise gates to stabilize attention and outputs
    dA_gated = apply_gates(
        dA_norm,
        qk=gates.qk,
        qkv_out=gates.qkv_out,
        ffn_conv=gates.ffn_conv,
        time_embed_out=gates.time_embed_out,
    )
    dBK_gated = apply_gates(
        dBK_norm,
        qk=gates.qk,
        qkv_out=gates.qkv_out,
        ffn_conv=gates.ffn_conv,
        time_embed_out=gates.time_embed_out,
    )

    # Final mixing of A and bridged-BK contributions
    dA_final = mm.scale(dA_gated, factor=weights.alpha_a)
    dBK_final = mm.scale(dBK_gated, factor=weights.alpha_bk)
    d_mix = mm.add(dA_final, dBK_final)

    # Apply to base and optional SLERP stabilization
    W_star_raw = mm.add_difference(W0, d_mix, alpha=1.0)
    W_star = mm.slerp(W0, W_star_raw, alpha=norm.slerp_t)

    return W_star


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anchored Tri-Delta Merge for SDXL-based anime models using sd_mecha."
    )
    parser.add_argument("--sdxl_base", type=str, required=True, help="Path to SDXL base checkpoint (.safetensors)")
    parser.add_argument("--kohaku_base", type=str, required=True, help="Path to Kohaku-XL base checkpoint (.safetensors)")
    parser.add_argument("--model_a", type=str, required=True, help="Path to Model A checkpoint (.safetensors)")
    parser.add_argument("--model_b", type=str, required=True, help="Path to Model B checkpoint (.safetensors)")
    parser.add_argument("--output", type=str, required=True, help="Path to write merged output (.safetensors)")

    # Optional overrides
    parser.add_argument("--lambda_k", type=float, default=Weights.lambda_k, help="Bridge weight for Kohaku delta")
    parser.add_argument("--lambda_b", type=float, default=Weights.lambda_b, help="Bridge weight for Model B delta")
    parser.add_argument("--alpha_a", type=float, default=Weights.alpha_a, help="Final weight for Model A")
    parser.add_argument("--alpha_bk", type=float, default=Weights.alpha_bk, help="Final weight for bridged Model B")

    parser.add_argument("--qk", type=float, default=Gates.qk, help="Gate multiplier for Q/K projections")
    parser.add_argument("--qkv_out", type=float, default=Gates.qkv_out, help="Gate multiplier for V/out projections")
    parser.add_argument("--ffn_conv", type=float, default=Gates.ffn_conv, help="Gate multiplier for FFN/conv")
    parser.add_argument("--time_embed_out", type=float, default=Gates.time_embed_out, help="Gate multiplier for time/embed/out")

    parser.add_argument("--target_ratio", type=float, default=NormCfg.target_ratio, help="Target RMS ratio vs base for delta normalization")
    parser.add_argument("--clip_multiple", type=float, default=NormCfg.clip_multiple, help="Clip factor for RMS scaling")
    parser.add_argument("--slerp_t", type=float, default=NormCfg.slerp_t, help="Slerp t towards base")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = Paths(
        sdxl_base=args.sdxl_base,
        kohaku_base=args.kohaku_base,
        model_a=args.model_a,
        model_b=args.model_b,
        output=args.output,
    )
    weights = Weights(
        lambda_k=args.lambda_k,
        lambda_b=args.lambda_b,
        alpha_a=args.alpha_a,
        alpha_bk=args.alpha_bk,
    )
    norm = NormCfg(
        target_ratio=args.target_ratio,
        clip_multiple=args.clip_multiple,
        slerp_t=args.slerp_t,
    )
    gates = Gates(
        qk=args.qk,
        qkv_out=args.qkv_out,
        ffn_conv=args.ffn_conv,
        time_embed_out=args.time_embed_out,
    )

    recipe = build_recipe(paths, weights=weights, norm=norm, gates=gates)
    sd_mecha.merge(recipe, output=paths.output)
    print(f"Merged model written to: {paths.output}")


if __name__ == "__main__":
    main()
