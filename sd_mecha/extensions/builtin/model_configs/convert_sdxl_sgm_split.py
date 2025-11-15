import torch
from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return, StateDict
from sd_mecha.streaming import StateDictKeyError


def _n_heads(key: str) -> int:
    if key.startswith("conditioner.embedders.0"):
        return 12
    if key.startswith("conditioner.embedders.1"):
        return 20
    return 64


@merge_method(is_conversion=True)
def to_sdxl_sgm_split(
    sgm: Parameter(StateDict[torch.Tensor], model_config="sdxl-sgm"),
    **kwargs,
) -> Return(torch.Tensor, model_config="sdxl-sgm_split"):
    key = kwargs["key"]

    if key in sgm:
        return sgm[key]

    parts = key.split(".")

    if "attn" in key:
        head = int(parts[-2])
        wb = parts[-1]
        token = parts[-3]  # 'to_q' | 'k_proj' | '0' for to_out | 'out_proj'...

        # ---------- Q/K/V ----------
        if token in ("to_q", "to_k", "to_v", "q_proj", "k_proj", "v_proj"):
            which = {"to_q": "q", "to_k": "k", "to_v": "v", "q_proj": "q", "k_proj": "k", "v_proj": "v"}[token]
            base = ".".join(parts[:-3])  # drop .{tok}.{head}.{wb}

            # 1) try fused in_proj_*
            fused_key = f"{base}.in_proj_{wb}"
            if fused_key in sgm:
                fused = sgm[fused_key]          # (3*D, Din) or (3*D,)
                n = _n_heads(key)
                threeD = fused.shape[0]
                D = threeD // 3
                d = D // n
                off = {"q": 0, "k": 1, "v": 2}[which] * D
                s = off + head * d
                e = off + (head + 1) * d
                return fused[s:e]

            # 2) CLIP-L *_proj
            proj_key = f"{base}.{which}_proj.{wb}"
            if proj_key in sgm:
                t = sgm[proj_key]               # rows contain heads
                n = _n_heads(key)
                d = t.shape[0] // n
                return t[head*d:(head+1)*d]

            # 3) UNet to_{qkv}
            to_key = f"{base}.to_{which}.{wb}"
            t = sgm[to_key]
            n = _n_heads(key)
            d = t.shape[0] // n
            return t[head*d:(head+1)*d]

        # ---------- OUT (to_out.N / out_proj) ----------
        base_no_head = ".".join(parts[:-2])  # keep to_out.N or out_proj
        src = f"{base_no_head}.{wb}"
        if src in sgm:
            t = sgm[src]
            n = _n_heads(key)
            if t.ndim == 1:
                d = t.shape[0] // n
                return t[head*d:(head+1)*d]
            d = t.shape[1] // n
            return t[:, head*d:(head+1)*d]

    if ".output_blocks." in key and key.endswith(".weight") and (
        ".in_layers.2." in key or ".skip_connection." in key
    ):
        if key.endswith(".0.weight"):
            half = 0
            fused_key = key[:-len(".0.weight")] + ".weight"
        elif key.endswith(".1.weight"):
            half = 1
            fused_key = key[:-len(".1.weight")] + ".weight"
        else:
            raise KeyError(f"Unexpected IL2/skip key: {key}")

        W = sgm[fused_key]
        Cout, Cin = W.shape[:2]
        if Cout * 2 >= Cin:
            s = Cout * half
            e = Cout * (half + 1)
        else:
            s = (2 * Cout) * half
            e = (2 * Cout) * (half + 1)
        return W[:, s:e]

    if ".ff.net.0.proj." in key:
        wb = parts[-1]
        half = 0 if parts[-2] == "0" else 1
        fused_key = ".".join(parts[:-2] + [wb])   # drop .{half}.{wb} -> .{wb}
        fused = sgm[fused_key]
        D2 = fused.shape[0] // 2
        return fused[half*D2:(half+1)*D2]

    raise StateDictKeyError(key)


# =============================================================================
# sdxl-sgm_split → sdxl-sgm
# =============================================================================
@merge_method(is_conversion=True)
def to_sdxl_sgm(
    sgm_split: Parameter(StateDict[torch.Tensor], model_config="sdxl-sgm_split"),
    **kwargs,
) -> Return(torch.Tensor, model_config="sdxl-sgm"):
    key = kwargs["key"]

    # pass-through
    if key in sgm_split:
        return sgm_split[key]

    parts = key.split(".")
    wb = parts[-1] if parts else None

    # -------- fused in_proj_* <= concat q/k/v heads (rows) --------
    if key.endswith("in_proj_weight") or key.endswith("in_proj_bias"):
        wb = "weight" if key.endswith("weight") else "bias"
        base = key[: - (len("in_proj_") + len(wb) + 1)]
        n = _n_heads(key)

        def cat(which: str) -> torch.Tensor:
            return torch.cat([sgm_split[f"{base}.to_{which}.{h}.{wb}"] for h in range(n)], dim=0)

        return torch.cat((cat("q"), cat("k"), cat("v")), dim=0)

    if any(tok in key for tok in (".to_q.", ".to_k.", ".to_v.", ".q_proj.", ".k_proj.", ".v_proj.")):
        base = ".".join(parts[:-1])
        n = _n_heads(key)
        heads = [sgm_split[f"{base}.{h}.{wb}"] for h in range(n)]
        return torch.cat(heads, dim=0)

    # -------- to_out.N / out_proj --------
    if ".to_out." in key or ".out_proj." in key:
        base = ".".join(parts[:-1])
        n = _n_heads(key)
        heads = [sgm_split[f"{base}.{h}.{wb}"] for h in range(n)]
        if wb == "bias":
            return torch.cat(heads, dim=0)
        return torch.cat(heads, dim=1)

    # -------- IL2 / skip re-fuse: concat columns --------
    if ".output_blocks." in key and key.endswith(".weight") and (
        ".in_layers.2." in key or ".skip_connection." in key
    ):
        a = sgm_split[key.replace(".weight", ".0.weight")]
        b = sgm_split[key.replace(".weight", ".1.weight")]
        return torch.cat((a, b), dim=1)

    # -------- FF fuse: concat rows --------
    if ".ff.net.0.proj." in key:
        base = ".".join(parts[:-1])
        a = sgm_split[f"{base}.0.{wb}"]
        b = sgm_split[f"{base}.1.{wb}"]
        return torch.cat((a, b), dim=0)

    raise KeyError(f"No rule for {key!r} (split → sgm)")
