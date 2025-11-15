import torch
from torch import Tensor
from typing import Optional, Tuple
from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return, StateDict
from .algorithms import balance_head_energy, permute_heads as do_permute_heads, align_heads, bundle_weight_bias, split_weight_bias


@merge_method
def sdxl_sgm_balance_heads_energy(
    a: Parameter(StateDict[Tensor], model_config="sdxl-sgm"),
    **kwargs,
) -> Return(Tensor, model_config="sdxl-sgm"):
    cache = kwargs.get("cache")
    if cache is None:
        raise RuntimeError("Merge method `sdxl_sgm_balance_heads_energy` must be used with a cache.")

    key = kwargs["key"]
    if key in cache:
        return cache.pop(key)

    qkvo = try_fetch_qkvo(a, key)
    if qkvo is None:
        return a[key]

    q, k, v, o = qkvo
    q, k = balance_head_energy(q, k)
    v, o = balance_head_energy(v, o)
    return return_qkvo(q, k, v, o, key, cache)


@merge_method
def sdxl_sgm_align_attention(
    a: Parameter(StateDict[Tensor], model_config="sdxl-sgm"),
    ref: Parameter(StateDict[Tensor], model_config="sdxl-sgm"),
    permute_heads: Parameter(bool, model_config="sdxl-sgm") = False,
    **kwargs,
) -> Return(Tensor, model_config="sdxl-sgm"):
    cache = kwargs.get("cache")
    if cache is None:
        raise RuntimeError("Merge method `sdxl_sgm_align_attention` must be used with a cache.")

    key = kwargs["key"]
    if key in cache:
        return cache.pop(key)

    a_qkvo = try_fetch_qkvo(a, key)
    if a_qkvo is None:
        return a[key]

    a_q, a_k, a_v, a_o = a_qkvo
    b_q, b_k, b_v, b_o = try_fetch_qkvo(ref, key)

    if permute_heads:
        a_q, a_k, a_v, a_o = do_permute_heads(a_q, a_k, a_v, a_o, b_q, b_k, b_v, b_o)

    a_q, a_k = align_heads(a_q, a_k, b_q, b_k)
    a_v, a_o = align_heads(a_v, a_o, b_v, b_o)

    return return_qkvo(a_q, a_k, a_v, a_o, key, cache)


def try_fetch_qkvo(
    sd: StateDict[Tensor],
    key: str,
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
    is_clip_l_attn = key.endswith(".k_proj.bias")
    is_clip_g_attn = key.endswith("in_proj_bias")
    is_unet_attn = key.endswith(".to_k.weight")

    if is_clip_l_attn:  # clip_l
        key_q_bias = key.replace(".k_proj.", ".q_proj.")
        key_k_bias = key
        key_v_bias = key.replace(".k_proj.", ".v_proj.")
        key_q = key_q_bias.replace(".bias", ".weight")
        key_k = key_k_bias.replace(".bias", ".weight")
        key_v = key_v_bias.replace(".bias", ".weight")
        key_o = key_k.replace(".k_proj.", ".out_proj.")

        k_bias, k = sd[key_k_bias], sd[key_k]
        o = sd[key_o]
        q_bias, q = sd[key_q_bias], sd[key_q]
        v_bias, v = sd[key_v_bias], sd[key_v]

        k = bundle_weight_bias(k, k_bias).unflatten(0, (-1, 64))
        q = bundle_weight_bias(q, q_bias).unflatten(0, (-1, 64))
        v = bundle_weight_bias(v, v_bias).unflatten(0, (-1, 64))
        o = bundle_weight_bias(o.mT, torch.zeros_like(o.mT[..., -1])).unflatten(0, (-1, 64))
    elif is_clip_g_attn:  # clip_g
        key_bias = key
        key_weight = key.replace("_bias", "_weight")
        key_o = key.replace(".in_proj_bias", ".out_proj.weight")

        bias = sd[key_bias]
        weight = sd[key_weight]
        o = sd[key_o]

        dim = weight.shape[0] // 3

        q = bundle_weight_bias(weight[:dim], bias[:dim]).unflatten(0, (-1, 64))
        k = bundle_weight_bias(weight[dim:dim*2], bias[dim:dim*2]).unflatten(0, (-1, 64))
        v = bundle_weight_bias(weight[dim*2:], bias[dim*2:]).unflatten(0, (-1, 64))
        o = bundle_weight_bias(o.mT, torch.zeros_like(o.mT[..., -1])).unflatten(0, (-1, 64))
    elif is_unet_attn:  # unet
        key_q = key.replace(".to_k.", ".to_q.")
        key_k = key
        key_v = key.replace(".to_k.", ".to_v.")
        key_o = key.replace(".to_k.", ".to_out.0.")

        k = sd[key_k].unflatten(0, (-1, 64))
        o = sd[key_o].mT.unflatten(0, (-1, 64))
        q = sd[key_q].unflatten(0, (-1, 64))
        v = sd[key_v].unflatten(0, (-1, 64))

    else:
        return None

    return q, k, v, o


def return_qkvo(q, k, v, o, key, cache):
    is_clip_l_attn = key.endswith(".k_proj.bias")
    is_clip_g_attn = key.endswith("in_proj_bias")

    if is_clip_l_attn:
        key_q_bias = key.replace(".k_proj.", ".q_proj.")
        key_k_bias = key
        key_v_bias = key.replace(".k_proj.", ".v_proj.")
        key_q = key_q_bias.replace(".bias", ".weight")
        key_k = key_k_bias.replace(".bias", ".weight")
        key_v = key_v_bias.replace(".bias", ".weight")
        key_o = key_k.replace(".k_proj.", ".out_proj.")

        q, q_bias = split_weight_bias(q.flatten(end_dim=1))
        k, k_bias = split_weight_bias(k.flatten(end_dim=1))
        v, v_bias = split_weight_bias(v.flatten(end_dim=1))
        o = split_weight_bias(o.flatten(end_dim=1))[0].mT
        cache[key_q] = q
        cache[key_k] = k
        cache[key_v] = v
        cache[key_o] = o
        cache[key_q_bias] = q_bias
        cache[key_v_bias] = v_bias
        return k_bias
    elif is_clip_g_attn:
        key_weight = key.replace("_bias", "_weight")
        key_o = key.replace(".in_proj_bias", ".out_proj.weight")

        q, q_bias = split_weight_bias(q.flatten(end_dim=1))
        k, k_bias = split_weight_bias(k.flatten(end_dim=1))
        v, v_bias = split_weight_bias(v.flatten(end_dim=1))
        o = split_weight_bias(o.flatten(end_dim=1))[0].mT
        weight = torch.cat((q, k, v), dim=0)
        bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
        cache[key_weight] = weight
        cache[key_o] = o
        return bias
    else:  # is_unet_attn
        key_q = key.replace(".to_k.", ".to_q.")
        key_v = key.replace(".to_k.", ".to_v.")
        key_o = key.replace(".to_k.", ".to_out.0.")

        q = q.flatten(end_dim=1)
        k = k.flatten(end_dim=1)
        v = v.flatten(end_dim=1)
        o = o.flatten(end_dim=1).mT
        cache[key_q] = q
        cache[key_v] = v
        cache[key_o] = o
        return k
