import torch
from torch import Tensor
from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return, StateDict
from align_attention import balance_head_energy, permute_and_align_heads, bundle_weight_bias, split_weight_bias


@merge_method
def sdxl_sgm_align_attention(
    a: Parameter(StateDict[Tensor], model_config="sdxl-sgm"),
    ref: Parameter(StateDict[Tensor], model_config="sdxl-sgm"),
    **kwargs,
) -> Return(Tensor):
    cache = kwargs.get("cache")
    if cache is None:
        raise RuntimeError("Merge method `sdxl_sgm_align_attention` must be used with a cache.")

    key = kwargs["key"]
    if key in cache:
        return cache.pop(key)

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

        a_q = bundle_weight_bias(a[key_q], a[key_q_bias]).unflatten(0, (-1, 64))
        a_k = bundle_weight_bias(a[key_k], a[key_k_bias]).unflatten(0, (-1, 64))
        a_v = bundle_weight_bias(a[key_v], a[key_v_bias]).unflatten(0, (-1, 64))
        a_o = a[key_o].mT.unflatten(0, (-1, 64))

        b_q = bundle_weight_bias(ref[key_q], ref[key_q_bias]).unflatten(0, (-1, 64))
        b_k = bundle_weight_bias(ref[key_k], ref[key_k_bias]).unflatten(0, (-1, 64))
        b_v = bundle_weight_bias(ref[key_v], ref[key_v_bias]).unflatten(0, (-1, 64))
        b_o = ref[key_o].mT.unflatten(0, (-1, 64))
    elif is_clip_g_attn:  # clip_g
        key_bias = key
        key_weight = key.replace("_bias", "_weight")
        key_o = key.replace(".in_proj_bias", ".out_proj.weight")

        a_w = a[key_weight]
        a_b = a[key_weight]
        b_w = ref[key_bias]
        b_b = ref[key_bias]

        dim = a_w.shape[0] // 3

        a_q = bundle_weight_bias(a_w[:dim], a_b[:dim]).unflatten(0, (-1, 64))
        a_k = bundle_weight_bias(a_w[dim:dim*2], a_b[dim:dim*2]).unflatten(0, (-1, 64))
        a_v = bundle_weight_bias(a_w[dim*2:], a_b[dim*2:]).unflatten(0, (-1, 64))
        a_o = a[key_o].mT.unflatten(0, (-1, 64))

        b_q = bundle_weight_bias(b_w[:dim], b_b[:dim]).unflatten(0, (-1, 64))
        b_k = bundle_weight_bias(b_w[dim:dim*2], b_b[dim:dim*2]).unflatten(0, (-1, 64))
        b_v = bundle_weight_bias(b_w[dim*2:], b_b[dim*2:]).unflatten(0, (-1, 64))
        b_o = ref[key_o].mT.unflatten(0, (-1, 64))
    elif is_unet_attn:  # unet
        key_q = key.replace(".to_k.", ".to_q.")
        key_k = key
        key_v = key.replace(".to_k.", ".to_v.")
        key_o = key.replace(".to_k.", ".to_out.0.")

        a_q = a[key_q].unflatten(0, (-1, 64))
        a_k = a[key_k].unflatten(0, (-1, 64))
        a_v = a[key_v].unflatten(0, (-1, 64))
        a_o = a[key_o].mT.unflatten(0, (-1, 64))

        b_q = ref[key_q].unflatten(0, (-1, 64))
        b_k = ref[key_k].unflatten(0, (-1, 64))
        b_v = ref[key_v].unflatten(0, (-1, 64))
        b_o = ref[key_o].mT.unflatten(0, (-1, 64))
    else:
        return a[key]

    a_q, a_k = balance_head_energy(a_q, a_k)
    b_q, b_k = balance_head_energy(b_q, b_k)
    a_q, a_k = permute_and_align_heads(a_q, a_k, b_q, b_k)
    del b_q, b_k

    a_v, a_o = balance_head_energy(a_v, a_o)
    b_v, b_o = balance_head_energy(b_v, b_o)
    a_v, a_o = permute_and_align_heads(a_v, a_o, b_v, b_o)
    del b_v, b_o

    if is_clip_l_attn:
        a_q, a_q_bias = split_weight_bias(a_q.flatten(end_dim=1))
        a_k, a_k_bias = split_weight_bias(a_k.flatten(end_dim=1))
        a_v, a_v_bias = split_weight_bias(a_v.flatten(end_dim=1))
        a_o = a_o.flatten(end_dim=1).mT
        cache[key_q] = a_q
        cache[key_k] = a_k
        cache[key_v] = a_v
        cache[key_o] = a_o
        cache[key_q_bias] = a_q_bias
        cache[key_v_bias] = a_v_bias
        return a_k_bias
    elif is_clip_g_attn:
        a_q, a_q_bias = split_weight_bias(a_q.flatten(end_dim=1))
        a_k, a_k_bias = split_weight_bias(a_k.flatten(end_dim=1))
        a_v, a_v_bias = split_weight_bias(a_v.flatten(end_dim=1))
        a_o = a_o.flatten(end_dim=1).mT
        weight = torch.cat((a_q, a_k, a_v), dim=0)
        bias = torch.cat((a_q_bias, a_k_bias, a_v_bias), dim=0)
        cache[key_weight] = weight
        cache[key_o] = a_o
        return bias
    else:  # is_unet_attn
        a_q = a_q.flatten(end_dim=1)
        a_k = a_k.flatten(end_dim=1)
        a_v = a_v.flatten(end_dim=1)
        a_o = a_o.flatten(end_dim=1).mT
        cache[key_q] = a_q
        cache[key_v] = a_v
        cache[key_o] = a_o
        return a_k
