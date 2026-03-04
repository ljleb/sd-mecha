import re
import torch
from collections import defaultdict
from torch import Tensor
from typing import Optional, Tuple
from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return, StateDict
from sd_mecha.keys_map import KeyMapBuilder
from .algorithms import balance_head_energy, permute_heads as do_permute_heads, align_heads, bundle_weight_bias, split_weight_bias
from .interface import balance_attention_energy, align_attention


def implement_merge_methods(config, base, extract_qkvo, return_qkvo):
    @merge_method(implements=balance_attention_energy)
    class sdxl_sgm_balance_attention_energy(base):
        def __call__(
            self,
            a: Parameter(StateDict[Tensor], model_config=config),
            **kwargs,
        ) -> Return(Tensor, model_config=config):
            keys = kwargs["key_relation"].outputs

            qkvo = extract_qkvo(a, keys)
            if qkvo is None:
                return a[keys[0]]

            q, k, v, o = qkvo
            q, k = balance_head_energy(q, k)
            v, o = balance_head_energy(v, o)
            return return_qkvo(q, k, v, o, keys)

    @merge_method(implements=align_attention)
    class sdxl_sgm_align_attention(base):
        def __call__(
            self,
            a: Parameter(StateDict[Tensor], model_config=config),
            ref: Parameter(StateDict[Tensor], model_config=config),
            permute_heads: Parameter(bool, model_config=config) = False,
            **kwargs,
        ) -> Return(StateDict[Tensor], model_config=config):
            keys = kwargs["key_relation"].outputs
            a_qkvo = extract_qkvo(a, keys)
            if a_qkvo is None:
                return a[keys[0]]

            a_q, a_k, a_v, a_o = a_qkvo
            b_q, b_k, b_v, b_o = extract_qkvo(ref, keys)

            if permute_heads:
                a_q, a_k, a_v, a_o = do_permute_heads(a_q, a_k, a_v, a_o, b_q, b_k, b_v, b_o)

            a_q, a_k = align_heads(a_q, a_k, b_q, b_k)
            a_v, a_o = align_heads(a_v, a_o, b_v, b_o)

            return return_qkvo(a_q, a_k, a_v, a_o, keys)


class SdxlSgmAttentionBase:
    clip_l_re = re.compile(r"layers\.(\d+)\.self_attn\.(.+)")
    clip_g_re = re.compile(r"resblocks\.(\d+)\.attn\.(.+)")
    vae_re = re.compile(r"\.([a-z]+)\.mid\.attn_1\.(.+)")
    unet_re = re.compile(r"\.([a-z]+)_blocks?\.(\d+\.)(?:\d+\.)?transformer_blocks\.(\d+)\.attn[12]\.(.+)")

    @classmethod
    def map_keys(cls, b: KeyMapBuilder):
        attention_keys = defaultdict(list)
        for key in b.keys():
            if (match := cls.clip_l_re.search(key)) and match.group(2) != "out_proj.bias":
                layer_id = ("clip_l", match.group(1))
                attention_keys[layer_id].append(key)
            elif (match := cls.clip_g_re.search(key)) and match.group(2) != "out_proj.bias":
                layer_id = ("clip_g", match.group(1))
                attention_keys[layer_id].append(key)
            elif (match := cls.vae_re.search(key)) and match.group(2) != "proj_out.bias":
                layer_id = ("vae", match.group(1), match.group(2))
                attention_keys[layer_id].append(key)
            elif (match := cls.unet_re.search(key)) and match.group(4) != "to_out.0.bias":
                layer_id = ("unet", match.group(1), match.group(2), match.group(3))
                attention_keys[layer_id].append(key)
            else:
                b[key] = b.keys[key]

        for group in attention_keys.values():
            b[tuple(group)] = b.keys[tuple(group)]


def extract_qkvo_sdxl_sgm(
    sd: StateDict[Tensor],
    keys: Tuple[str, ...],
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
    if len(keys) == 1:
        return None

    if is_clip_l(keys[0]) or is_vae(keys[0]):
        (
            key_k_bias, key_k,
            key_o,
            key_q_bias, key_q,
            key_v_bias, key_v,
        ) = keys

        k_bias, k = sd[key_k_bias], sd[key_k]
        o = sd[key_o]
        q_bias, q = sd[key_q_bias], sd[key_q]
        v_bias, v = sd[key_v_bias], sd[key_v]

        k = bundle_weight_bias(k, k_bias).unflatten(0, (-1, 64))
        q = bundle_weight_bias(q, q_bias).unflatten(0, (-1, 64))
        v = bundle_weight_bias(v, v_bias).unflatten(0, (-1, 64))
        o = bundle_weight_bias(o.mT, torch.zeros_like(o.mT[..., -1])).unflatten(0, (-1, 64))
    elif is_clip_g(keys[0]):
        (
            key_bias, key_weight,
            key_o,
        ) = keys

        bias = sd[key_bias]
        weight = sd[key_weight]
        o = sd[key_o]

        dim = weight.shape[0] // 3

        q = bundle_weight_bias(weight[:dim], bias[:dim]).unflatten(0, (-1, 64))
        k = bundle_weight_bias(weight[dim:dim*2], bias[dim:dim*2]).unflatten(0, (-1, 64))
        v = bundle_weight_bias(weight[dim*2:], bias[dim*2:]).unflatten(0, (-1, 64))
        o = bundle_weight_bias(o.mT, torch.zeros_like(o.mT[..., -1])).unflatten(0, (-1, 64))
    elif is_unet(keys[0]):
        key_k, key_o, key_q, key_v = keys

        k = sd[key_k].unflatten(0, (-1, 64))
        o = sd[key_o].mT.unflatten(0, (-1, 64))
        q = sd[key_q].unflatten(0, (-1, 64))
        v = sd[key_v].unflatten(0, (-1, 64))
    else:
        raise KeyError(keys)

    return q, k, v, o


def return_qkvo_sdxl_sgm(q, k, v, o, keys):
    if is_clip_l(keys[0]) or is_vae(keys[0]):
        (
            key_k_bias, key_k,
            key_o,
            key_q_bias, key_q,
            key_v_bias, key_v,
        ) = keys

        k, k_bias = split_weight_bias(k.flatten(end_dim=1))
        o = split_weight_bias(o.flatten(end_dim=1))[0].mT
        q, q_bias = split_weight_bias(q.flatten(end_dim=1))
        v, v_bias = split_weight_bias(v.flatten(end_dim=1))

        result = {
            key_k_bias: k_bias, key_k: k,
            key_o: o,
            key_q_bias: q_bias, key_q: q,
            key_v_bias: v_bias, key_v: v,
        }
    elif is_clip_g(keys[0]):
        (
            key_bias, key_weight,
            key_o,
        ) = keys

        k, k_bias = split_weight_bias(k.flatten(end_dim=1))
        o = split_weight_bias(o.flatten(end_dim=1))[0].mT
        q, q_bias = split_weight_bias(q.flatten(end_dim=1))
        v, v_bias = split_weight_bias(v.flatten(end_dim=1))

        weight = torch.cat((q, k, v), dim=0)
        bias = torch.cat((q_bias, k_bias, v_bias), dim=0)

        result = {
            key_bias: bias, key_weight: weight,
            key_o: o,
        }
    elif is_unet(keys[0]):
        (
            key_k,
            key_o,
            key_q,
            key_v,
        ) = keys

        k = k.flatten(end_dim=1)
        o = o.flatten(end_dim=1).mT
        q = q.flatten(end_dim=1)
        v = v.flatten(end_dim=1)

        result = {
            key_k: k,
            key_o: o,
            key_q: q,
            key_v: v,
        }
    else:
        raise KeyError(keys)

    return result


implement_merge_methods("sdxl-sgm", SdxlSgmAttentionBase, extract_qkvo_sdxl_sgm, return_qkvo_sdxl_sgm)


def is_clip_l(k):
    return k.startswith("conditioner.embedders.0")


def is_clip_g(k):
    return k.startswith("conditioner.embedders.1")


def is_vae(k):
    return k.startswith("first_stage_model")


def is_unet(k):
    return k.startswith("model.diffusion_model")
