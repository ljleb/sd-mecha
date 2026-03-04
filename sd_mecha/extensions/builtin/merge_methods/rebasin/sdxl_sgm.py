import functools
import torch
from typing import Set, Dict, Tuple
from torch import Tensor
from scipy.optimize import linear_sum_assignment
from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return, StateDict
from sd_mecha.extensions import model_configs
from sd_mecha.keys_map import KeyMapBuilder
from sd_mecha.streaming import TensorMetadata
from .perm_graph import PermGraph, PermSlice, Permutation
from .interface import rebasin, randperm


sdxl_config = model_configs.resolve("sdxl-sgm")


def create_graph():
    metadata = sdxl_config.metadata()
    perm_graph = PermGraph(metadata)
    for key, meta in metadata.items():
        num_heads = _num_attention_heads(key)

        if key.startswith("conditioner.embedders.0"):
            clip_l_perms(key, meta, perm_graph, num_heads)

        if key.startswith("conditioner.embedders.1"):
            clip_g_perms(key, meta, perm_graph, num_heads)

        if key.startswith("model.diffusion_model"):
            unet_perms(key, meta, perm_graph, num_heads)

    return perm_graph


def clip_l_perms(key: str, meta: TensorMetadata, perm_graph: PermGraph, num_heads: int):
    if ".q_proj." in key and key.endswith("weight"):  # qk
        head_width = meta.shape[0] // num_heads

        q_key = key
        q_bias_key = q_key.replace(".weight", ".bias")
        k_key = q_key.replace(".q_proj.", ".k_proj.")
        k_bias_key = k_key.replace(".weight", ".bias")

        for start_i in range(0, meta.shape[0], head_width):
            slice = PermSlice(0, start_i)
            perm_graph.add(Permutation(head_width, {
                k: slice for k in (q_key, q_bias_key, k_key, k_bias_key)
            }))

    if ".v_proj." in key and key.endswith("weight"):  # vo
        head_width = meta.shape[0] // num_heads

        v_key = key
        v_bias_key = v_key.replace(".weight", ".bias")
        out_key = v_key.replace(".v_proj.", ".out_proj.")

        for start_i in range(0, meta.shape[0], head_width):
            v_slice = PermSlice(0, start_i)
            perm_graph.add(Permutation(head_width, {
                v_key: v_slice,
                v_bias_key: v_slice,
                out_key: PermSlice(1, start_i),
            }))

    if ".mlp.fc1" in key and key.endswith(".weight"):  # ff
        perm_width = meta.shape[0]

        fc1_key = key
        fc1_bias_key = fc1_key.replace(".weight", ".bias")
        fc2_key = fc1_key.replace(".fc1.", ".fc2.")

        fc1_slice = PermSlice(0)
        fc2_slice = PermSlice(1)
        perm_graph.add(Permutation(perm_width, {
            fc1_key: fc1_slice,
            fc1_bias_key: fc1_slice,
            fc2_key: fc2_slice,
        }))


def clip_g_perms(key: str, meta: TensorMetadata, perm_graph: PermGraph, num_heads: int):
    if ".in_proj_" in key and key.endswith("weight"):  # qkvo
        unfused_size = meta.shape[0] // 3
        head_width = unfused_size // num_heads
        k_offset = unfused_size
        v_offset = k_offset * 2

        in_key = key
        in_bias_key = key.replace("_weight", "_bias")
        out_key = key.replace(".in_proj_weight", ".out_proj.weight")

        for start_i in range(0, unfused_size, head_width):
            # qk
            q_slice = PermSlice(0, start_i)
            k_slice = PermSlice(0, start_i + k_offset)
            perm_graph.add(Permutation(head_width, {
                in_key: (q_slice, k_slice),
                in_bias_key: (q_slice, k_slice),
            }))

            # vo
            v_slice = PermSlice(0, start_i + v_offset)
            out_slice = PermSlice(1, start_i)
            perm_graph.add(Permutation(head_width, {
                in_key: v_slice,
                in_bias_key: v_slice,
                out_key: out_slice,
            }))

    if ".mlp.c_fc." in key and key.endswith(".weight"):  # ff
        perm_width = meta.shape[0]

        fc1_key = key
        fc1_bias_key = key.replace(".weight", ".bias")
        fc2_key = key.replace(".c_fc.", ".c_proj.")

        fc1_slice = PermSlice(0)
        fc2_slice = PermSlice(1)
        perm_graph.add(Permutation(perm_width, {
            fc1_key: fc1_slice,
            fc1_bias_key: fc1_slice,
            fc2_key: fc2_slice,
        }))


def unet_perms(key: str, meta: TensorMetadata, perm_graph: PermGraph, num_heads: int):
    if ".to_q." in key and key.endswith("weight"):  # unet qk
        head_width = meta.shape[0] // num_heads

        q_key = key
        k_key = key.replace(".to_q.", ".to_k.")

        for start_i in range(0, meta.shape[0], head_width):
            slice = PermSlice(0, start_i)
            perm_graph.add(Permutation(head_width, {
                k: slice for k in (q_key, k_key)
            }))

    if ".to_v." in key and key.endswith("weight"):  # unet vo
        head_width = meta.shape[0] // num_heads

        v_key = key
        out_key = key.replace(".to_v.", ".to_out.0.")

        for start_i in range(0, meta.shape[0], head_width):
            v_slice = PermSlice(0, start_i)
            out_slice = PermSlice(1, start_i)
            perm_graph.add(Permutation(head_width, {
                v_key: v_slice,
                out_key: out_slice,
            }))

    if ".ff.net.2." in key and key.endswith(".weight"):  # unet ff
        perm_width = meta.shape[1]

        fc1_key = key.replace(".ff.net.2.", ".ff.net.0.")
        fc1_bias_key = fc1_key.replace(".weight", ".bias")
        fc2_key = key

        fc1_slices = PermSlice(0), PermSlice(0, perm_width)
        fc2_slice = PermSlice(1)
        perm_graph.add(Permutation(perm_width, {
            fc1_key: fc1_slices,
            fc1_bias_key: fc1_slices,
            fc2_key: fc2_slice,
        }))

    if ".time_embed.0." in key and key.endswith(".weight"):  # unet timestep embed
        perm_width = meta.shape[0]

        fc1_key = key
        fc1_bias_key = fc1_key.replace(".weight", ".bias")
        fc2_key = key.replace(".time_embed.0.", ".time_embed.2.")

        fc1_slice = PermSlice(0)
        fc2_slice = PermSlice(1)
        perm_graph.add(Permutation(perm_width, {
            fc1_key: fc1_slice,
            fc1_bias_key: fc1_slice,
            fc2_key: fc2_slice,
        }))

    if ".label_emb.0.0." in key and key.endswith(".weight"):  # unet label embed
        perm_width = meta.shape[0]

        fc1_key = key
        fc1_bias_key = key.replace(".weight", ".bias")
        fc2_key = key.replace(".label_emb.0.0.", ".label_emb.0.2.")

        fc1_slice = PermSlice(0)
        fc2_slice = PermSlice(1)
        perm_graph.add(Permutation(perm_width, {
            fc1_key: fc1_slice,
            fc1_bias_key: fc1_slice,
            fc2_key: fc2_slice,
        }))


def _num_attention_heads(key: str) -> int:
    if key.startswith("conditioner.embedders.0"):
        return 12
    if key.startswith("conditioner.embedders.1"):
        return 20
    return 64


perm_graph = create_graph()


@merge_method(implements=rebasin)
class rebasin_sdxl:
    @staticmethod
    def map_keys(b: KeyMapBuilder):
        used: Set[str] = set()

        for closure in perm_graph.closures():
            b[closure.keys] = b.keys[closure.keys] @ closure
            used.update(closure.keys)

        for k in b.keys():
            if k not in used:
                b[k] = b.keys[k] @ None

    def __call__(
        self,
        a: Parameter(StateDict[Tensor], model_config=sdxl_config),
        ref: Parameter(StateDict[Tensor], model_config=sdxl_config),
        iters: Parameter(int),
        **kwargs,
    ) -> Return(StateDict[Tensor], model_config=sdxl_config):
        out_keys, closure = _get_request(kwargs)

        if closure is None:
            return {k: a[k] for k in out_keys}

        load_a = _make_loader(a)
        load_ref = _make_loader(ref)
        device, dtype = _infer_device_dtype(load_a, closure)

        pid_perm = _init_pid_perm(closure, device)
        _solve_all_components(closure, pid_perm, load_a, load_ref, device, dtype, iters)
        out_all = _apply_inverse_all(closure, pid_perm, load_a)

        return {k: out_all[k] for k in out_keys}


def _infer_device_dtype(load_a, closure):
    rep_key = closure.keys[0]
    t = load_a(rep_key)
    return t.device, t.dtype


def _solve_all_components(
    closure,
    pid_perm: Dict[int, Tensor],
    load_a,
    load_ref,
    device,
    dtype,
    iters: int,
) -> None:
    for comp in closure.components:
        _solve_component(closure, comp, pid_perm, load_a, load_ref, device, dtype, iters)


def _solve_component(
    closure,
    comp: Tuple[int, ...],
    pid_perm: Dict[int, Tensor],
    load_a,
    load_ref,
    device,
    dtype,
    iters: int,
) -> None:
    apps_by_key = _apps_by_key_for_component(closure, comp)

    for _ in range(max(1, iters)):
        changed = False

        for pid in comp:
            new_p = _update_one_pid(
                closure, pid, pid_perm, apps_by_key, load_a, load_ref, device, dtype
            )
            if not torch.equal(new_p, pid_perm[pid]):
                pid_perm[pid] = new_p
                changed = True

        if not changed:
            break


def _apps_by_key_for_component(closure, comp: Tuple[int, ...]):
    apps_by_key: Dict[str, Tuple[Tuple[int, int, int], ...]] = {}
    comp_set = set(comp)

    for k in closure.keys:
        apps = closure.key_to_apps.get(k)
        if not apps:
            continue
        filt = tuple(app for app in apps if app[0] in comp_set)
        if filt:
            apps_by_key[k] = filt

    return apps_by_key


def _update_one_pid(
    closure,
    pid: int,
    pid_perm: Dict[int, Tensor],
    apps_by_key: Dict[str, Tuple[Tuple[int, int, int], ...]],
    load_a,
    load_ref,
    device,
    dtype,
) -> Tensor:
    w = closure.pid_to_width[pid]
    nodes = closure.pid_to_nodes[pid]

    cost = torch.zeros((w, w), device=device, dtype=dtype)

    for n in nodes:
        k = n.key
        axis_cur = _norm_axis(n.axis, load_a(k).dim())
        offset = n.offset

        tensor_a = _a_vec(load_a(k), axis_cur, offset, w)
        tensor_b = _b_vec(load_ref(k), apps_by_key.get(k, ()), axis_cur, offset, w, pid_perm, closure)
        cost.addmm_(tensor_a, tensor_b.T)

    return _solve_lap_max(cost)


def _a_vec(tensor_a: Tensor, axis_cur: int, offset: int, w: int) -> Tensor:
    a_window = _slice_window(tensor_a, axis_cur, offset, w)
    return _front_flat(a_window, axis_cur)


def _b_vec(
    tensor_ref: Tensor,
    apps: Tuple[Tuple[int, int, int], ...],
    axis_cur: int,
    offset: int,
    width: int,
    pid_perm: Dict[int, Tensor],
    closure,
) -> Tensor:
    b_ref_other = _apply_other_axis_perms(tensor_ref, apps, axis_cur, pid_perm, closure)
    b_window = _slice_window(b_ref_other, axis_cur, offset, width)
    return _front_flat(b_window, axis_cur)


def _apply_other_axis_perms(
    tensor_ref: Tensor,
    apps: Tuple[Tuple[int, int, int], ...],
    axis_cur: int,
    pid_perm: Dict[int, Tensor],
    closure,
) -> Tensor:
    # Clone only if we actually apply something
    out = tensor_ref
    cloned = False

    for pid2, ax2, off2 in apps:
        ax2n = _norm_axis(ax2, tensor_ref.dim())
        if ax2n == axis_cur:
            continue

        if not cloned:
            out = tensor_ref.clone()
            cloned = True

        _apply_window_perm_inplace(
            out,
            axis=ax2n,
            offset=off2,
            width=closure.pid_to_width[pid2],
            perm=pid_perm[pid2],
        )

    return out


def _apply_inverse_all(closure, pid_perm: Dict[int, Tensor], load_a) -> Dict[str, Tensor]:
    out_all: Dict[str, Tensor] = {}

    for k in closure.keys:
        tensor_a = load_a(k)
        apps = closure.key_to_apps.get(k)
        if not apps:
            out_all[k] = tensor_a
            continue

        out_all[k] = _apply_inverse_one(tensor_a, apps, closure, pid_perm)

    return out_all


def _apply_inverse_one(
    tensor_a: Tensor,
    apps: Tuple[Tuple[int, int, int], ...],
    closure,
    pid_perm: Dict[int, Tensor],
) -> Tensor:
    out = tensor_a.clone()
    for pid, ax, off in apps:
        w = closure.pid_to_width[pid]
        inv = _invert_perm(pid_perm[pid])
        _apply_window_perm_inplace(out, axis=ax, offset=off, width=w, perm=inv)
    return out


@merge_method(implements=randperm)
class randperm_sdxl:
    @staticmethod
    def map_keys(b: KeyMapBuilder):
        used: Set[str] = set()

        for closure in perm_graph.closures():
            b[closure.keys] = b.keys[closure.keys] @ closure
            used.update(closure.keys)

        for k in b.keys():
            if k not in used:
                b[k] = b.keys[k] @ None

    def __call__(
        self,
        a: Parameter(StateDict[Tensor], model_config=sdxl_config),
        seed: Parameter(int) = None,
        **kwargs,
    ) -> Return(StateDict[Tensor], model_config=sdxl_config):
        out_keys, closure = _get_request(kwargs)

        if closure is None:
            return {k: a[k] for k in out_keys}

        load_a = _make_loader(a)
        device = load_a(closure.keys[0]).device

        pid_perm = _init_rand_perm(closure, seed, device)
        out_all = _apply_all_perms(closure, pid_perm, load_a)

        return {k: out_all[k] for k in out_keys}


def _get_request(kwargs) -> Tuple[Tuple[str, ...], "PermClosure | None"]:
    rel = kwargs["key_relation"]
    return rel.outputs, rel.meta


def _make_loader(a):
    return functools.partial(_load, memo={}, sd=a)


def _init_pid_perm(closure, device) -> Dict[int, Tensor]:
    pid_perm: Dict[int, Tensor] = {}

    for pid in closure.perms:
        w = closure.pid_to_width[pid]
        pid_perm[pid] = torch.arange(w, dtype=torch.long, device=device)

    return pid_perm


def _init_rand_perm(closure, seed: int | None, device) -> Dict[int, Tensor]:
    pid_perm: Dict[int, Tensor] = {}

    for pid in closure.perms:
        w = closure.pid_to_width[pid]
        gen = _make_generator(seed, pid, device)
        pid_perm[pid] = torch.randperm(w, generator=gen, device=device)

    return pid_perm


def _make_generator(seed: int | None, pid: int, device):
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) + int(pid))
    return g


def _apply_all_perms(closure, pid_perm: Dict[int, Tensor], load_a) -> Dict[str, Tensor]:
    out_all: Dict[str, Tensor] = {}

    for k in closure.keys:
        t = load_a(k)
        apps = closure.key_to_apps.get(k)
        if not apps:
            out_all[k] = t
            continue

        out_all[k] = _apply_one_key(t, apps, closure, pid_perm)

    return out_all


def _apply_one_key(
    t: Tensor,
    apps: Tuple[Tuple[int, int, int], ...],
    closure,
    pid_perm: Dict[int, Tensor],
) -> Tensor:
    out = t.clone()
    for pid, ax, off in apps:
        w = closure.pid_to_width[pid]
        _apply_window_perm_inplace(
            out,
            axis=ax,
            offset=off,
            width=w,
            perm=pid_perm[pid],
        )
    return out


def _load(k: str, memo, sd) -> Tensor:
    t = memo.get(k)
    if t is None:
        t = sd[k]
        memo[k] = t
    return t


def _front_flat(t: Tensor, axis: int) -> Tensor:
    t = torch.movedim(t, axis, 0)
    return t.reshape(t.shape[0], -1).contiguous()


def _solve_lap_max(sim: Tensor) -> Tensor:
    """Return perm p s.t. sum_i sim[i, p[i]] is maximized. sim on CPU."""
    if sim.dtype.itemsize < 2:
        sim = sim.float()
    r, c = linear_sum_assignment((-sim).numpy(force=True))
    return torch.as_tensor(c, device=sim.device, dtype=torch.long)


def _apply_window_perm_inplace(
    t: Tensor,
    *,
    axis: int,
    offset: int,
    width: int,
    perm: Tensor,
) -> None:
    ax = _norm_axis(axis, t.dim())
    win = _slice_window(t, ax, offset, width)
    winp = torch.index_select(win, ax, perm)
    idx = [slice(None)] * t.dim()
    idx[ax] = slice(offset, offset + width)
    t[tuple(idx)] = winp


def _slice_window(t: Tensor, axis: int, offset: int, width: int) -> Tensor:
    ax = _norm_axis(axis, t.dim())
    idx = [slice(None)] * t.dim()
    idx[ax] = slice(offset, offset + width)
    return t[tuple(idx)]


def _norm_axis(axis: int, rank: int) -> int:
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError(f"axis {axis} out of range for rank {rank}")
    return axis


def _invert_perm(p: Tensor) -> Tensor:
    inv = torch.empty_like(p)
    inv[p] = torch.arange(p.numel(), device=p.device, dtype=p.dtype)
    return inv
