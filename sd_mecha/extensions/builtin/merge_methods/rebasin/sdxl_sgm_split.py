import threading
import torch
from typing import Set, Dict, List, Tuple
from torch import Tensor
from scipy.optimize import linear_sum_assignment
from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return, StateDict
from sd_mecha.extensions import model_configs
from .perm_graph import PermGraph


sgm_split_config = model_configs.resolve("sdxl-sgm_split")
perm_graph = PermGraph(sgm_split_config.metadata())

for key, meta in sgm_split_config.metadata().items():
    if key.startswith("conditioner.embedders.0"):
        if ".q_proj." in key and key.endswith("weight"):  # clip_l qk
            bias_key = key.replace(".weight", ".bias")
            perm_graph.add(key, 0, bias_key, 0)

            k_key = key.replace(".q_proj.", ".k_proj.")
            perm_graph.add(key, 0, k_key, 0)

            k_bias_key = k_key.replace(".weight", ".bias")
            perm_graph.add(k_key, 0, k_bias_key, 0)

        if ".v_proj." in key and key.endswith("weight"):  # clip_l vo
            bias_key = key.replace(".weight", ".bias")
            perm_graph.add(key, 0, bias_key, 0)

            out_key = key.replace(".v_proj.", ".out_proj.")
            perm_graph.add(key, 0, out_key, 1)

        if ".mlp.fc1" in key and key.endswith(".weight"):  # clip_l ff
            bias_key = key.replace(".weight", ".bias")
            perm_graph.add(key, 0, bias_key, 0)

            fc2_key = key.replace(".fc1.", ".fc2.")
            perm_graph.add(key, 0, fc2_key, 1)

    if key.startswith("conditioner.embedders.1"):
        if ".to_q." in key and key.endswith("weight"):  # clip_g qk
            bias_key = key.replace(".weight", ".bias")
            perm_graph.add(key, 0, bias_key, 0)

            k_key = key.replace(".to_q.", ".to_k.")
            perm_graph.add(key, 0, k_key, 0)

            k_bias_key = k_key.replace(".weight", ".bias")
            perm_graph.add(k_key, 0, k_bias_key, 0)

        if ".to_v." in key and key.endswith("weight"):  # clip_g vo
            bias_key = key.replace(".weight", ".bias")
            perm_graph.add(key, 0, bias_key, 0)

            out_key = key.replace(".to_v.", ".out_proj.")
            perm_graph.add(key, 0, out_key, 1)

        if ".mlp.c_fc." in key and key.endswith(".weight"):  # clip_g ff
            bias_key = key.replace(".weight", ".bias")
            perm_graph.add(key, 0, bias_key, 0)

            fc2_key = key.replace(".c_fc.", ".c_proj.")
            perm_graph.add(key, 0, fc2_key, 1)

    if key.startswith("model.diffusion_model"):
        if ".to_q." in key and key.endswith("weight"):  # unet qk
            k_key = key.replace(".to_q.", ".to_k.")
            perm_graph.add(key, 0, k_key, 0)

        if ".to_v." in key and key.endswith("weight"):  # unet vo
            out_key = key.replace(".to_v.", ".to_out.0.")
            perm_graph.add(key, 0, out_key, 1)

        if ".net.0.proj.0." in key and key.endswith(".weight"):  # unet ff
            bias_key = key.replace(".weight", ".bias")
            perm_graph.add(key, 0, bias_key, 0)

            p1_key = key.replace(".0.weight", ".1.weight")
            perm_graph.add(key, 0, p1_key, 0)

            p1_bias_key = p1_key.replace(".weight", ".bias")
            perm_graph.add(p1_key, 0, p1_bias_key, 0)

            fc2_key = key.replace(".net.0.proj.0.", ".net.2.")
            perm_graph.add(key, 0, fc2_key, 1)

        if ".time_embed.0." in key and key.endswith(".weight"):  # unet timestep embed
            bias_key = key.replace(".weight", ".bias")
            perm_graph.add(key, 0, bias_key, 0)

            fc2_key = key.replace(".time_embed.0.", ".time_embed.2.")
            perm_graph.add(key, 0, fc2_key, 1)

        if ".label_emb.0.0." in key and key.endswith(".weight"):  # unet label embed
            bias_key = key.replace(".weight", ".bias")
            perm_graph.add(key, 0, bias_key, 0)

            fc2_key = key.replace(".label_emb.0.0.", ".label_emb.0.2.")
            perm_graph.add(key, 0, fc2_key, 1)


_GLOBAL_INIT_LOCK = threading.Lock()
_RP_INIT_LOCK = threading.Lock()


def _front_flat(t: Tensor, axis: int) -> Tensor:
    t = torch.movedim(t, axis, 0)
    return t.reshape(t.shape[0], -1).contiguous()


def _solve_lap_max(sim_cpu: Tensor) -> Tensor:
    """Return perm p s.t. sum_i sim[i, p[i]] is maximized. sim on CPU."""
    if sim_cpu.device.type != "cpu":
        sim_cpu = sim_cpu.to("cpu")
    if sim_cpu.dtype.itemsize < 2:
        sim_cpu = sim_cpu.float()
    r, c = linear_sum_assignment((-sim_cpu).numpy(force=True))
    return torch.as_tensor(c, device=sim_cpu.device, dtype=torch.long)

def _apply_other_axis_perms(t: Tensor, axis_cur: int, axes_perms: Dict[int, Tensor]) -> Tensor:
    """Apply all axis permutations except for axis_cur to tensor t."""
    out = t
    for ax_idx, p in axes_perms.items():
        if ax_idx == axis_cur:
            continue
        # p is a 1D LongTensor containing the order for axis ax_idx
        out = torch.index_select(out, ax_idx, p)
    return out


@merge_method
def sdxl_sgm_split_rebasin(
    a: Parameter(StateDict[Tensor], model_config=sgm_split_config), # type: ignore
    ref: Parameter(StateDict[Tensor], model_config=sgm_split_config), # type: ignore
    iters: Parameter(int) = 10, # type: ignore
    max_match_size: Parameter(int) = None, # type: ignore
    **kwargs,
) -> Return(Tensor, model_config=sgm_split_config): # type: ignore
    """
    Iterative weight-matching over key-closure groups with one permutation per hyperedge.

    Returns a permuted to be as close as possible to ref.
    """
    key: str = kwargs["key"]
    cache: Dict = kwargs.get("cache")
    if cache is None:
        raise RuntimeError("A cache must be passed to this merge method.")

    # One-time global scaffolding
    if "rebasin" not in cache:
        with _GLOBAL_INIT_LOCK:
            if "rebasin" not in cache:
                key_to_gid, groups_edges, groups_nodes, hedge_nodes = perm_graph.components()
                cache["rebasin"] = {
                    "key_to_gid": key_to_gid,         # key -> group id
                    "groups_edges": groups_edges,     # gid -> set(edge ids)
                    "groups_nodes": groups_nodes,     # gid -> set[(key, axis)]
                    "hedge_nodes": hedge_nodes,       # eid -> list[(key, axis)]
                    "locks_by_group": {},             # gid -> Lock
                    "perm_by_edge": {},               # eid -> LongTensor (CPU)
                    "solved_groups": set(),           # set[gid] that are solved
                }

    rc = cache["rebasin"]
    key_to_gid: Dict[str, int] = rc["key_to_gid"]
    groups_edges: List[Set[int]] = rc["groups_edges"]
    groups_nodes: List[Set[Tuple[str, int]]] = rc["groups_nodes"]
    hedge_nodes: List[List[Tuple[str, int]]] = rc["hedge_nodes"]
    locks_by_group: Dict[int, threading.Lock] = rc["locks_by_group"]
    perm_by_edge: Dict[int, Tensor] = rc["perm_by_edge"]
    solved_groups: Set[int] = rc["solved_groups"]

    gid = key_to_gid.get(key)
    if gid is None:
        # No permutation constraints for this key.
        return a[key]

    if gid not in locks_by_group:
        with _RP_INIT_LOCK:
            if gid not in locks_by_group:
                locks_by_group[gid] = threading.Lock()
    lock = locks_by_group[gid]

    with lock, torch.inference_mode():
        # If already solved, align this key with cached permutations.
        if gid in solved_groups:
            return _align_single_key(a, key, gid, perm_by_edge, hedge_nodes, groups_edges)

        # Local memoization for speed
        memo_a: Dict[str, Tensor] = {}
        memo_ref: Dict[str, Tensor] = {}

        def load_a(k: str) -> Tensor:
            v = memo_a.get(k)
            if v is None:
                v = a[k]
                memo_a[k] = v
            return v

        def load_ref(k: str) -> Tensor:
            v = memo_ref.get(k)
            if v is None:
                v = ref[k]
                memo_ref[k] = v
            return v

        device = load_a(key).device

        # Accumulated axis permutations per key (CPU tensors)
        key_axis_perm: Dict[str, Dict[int, Tensor]] = {}

        # Initialize missing permutations per edge in group
        for eid in groups_edges[gid]:
            nodes = hedge_nodes[eid]
            k0, ax0 = nodes[0]
            N = int(load_a(k0).shape[ax0])
            # sanity: all nodes in edge share same axis size
            for k_i, ax_i in nodes[1:]:
                if int(load_a(k_i).shape[ax_i]) != N:
                    raise ValueError(f"Hyperedge {eid} inconsistent size: {k_i} axis {ax_i} != {N}")
            if eid not in perm_by_edge:
                perm_by_edge[eid] = torch.arange(N, device=torch.device("cpu"), dtype=torch.long)

        # Seed local key_axis_perm from cache
        for eid in groups_edges[gid]:
            p_cpu = perm_by_edge[eid]
            for k_i, ax_i in hedge_nodes[eid]:
                d = key_axis_perm.setdefault(k_i, {})
                if ax_i in d:
                    if d[ax_i].numel() != p_cpu.numel() or not torch.equal(d[ax_i].cpu(), p_cpu):
                        raise ValueError(f"Conflicting permutations for ({k_i}, axis={ax_i}) across edges.")
                else:
                    d[ax_i] = p_cpu.clone()

        # Coordinate descent over hyperedges
        for _ in range(max(1, int(iters))):
            changed = False
            for eid in groups_edges[gid]:
                nodes = hedge_nodes[eid]
                k0, ax0 = nodes[0]
                N = int(load_a(k0).shape[ax0])

                # Optional cap to avoid huge N x N LAP when desired
                if max_match_size is not None and N > int(max_match_size):
                    ident = torch.arange(N, device=torch.device("cpu"), dtype=torch.long)
                    if not torch.equal(perm_by_edge[eid], ident):
                        perm_by_edge[eid] = ident
                        for k_i, ax_i in nodes:
                            key_axis_perm.setdefault(k_i, {})[ax_i] = ident.clone()
                        changed = True
                    continue

                # Build S on device in float32, then run Hungarian on CPU
                S = torch.zeros((N, N), device=device, dtype=torch.float32)
                for k_i, axis_cur in nodes:
                    Av = _front_flat(load_a(k_i), axis_cur).to(dtype=torch.float32)

                    t_ref = load_ref(k_i)
                    # Prepare index permutations (to ref device) for axes != current
                    axes_perms_dev: Dict[int, Tensor] = {}
                    for ax_idx, p_cpu in key_axis_perm.get(k_i, {}).items():
                        if ax_idx == axis_cur:
                            continue
                        axes_perms_dev[ax_idx] = p_cpu.to(device=t_ref.device)

                    B_other = _apply_other_axis_perms(t_ref, axis_cur, axes_perms_dev) if axes_perms_dev else t_ref
                    Bv = _front_flat(B_other, axis_cur).to(dtype=torch.float32)

                    # Accumulate similarity
                    S.addmm_(Av, Bv.T, beta=1.0, alpha=1.0)

                new_perm = _solve_lap_max(S.detach().to("cpu"))
                if not torch.equal(new_perm, perm_by_edge[eid]):
                    changed = True
                    perm_by_edge[eid] = new_perm
                    for k_i, ax_i in nodes:
                        key_axis_perm.setdefault(k_i, {})[ax_i] = new_perm

            if not changed:
                break

        # Mark group solved and align requested key
        solved_groups.add(gid)
        return _align_single_key(a, key, gid, perm_by_edge, hedge_nodes, groups_edges)


def _align_single_key(
    a: StateDict[Tensor],
    key: str,
    gid: int,
    perm_by_edge: Dict[int, Tensor],
    hedge_nodes: List[List[Tuple[str, int]]],
    groups_edges: List[Set[int]],
) -> Tensor:
    """Apply solved permutations to tensor 'a[key]' and return aligned tensor."""
    tA = a[key]
    device = tA.device
    idx_inv = [slice(None)] * tA.dim()

    # Collect inverse permutations on all axes for this key from edges in its group
    for eid in groups_edges[gid]:
        nodes = hedge_nodes[eid]
        p_cpu = perm_by_edge.get(eid)
        if p_cpu is None:
            continue
        for k_i, ax_i in nodes:
            if k_i != key:
                continue
            p_dev = p_cpu.to(device=device)
            inv = torch.empty_like(p_dev)
            inv[p_dev] = torch.arange(len(p_dev), device=device)
            idx_inv[ax_i] = inv

    return tA[tuple(idx_inv)]


@merge_method
def sdxl_sgm_split_randperm(
    a: Parameter(StateDict[Tensor], model_config=sgm_split_config),
    seed: Parameter(int) = None,
    **kwargs,
) -> Return(Tensor, model_config=sgm_split_config):
    """Baseline: apply a shared random permutation per hyperedge group (deterministic with seed)."""
    key: str = kwargs["key"]
    cache: Dict = kwargs.get("cache")
    if cache is None:
        raise RuntimeError("A cache must be passed to this merge method.")

    # One-time topology & locks & maps (safe double-checked)
    if "hyperedge_by_key" not in cache:
        with _RP_INIT_LOCK:
            if "hyperedge_by_key" not in cache:
                hyperedges = perm_graph.hyperedges()
                hyperedge_by_key = {}
                locks = {}
                for hyperedge in hyperedges:
                    lock = threading.Lock()
                    for h_key, _ in hyperedge.items():
                        hyperedge_by_key[h_key] = hyperedge
                        locks[h_key] = lock

                cache.update({
                    "hyperedges": hyperedges,
                    "hyperedge_by_key": hyperedge_by_key,
                    "locks": locks,
                    "permutations": {},
                })

    v = a[key]
    hyperedge_by_key = cache["hyperedge_by_key"]
    if key not in hyperedge_by_key:
        return v

    lock = cache["locks"][key]
    with lock, torch.inference_mode():
        permutations = cache["permutations"]
        hyperedges = cache["hyperedges"]
        dim = hyperedge_by_key[key][key]
        if key not in permutations:
            if seed is not None:
                generator = torch.Generator(device=v.device)
                generator.manual_seed(seed + hyperedges.index(hyperedge_by_key[key]))
            else:
                generator = None
            size = v.shape[dim]
            permutation = torch.randperm(size, generator=generator, device=v.device)
            for h_key, _h_dim in hyperedge_by_key[key].items():
                permutations[h_key] = permutation

        return torch.index_select(v, dim, permutations[key].to(device=v.device))
