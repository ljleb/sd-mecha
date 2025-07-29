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

        # if ".in_layers.2." in key and ".output_blocks." not in key and key.endswith(".weight"):  # unet input res block
        #     bias_key = key.replace(".weight", ".bias")
        #     perm_graph.add(key, 0, bias_key, 0)
        #
        #     emb_key = key.replace(".in_layers.2.", ".emb_layers.1.")
        #     perm_graph.add(key, 0, emb_key, 0)
        #     emb_bias_key = emb_key.replace(".weight", ".bias")
        #     perm_graph.add(emb_key, 0, emb_bias_key, 0)
        #
        #     out_key = key.replace(".in_layers.2.", ".out_layers.0.")
        #     perm_graph.add(key, 0, out_key, 0)
        #     out_bias_key = out_key.replace(".weight", ".bias")
        #     perm_graph.add(out_key, 0, out_bias_key, 0)
        #     out3_key = key.replace(".in_layers.2.", ".out_layers.3.")
        #     perm_graph.add(out_key, 0, out3_key, 1)

        # if ".in_layers.2.0." in key and ".output_blocks." in key and key.endswith(".weight"):  # unet output res block
        #     bias_key = key.replace(".in_layers.2.0.weight", ".in_layers.2.bias")
        #     perm_graph.add(key, 0, bias_key, 0)
        #
        #     p2_key = key.replace(".in_layers.2.0.", ".in_layers.2.1.")
        #     perm_graph.add(key, 0, p2_key, 0)
        #
        #     emb_key = key.replace(".in_layers.2.0.", ".emb_layers.1.")
        #     perm_graph.add(key, 0, emb_key, 0)
        #     emb_bias_key = emb_key.replace(".weight", ".bias")
        #     perm_graph.add(emb_key, 0, emb_bias_key, 0)
        #
        #     out_key = key.replace(".in_layers.2.0.", ".out_layers.0.")
        #     perm_graph.add(key, 0, out_key, 0)
        #     out_bias_key = out_key.replace(".weight", ".bias")
        #     perm_graph.add(out_key, 0, out_bias_key, 0)
        #     out3_key = key.replace(".in_layers.2.0.", ".out_layers.3.")
        #     perm_graph.add(out_key, 0, out3_key, 1)

        if ".time_embed.0." in key and key.endswith(".weight"):  # unet timestep embed
            bias_key = key.replace(".weight", ".bias")
            perm_graph.add(key, 0, bias_key, 0)

            fc2_key = key.replace(".time_embed.0.", ".time_embed.2.")
            perm_graph.add(key, 0, fc2_key, 1)

        if ".label_emb.0.0." in key and key.endswith(".weight"):  # unet timestep embed
            bias_key = key.replace(".weight", ".bias")
            perm_graph.add(key, 0, bias_key, 0)

            fc2_key = key.replace(".label_emb.0.0.", ".label_emb.0.2.")
            perm_graph.add(key, 0, fc2_key, 1)


_GLOBAL_INIT_LOCK = threading.Lock()


def _front_flat(t: Tensor, axis: int) -> Tensor:
    t = torch.movedim(t, axis, 0)
    return t.reshape(t.shape[0], -1).contiguous()


def _solve_lap_max(sim: Tensor) -> Tensor:
    """Return perm p s.t. sum_i sim[i, p[i]] is maximized. sim on CPU."""
    if sim.dtype.itemsize < 2:
        sim = sim.float()
    r, c = linear_sum_assignment((-sim).numpy(force=True))
    return torch.as_tensor(c, device=sim.device, dtype=torch.long)


@merge_method
def sdxl_sgm_split_lerp_rebasin(
    a: Parameter(StateDict[Tensor], model_config=sgm_split_config),
    b: Parameter(StateDict[Tensor], model_config=sgm_split_config),
    alpha: Parameter(Tensor) = 0.5,
    rebasin_iters: Parameter(int) = 4,
    **kwargs,
) -> Return(Tensor, model_config=sgm_split_config):
    """
    Iterative weight-matching over key-closure groups with one permutation per hyperedge.
    """
    key: str = kwargs["key"]
    cache: Dict = kwargs.get("cache") or {}

    # ---------------- one-time scaffolding in kwargs["cache"] ----------------
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
                    "merged_by_group": {},            # gid -> {key: Tensor}
                }

    rc = cache["rebasin"]
    key_to_gid: Dict[str, int] = rc["key_to_gid"]
    groups_edges: List[Set[int]] = rc["groups_edges"]
    groups_nodes: List[Set[Tuple[str, int]]] = rc["groups_nodes"]
    hedge_nodes: List[List[Tuple[str, int]]] = rc["hedge_nodes"]
    locks_by_group: Dict[int, threading.Lock] = rc["locks_by_group"]
    perm_by_edge: Dict[int, Tensor] = rc["perm_by_edge"]
    merged_by_group: Dict[int, Dict[str, Tensor]] = rc["merged_by_group"]

    gid = key_to_gid.get(key)
    if gid is None:
        # No permutation constraints on this key.
        return (1 - alpha) * a[key] + alpha * b[key]

    if gid not in locks_by_group:
        locks_by_group[gid] = threading.Lock()
    lock = locks_by_group[gid]

    with lock:
        gcache = merged_by_group.get(gid)
        if gcache is not None and key in gcache:
            return gcache.pop(key)

        # --------- local memoization of StateDict indexing (critical) --------
        memoA: Dict[str, Tensor] = {}
        memoB: Dict[str, Tensor] = {}

        def LA(k: str) -> Tensor:
            t = memoA.get(k)
            if t is None:
                t = a[k]
                memoA[k] = t
            return t

        def LB(k: str) -> Tensor:
            t = memoB.get(k)
            if t is None:
                t = b[k]
                memoB[k] = t
            return t

        device = LA(key).device
        dtype = LA(key).dtype

        # Current axis-permutation map for each key (accumulated from edge perms)
        key_axis_perm: Dict[str, Dict[int, torch.Tensor]] = {}

        # Initialize missing permutations for edges in this group
        for eid in groups_edges[gid]:
            nodes = hedge_nodes[eid]
            # Determine size N and check consistency within this hyperedge
            k0, ax0 = nodes[0]
            N = LA(k0).shape[ax0]
            for k, ax in nodes[1:]:
                if LA(k).shape[ax] != N:
                    raise ValueError(f"Hyperedge {eid} inconsistent size: {k} axis {ax} != {N}")
            if eid not in perm_by_edge:
                perm_by_edge[eid] = torch.arange(N, device=device, dtype=torch.long)

        # Seed key_axis_perm from existing edge perms (for other axes)
        for eid in groups_edges[gid]:
            perm = perm_by_edge[eid]
            for k, ax in hedge_nodes[eid]:
                d = key_axis_perm.setdefault(k, {})
                # If same (k, ax) appears in multiple edges, enforce equality
                if ax in d:
                    if d[ax].numel() != perm.numel() or not torch.equal(d[ax], perm):
                        raise ValueError(f"Conflicting permutations for ({k}, axis={ax}) across edges.")
                else:
                    d[ax] = perm.clone()

        # ---------------- coordinate-descent over hyperedges -----------------
        for _ in range(max(1, rebasin_iters)):
            changed = False
            for eid in groups_edges[gid]:
                nodes = hedge_nodes[eid]
                k0, ax0 = nodes[0]
                N = LA(k0).shape[ax0]

                # Build similarity with all *other-axis* permutations applied.
                S = torch.zeros((N, N), device=device, dtype=dtype)
                for k, axis_cur in nodes:
                    Av = _front_flat(LA(k), axis_cur)

                    tB = LB(k)
                    # Build index applying perms on all axes except current
                    idx = [slice(None)] * tB.dim()
                    axes_perms = key_axis_perm.get(k, {})
                    for ax_idx, p in axes_perms.items():
                        if ax_idx == axis_cur:
                            continue
                        idx[ax_idx] = p
                    B_other = tB[tuple(idx)]
                    Bv = _front_flat(B_other, axis_cur)

                    S.addmm_(Av, Bv.T)

                new_perm = _solve_lap_max(S)
                if not torch.equal(new_perm, perm_by_edge[eid]):
                    changed = True
                    perm_by_edge[eid] = new_perm
                    # Update key_axis_perm for this edge's axes
                    for k, axis_cur in nodes:
                        key_axis_perm.setdefault(k, {})[axis_cur] = new_perm

            if not changed:
                break

        # ---------------- produce aligned & blended outputs ------------------
        out: Dict[str, Tensor] = {}
        for k, _ax in groups_nodes[gid]:
            tA = LA(k)
            tB = LB(k)
            idx = [slice(None)] * tB.dim()
            for ax_idx, p in key_axis_perm.get(k, {}).items():
                idx[ax_idx] = p
            B_aligned = tB[tuple(idx)]
            out[k] = (1 - alpha) * tA + alpha * B_aligned

        merged_by_group[gid] = out
        return out.pop(key)


_RP_INIT_LOCK = threading.Lock()


@merge_method
def sdxl_sgm_split_randn_permutation(
    a: Parameter(StateDict[Tensor], model_config=sgm_split_config),
    seed: Parameter(int) = None,
    **kwargs,
) -> Return(Tensor, model_config=sgm_split_config):
    key: str = kwargs["key"]
    cache: Dict = kwargs.get("cache")
    if cache is None:
        raise RuntimeError("A cache must be passed to this merge method.")

    # ---------- one-time topology & locks & maps (safe double-checked) ----------
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
    with lock:
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
            for h_key, h_dim in hyperedge_by_key[key].items():
                permutations[h_key] = permutation

        return torch.index_select(v, dim, permutations[key].to(device=v.device))
