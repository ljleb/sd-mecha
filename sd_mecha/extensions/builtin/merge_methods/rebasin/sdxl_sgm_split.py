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
    if sim_cpu.dtype.itemsize < 2:
        sim_cpu = sim_cpu.float()
    r, c = linear_sum_assignment((-sim_cpu).numpy(force=True))
    return torch.as_tensor(c, device=sim_cpu.device, dtype=torch.long)

def _apply_other_axis_perms(t: Tensor, axis_cur: int, axes_perms: Dict[int, Tensor]) -> Tensor:
    """Apply all axis permutations except for axis_cur to tensor t."""
    if not axes_perms:
        return t
    out = t
    for ax_idx, p in axes_perms.items():
        if ax_idx == axis_cur:
            continue
        out = torch.index_select(out, ax_idx, p.to(device=out.device))
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
                    "solved_groups": set(),           # gid -> solved boolean flag
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
        # No permutation constraints on this key.
        return a[key]

    if gid not in locks_by_group:
        with _RP_INIT_LOCK:
            if gid not in locks_by_group:
                locks_by_group[gid] = threading.Lock()
    lock = locks_by_group[gid]
    
    with lock, torch.inference_mode():
        # If group already solved, return stored perms to align the key
        if gid in solved_groups:
            return _align_single_key(a, key, perm_by_edge, hedge_nodes, groups_edges)


        # --------- local memoization of StateDict indexing (critical) --------
        memo_a: Dict[str, Tensor] = {}
        memo_ref: Dict[str, Tensor] = {}

        def load_a(k: str) -> Tensor:
            t = memo_a.get(k)
            if t is None:
                t = a[k]
                memo_a[k] = t
            return t

        def load_ref(k: str) -> Tensor:
            t = memo_ref.get(k)
            if t is None:
                t = ref[k]
                memo_ref[k] = t
            return t

        device = load_a(key).device

        # Current axis-permutation map for each key (accumulated from edge perms)
        key_axis_perm: Dict[str, Dict[int, torch.Tensor]] = {}

        # Initialize missing permutations for edges in this group
        for eid in groups_edges[gid]:
            nodes = hedge_nodes[eid]
            # Determine size N and check consistency within this hyperedge
            k0, ax0 = nodes[0]
            N = int(load_a(k0).shape[ax0])
            for k, ax in nodes[1:]:
                if int(load_a(k).shape[ax]) != N:
                    raise ValueError(f"Hyperedge {eid} inconsistent size: {k} axis {ax} != {N}")
            if eid not in perm_by_edge:
                # Store perms on CPU by default (smaller cache footprint)
                perm_by_edge[eid] = torch.arange(N, device=torch.device("cpu"), dtype=torch.long)

        # Seed key_axis_perm from existing edge perms (for other axes)
        for eid in groups_edges[gid]:
            perm_cpu = perm_by_edge[eid]
            for k, ax in hedge_nodes[eid]:
                d = key_axis_perm.setdefault(k, {})
                if ax in d:
                    # Enforce equality if multiple edges reference same (k, ax)
                    if d[ax].numel() != perm_cpu.numel() or not torch.equal(d[ax].cpu(), perm_cpu):
                        raise ValueError(f"Conflicting permutations for ({k}, axis={ax}) across edges.")
                else:
                    d[ax] = perm_cpu.clone()

        # ---------------- coordinate-descent over hyperedges -----------------
        for _ in range(max(1, int(iters))):
            changed = False
            for eid in groups_edges[gid]:
                nodes = hedge_nodes[eid]
                k0, ax0 = nodes[0]
                N = int(load_a(k0).shape[ax0])

                # Optional safety valve for extremely large matches
                if max_match_size is not None and N > int(max_match_size):
                    # Identity; ensure key_axis_perm reflects identity on this edge's axes.
                    ident_cpu = torch.arange(N, device=torch.device("cpu"), dtype=torch.long)
                    if not torch.equal(perm_by_edge[eid], ident_cpu):
                        perm_by_edge[eid] = ident_cpu
                        for k, axis_cur in nodes:
                            key_axis_perm.setdefault(k, {})[axis_cur] = ident_cpu.clone()
                        changed = True
                    continue
                
                # Build similarity S in float32 on the merge device, then copy to CPU for Hungarian
                S = torch.zeros((N, N), device=device, dtype=torch.float32)
                for k, axis_cur in nodes:
                    Av = _front_flat(load_a(k), axis_cur).to(dtype=torch.float32)

                    t_ref = load_ref(k)
                    # Apply perms on all axes except current
                    axes_perms = key_axis_perm.get(k, {})  # CPU tensors
                    # Bring permutations to device only for indexing when needed
                    axes_perms_dev = {ax: p.to(device=t_ref.device) for ax, p in axes_perms.items() if ax != axis_cur}
                    if axes_perms_dev:
                        B_other = _apply_other_axis_perms(t_ref, axis_cur, axes_perms_dev)
                    else:
                        B_other = t_ref
                    Bv = _front_flat(B_other, axis_cur).to(dtype=torch.float32)

                    # Accumulate Av @ Bv.T
                    # This is on device (GPU if merge_device="cuda")
                    S.addmm_(Av, Bv.T, beta=1.0, alpha=1.0)

                # Solve Hungarian on CPU; keep perms cached on CPU
                new_perm = _solve_lap_max(S.detach().to("cpu"))
                if not torch.equal(new_perm, perm_by_edge[eid]):
                    changed = True
                    perm_by_edge[eid] = new_perm
                    # Update key_axis_perm for this edge's axes (store CPU)
                    for k, axis_cur in nodes:
                        key_axis_perm.setdefault(k, {})[axis_cur] = new_perm

            if not changed:
                break
            
        # Mark group solved
        solved_groups.add(gid)
        
        # Produce aligned tensor for requested key ONLY
        return _align_single_key(a, key, perm_by_edge, hedge_nodes, groups_edges)

def _align_single_key(
    a: StateDict[Tensor],
    key: str,
    perm_by_edge: Dict[int, Tensor],
    hedge_nodes: List[List[Tuple[str, int]]],
    groups_edges: List[Set[int]],
) -> Tensor:
    tA = a[key]
    device = tA.device
    idx_inv = [slice(None)] * tA.dim()
    
    # Aggregate axis perms from all edges that contain this key
    axes_perms_for_key: Dict[int, Tensor] = {}
    for eid in groups_edges[perm_graph.components()[0].get(key, -1)] if False else []:
        # Intentionally disabled
        pass
    
    # Build permutations for this key from the edges directly
    for eid, nodes in enumerate(hedge_nodes):
        for k, ax in nodes:
            if k != key:
                continue
            p_cpu = perm_by_edge(eid)
            if p_cpu is None:
                continue
            # Apply inverse permutation to align 'a' to reference
            p_dev = p_cpu.to(device=device)
            inv = torch.empty_like(p_dev)
            inv[p_dev] = torch.arange(len(p_dev), device=device)
            idx_inv[ax] = inv
    
    aligned = tA[tuple(idx_inv)]
    return aligned


@merge_method
def sdxl_sgm_split_randperm(
    a: Parameter(StateDict[Tensor], model_config=sgm_split_config), # type: ignore
    seed: Parameter(int) = None, # type: ignore
    **kwargs, # type: ignore
) -> Return(Tensor, model_config=sgm_split_config): # type: ignore
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
            for h_key, h_dim in hyperedge_by_key[key].items():
                permutations[h_key] = permutation

        return torch.index_select(v, dim, permutations[key].to(device=v.device))

