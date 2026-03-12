import dataclasses
from collections import defaultdict, deque
from types import MappingProxyType
from typing import DefaultDict, Dict, List, Set, Tuple, Mapping
from sd_mecha.streaming import TensorMetadata


@dataclasses.dataclass(frozen=True, slots=True)
class PermSlice:
    axis: int
    offset: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class Permutation:
    width: int
    slices: Mapping[str, PermSlice | Tuple[PermSlice, ...]]

    def keys(self):
        return self.slices.keys()


@dataclasses.dataclass(frozen=True, slots=True)
class PermNode:
    """One window for one key inside one permutation."""
    key: str
    axis: int
    offset: int


@dataclasses.dataclass(frozen=True, slots=True)
class PermClosure:
    """
    Invocation unit produced by PermGraph.closures().

    Existing fields (kept):
      - perms:         permutation ids participating in this invocation unit
      - components:    interacting components of perms (solver scheduling)
      - required_keys: keys that must be available as inputs

    Added fields (precomputed to avoid per-call work):
      - pid_to_width:  pid -> width
      - pid_to_nodes:  pid -> tuple[PermNode,...] (expanded slices)
      - key_to_apps:   key -> tuple[(pid, axis, offset), ...] for all perms in this closure
                       (used to apply other-axis perms / final inverse perms)
    """
    perms: Tuple[int, ...]
    components: Tuple[Tuple[int, ...], ...]
    keys: Tuple[str, ...]

    pid_to_width: Mapping[int, int] = dataclasses.field(default_factory=dict)
    pid_to_nodes: Mapping[int, Tuple[PermNode, ...]] = dataclasses.field(default_factory=dict)
    key_to_apps: Mapping[str, Tuple[Tuple[int, int, int], ...]] = dataclasses.field(default_factory=dict)

    @staticmethod
    def build_from_graph(
        *,
        perms: Tuple[int, ...],
        components: Tuple[Tuple[int, ...], ...],
        keys: Tuple[str, ...],
        pid_to_perm: Mapping[int, Permutation],
    ) -> "PermClosure":
        # Expand all (key -> (slice or tuple[slice])) into pid->nodes and key->apps.
        pid_to_width: Dict[int, int] = {}
        pid_to_nodes: Dict[int, Tuple[PermNode, ...]] = {}
        key_apps: Dict[str, List[Tuple[int, int, int]]] = {}

        for pid in perms:
            p = pid_to_perm[pid]
            pid_to_width[pid] = int(p.width)

            nodes: List[PermNode] = []
            for k, sls in p.slices.items():
                if isinstance(sls, PermSlice):
                    sls_iter = (sls,)
                else:
                    sls_iter = tuple(sls)

                for sl in sls_iter:
                    nodes.append(PermNode(k, int(sl.axis), int(sl.offset)))
                    key_apps.setdefault(k, []).append((pid, int(sl.axis), int(sl.offset)))

            pid_to_nodes[pid] = tuple(nodes)

        # Deterministic ordering for applications per key:
        # (axis, offset, pid) so sequential application is stable.
        key_to_apps: Dict[str, Tuple[Tuple[int, int, int], ...]] = {}
        for k, apps in key_apps.items():
            apps.sort(key=lambda t: (t[1], t[2], t[0]))
            key_to_apps[k] = tuple(apps)

        return PermClosure(
            perms=perms,
            components=components,
            keys=keys,
            pid_to_width=MappingProxyType(pid_to_width),
            pid_to_nodes=MappingProxyType(pid_to_nodes),
            key_to_apps=MappingProxyType(key_to_apps),
        )


class PermGraph:
    def __init__(self, md: Mapping[str, TensorMetadata]):
        self.md = md
        self._perms: List[Permutation] = []
        self._by_key: DefaultDict[str, List[int]] = defaultdict(list)

    def add(self, p: Permutation) -> int:
        """Register one permutation; returns its integer id (pid)."""
        pid = len(self._perms)
        _validate_perm_disjoint(self.md, p)

        self._perms.append(p)
        for k in p.keys():
            self._by_key[k].append(pid)
        return pid

    def keys_with_perms(self) -> Set[str]:
        return set(self._by_key.keys())

    def closures(self) -> List[PermClosure]:
        """
        Return invocation units.

        Closure criterion: permutations are connected if they share ANY key.
        This is the unit you must put on the LHS of b[...] = ...
        """
        # Build perm adjacency for closure connectivity (share any key)
        perm_adj: DefaultDict[int, Set[int]] = defaultdict(set)
        for k, pids in self._by_key.items():
            # connect all perms touching the same key
            for i in range(len(pids)):
                a = pids[i]
                for j in range(i + 1, len(pids)):
                    b = pids[j]
                    perm_adj[a].add(b)
                    perm_adj[b].add(a)

        visited: Set[int] = set()
        out: List[PermClosure] = []

        for start in range(len(self._perms)):
            if start in visited:
                continue

            # BFS over permutations via "share any key" adjacency
            q = deque([start])
            visited.add(start)
            closure_pids: List[int] = [start]
            closure_keys: Set[str] = set(self._perms[start].keys())

            while q:
                cur = q.popleft()
                for nxt in perm_adj.get(cur, ()):
                    if nxt not in visited:
                        visited.add(nxt)
                        q.append(nxt)
                        closure_pids.append(nxt)
                        closure_keys.update(self._perms[nxt].keys())

            # For this invocation unit, compute solver interaction components
            components = self._interaction_components(tuple(closure_pids))

            # required_keys: union of all keys touched by the closure's perms
            keys = tuple(k for k in self.md.keys() if k in closure_keys)

            out.append(PermClosure.build_from_graph(
                perms=tuple(sorted(closure_pids)),
                components=components,
                keys=keys,
                pid_to_perm={pid: self._perms[pid] for pid in closure_pids},
            ))

        return out

    def _interaction_components(self, pids: Tuple[int, ...]) -> Tuple[Tuple[int, ...], ...]:
        """
        Partition pids into connected components where edges exist iff _perms_interact(md, p, q).
        """
        if not pids:
            return tuple()

        # Build interaction adjacency within this closure
        adj: DefaultDict[int, Set[int]] = defaultdict(set)
        for i in range(len(pids)):
            a = pids[i]
            for j in range(i + 1, len(pids)):
                b = pids[j]
                if _perms_interact(self.md, self._perms[a], self._perms[b]):
                    adj[a].add(b)
                    adj[b].add(a)

        remaining: Set[int] = set(pids)
        comps: List[Tuple[int, ...]] = []

        while remaining:
            start = next(iter(remaining))
            q = deque([start])
            remaining.remove(start)
            comp: List[int] = [start]

            while q:
                cur = q.popleft()
                for nxt in adj.get(cur, ()):
                    if nxt in remaining:
                        remaining.remove(nxt)
                        q.append(nxt)
                        comp.append(nxt)

            comp.sort()
            comps.append(tuple(comp))

        comps.sort(key=lambda c: c[0])
        return tuple(comps)


def _perms_interact(md: Mapping[str, TensorMetadata], p: Permutation, q: Permutation) -> bool:
    shared = set(p.keys()) & set(q.keys())
    for k in shared:
        rank = len(md[k].shape)

        ps = p.slices[k]
        qs = q.slices[k]

        if not isinstance(ps, tuple):
            ps = ps,
        if not isinstance(qs, tuple):
            qs = qs,

        for pn in ps:
            pax = _norm_axis(pn.axis, rank)
            p0, p1 = pn.offset, pn.offset + p.width

            for qn in qs:
                qax = _norm_axis(qn.axis, rank)
                if pax != qax:
                    return True
                q0, q1 = qn.offset, qn.offset + q.width
                if _overlap(p0, p1, q0, q1):
                    return True
    return False


def _overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    # [a0,a1) overlaps [b0,b1)
    return not (a1 <= b0 or b1 <= a0)


def _validate_perm_disjoint(md: Mapping[str, TensorMetadata], p: Permutation) -> None:
    for k, sls in p.slices.items():
        if k not in md:
            raise KeyError(f"unknown key: {k}")
        rank = len(md[k].shape)

        if not isinstance(sls, tuple):
            sls = sls,

        # group by normalized axis
        by_axis: Dict[int, List[int]] = defaultdict(list)
        for sl in sls:
            ax = _norm_axis(sl.axis, rank)
            by_axis[ax].append(sl.offset)

        # validate bounds + disjointness per axis
        for ax, offsets in by_axis.items():
            dim = _axis_size(md, k, ax)
            # bounds
            for off in offsets:
                if off < 0 or off + p.width > dim:
                    raise ValueError(
                        f"{k}: slice [{off}:{off+p.width}] out of bounds for axis {ax} dim {dim}"
                    )
            # disjointness: sort by offset, adjacent must not overlap
            offsets.sort()
            for i in range(len(offsets) - 1):
                a0, a1 = offsets[i], offsets[i] + p.width
                b0, b1 = offsets[i + 1], offsets[i + 1] + p.width
                if _overlap(a0, a1, b0, b1):
                    raise ValueError(
                        f"{k}: overlapping slices on axis {ax}: "
                        f"[{a0}:{a1}] overlaps [{b0}:{b1}] (width={p.width})"
                    )


def _axis_size(md: Mapping[str, TensorMetadata], key: str, axis: int) -> int:
    t = md[key]
    ax = _norm_axis(axis, len(t.shape))
    return int(t.shape[ax])


def _norm_axis(axis: int, rank: int) -> int:
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError(f"axis {axis} out of range for rank {rank}")
    return axis
