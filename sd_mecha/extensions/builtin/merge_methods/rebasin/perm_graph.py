from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Mapping
from sd_mecha.streaming import TensorMetadata


Node = Tuple[str, int]  # (key, axis)


def _norm_axis(axis: int, rank: int) -> int:
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError(f"axis {axis} out of range for rank {rank}")
    return axis


def _axis_size(state: Mapping[str, TensorMetadata], key: str, axis: int) -> int:
    if key not in state:
        raise KeyError(f"unknown tensor key: {key}")
    t = state[key]
    if not hasattr(t, "shape"):
        raise TypeError(f"state[{key!r}] is not a tensor-like object with .shape")
    axis = _norm_axis(axis, len(t.shape))
    return int(t.shape[axis])


class PermGraph:
    """
    Minimal graph of permutation-sharing connections between (key, axis) nodes.

    - add(k1, a1, k2, a2): validates sizes equal, stores an undirected edge.
    - components(): returns list[set[(key, axis)]] (only components with >=2 nodes).
    - to_yaml_groups(): minimal list-of-lists representation [[key, axis], ...] per group.
    - dump_yaml(path) / load_yaml(path): helpers for that minimal format.
    """
    def __init__(self, state_dict: Mapping[str, TensorMetadata]):
        self.state_dict = state_dict
        self.edges: Set[Tuple[Node, Node]] = set()
        self.nodes: Set[Node] = set()

    def add(self, key1: str, axis1: int, key2: str, axis2: int) -> None:
        """Validate sizes match and record an undirected edge."""
        sz1 = _axis_size(self.state_dict, key1, axis1)
        sz2 = _axis_size(self.state_dict, key2, axis2)
        if sz1 != sz2:
            raise ValueError(
                f"mismatched axis sizes: ({key1}, {axis1})={sz1} vs ({key2}, {axis2})={sz2}"
            )
        # normalize axes (in case negatives were supplied)
        a1 = _norm_axis(axis1, len(self.state_dict[key1].shape))
        a2 = _norm_axis(axis2, len(self.state_dict[key2].shape))
        n1, n2 = (key1, a1), (key2, a2)
        # canonicalize undirected edge
        edge = (n1, n2) if n1 <= n2 else (n2, n1)
        self.edges.add(edge)
        self.nodes.add(n1)
        self.nodes.add(n2)

    def hyperedges(self) -> List[Dict[str, int]]:
        """Return hyperedges as a list of sets of (key, axis)."""
        # adjacency
        adj: Dict[Node, Set[Node]] = defaultdict(set)
        for a, b in self.edges:
            adj[a].add(b)
            adj[b].add(a)
        seen: Set[Node] = set()
        comps: List[Dict[str, int]] = []
        for start in self.nodes:
            if start in seen:
                continue
            # BFS
            comp: Dict[str, int] = dict()
            stack = [start]
            seen.add(start)
            while stack:
                u = stack.pop()
                comp[u[0]] = u[1]
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)

            if len(comp) >= 2:
                comps.append(comp)
        # stable ordering (optional)
        comps.sort(key=lambda s: sorted(s)[0])
        return comps

    def components(self):
        hyperedges = self.hyperedges()
        hedge_nodes = [sorted(list(h.items())) for h in hyperedges]
        hedge_keys = [{k for (k, _a) in h} for h in hedge_nodes]
        key_to_edges = {}
        for eid, ks in enumerate(hedge_keys):
            for k in ks:
                key_to_edges.setdefault(k, []).append(eid)

        visited = set()
        groups_edges, groups_nodes = [], []
        for start in range(len(hyperedges)):
            if start in visited:
                continue

            q = [start]
            visited.add(start)
            e_union = {start}
            n_union = set(hedge_nodes[start])
            seen_keys = set()

            while q:
                eid = q.pop()
                for k in hedge_keys[eid]:
                    if k in seen_keys:
                        continue

                    seen_keys.add(k)
                    for nxt in key_to_edges.get(k, []):
                        if nxt not in visited:
                            visited.add(nxt)
                            q.append(nxt)
                            e_union.add(nxt)
                            n_union.update(hedge_nodes[nxt])

            groups_edges.append(e_union)
            groups_nodes.append(n_union)

        key_to_gid = {k: gid for gid, nodes in enumerate(groups_nodes) for k, _a in nodes}
        return key_to_gid, groups_edges, groups_nodes, hedge_nodes

    # -------- minimal YAML I/O (optional) --------
    def to_yaml_groups(self) -> List[List[List[Any]]]:
        """
        Minimal structure: list of groups; each group is list of [key, axis].
        (YAML has no sets/tuples, so we emit lists.)
        """
        try:
            import yaml  # lazy import
        except Exception:
            pass
        groups = []
        for comp in self.components():
            items = [[k, int(a)] for (k, a) in sorted(comp)]
            groups.append(items)
        return groups

    def dump_yaml(self, path: str) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.safe_dump(self.to_yaml_groups(), f, sort_keys=False)

    @staticmethod
    def load_yaml(path: str) -> List[Set[Node]]:
        """Load minimal groups back into Python as list[set[(key, axis)]]."""
        import yaml
        data = yaml.safe_load(open(path))
        groups: List[Set[Node]] = []
        for group in data or []:
            groups.append({(str(k), int(a)) for k, a in group})
        return groups
