import functools
import heapq
import pathlib
from typing import Dict, Tuple, Any, List, Iterable, Mapping
from .extensions.merge_methods import RecipeNodeOrValue, value_to_node
from .extensions.model_configs import ModelConfig
from .extensions import merge_methods


def convert(recipe: RecipeNodeOrValue, config: str | ModelConfig, base_dirs: Iterable[pathlib.Path] = ()):
    all_converters = merge_methods.get_all_converters()
    converter_paths: Dict[str, List[Tuple[str, Any]]] = {}
    for converter in all_converters:
        input_configs = converter.get_input_configs()
        return_config = converter.get_return_config(input_configs.args, input_configs.kwargs)
        src_config = input_configs.args[0].identifier
        tgt_config = return_config.identifier
        converter_paths.setdefault(src_config, [])
        converter_paths.setdefault(tgt_config, [])
        converter_paths[src_config].append((tgt_config, converter))

    tgt_config = config if isinstance(config, str) else config.identifier

    if isinstance(recipe, Mapping):
        from sd_mecha.recipe_merger import infer_model_configs
        possible_configs = infer_model_configs(recipe)
        for possible_config in possible_configs:
            res = create_conversion_recipe(recipe, converter_paths, possible_config.identifier, tgt_config)
            if res is not None:
                return res
        raise ValueError(f"could not infer the intended config to convert from. explicitly specifying the input config might resolve the issue")

    recipe = value_to_node(recipe)
    from sd_mecha.recipe_merger import open_input_dicts
    with open_input_dicts(recipe, base_dirs, buffer_size_per_dict=0):
        src_config = recipe.model_config.identifier
    res = create_conversion_recipe(recipe, converter_paths, src_config, tgt_config)
    if res is None:
        raise ValueError(f"no config conversion exists from {src_config} to {tgt_config}")
    return res


def create_conversion_recipe(recipe, paths, src_config, tgt_config):
    shortest_path = dijkstra(paths, src_config, tgt_config)
    if shortest_path is None:
        return None
    return functools.reduce(lambda v, mm: mm(v), shortest_path, recipe)


def dijkstra(graph, start, goal):
    """
    graph: Dict[str, List[Tuple[str, any_id]]]
        For each node (str), a list of (neighbor_node, edge_id).
    start: str
    goal: str

    Returns: List of edge IDs (in order) that forms the shortest path from start to goal,
             or None if no path exists.
    """

    if start == goal:
        return []

    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    predecessors = {node: None for node in graph}  # will store the node we came from
    edge_used = {node: None for node in graph}  # will store which edge ID led here
    heap = [(0, start)]

    while heap:
        current_dist, current_node = heapq.heappop(heap)
        if current_dist > distances[current_node]:
            continue

        if current_node == goal:
            break

        for neighbor, edge_id in graph[current_node]:
            distance_via_current = current_dist + 1
            if distance_via_current < distances[neighbor]:
                distances[neighbor] = distance_via_current
                predecessors[neighbor] = current_node
                edge_used[neighbor] = edge_id
                heapq.heappush(heap, (distance_via_current, neighbor))

    if distances[goal] == float('inf'):
        return None

    path_ids = []
    node = goal
    while node != start:
        path_ids.append(edge_used[node])
        node = predecessors[node]

    path_ids.reverse()
    return path_ids
