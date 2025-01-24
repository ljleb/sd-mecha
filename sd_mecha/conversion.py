import functools
import heapq
import pathlib
from typing import Dict, Tuple, Any, List, Iterable
from sd_mecha.extensions.merge_method import RecipeNodeOrValue, value_to_node
from sd_mecha.extensions.model_config import ModelConfig
from sd_mecha.extensions import merge_method


def convert(recipe: RecipeNodeOrValue, config: str | ModelConfig = None, base_dirs: Iterable[pathlib.Path] = ()):
    recipe = value_to_node(recipe)
    all_converters = merge_method.get_all_converters()
    converter_paths: Dict[str, List[Tuple[str, Any]]] = {}
    for converter in all_converters:
        input_configs = converter.get_input_configs()
        return_config = converter.get_return_config(input_configs.args, input_configs.kwargs)
        src_config = input_configs.args[0].identifier
        tgt_config = return_config.identifier
        converter_paths.setdefault(src_config, [])
        converter_paths.setdefault(tgt_config, [])
        converter_paths[src_config].append((tgt_config, converter))

    from sd_mecha.recipe_merger import open_input_dicts
    with open_input_dicts(recipe, base_dirs, buffer_size_per_dict=0):
        src_config = recipe.model_config.identifier
    tgt_config = config if isinstance(config, str) else config.identifier
    shortest_path = dijkstra(converter_paths, src_config, tgt_config)
    if shortest_path is None:
        raise ValueError(f"no config conversion exists from {src_config} to {tgt_config}")
    return functools.reduce(lambda v, mm: mm.create_recipe(v), shortest_path, recipe)


def dijkstra(graph, start, goal):
    """
    graph: Dict[str, List[Tuple[str, any_id]]]
        For each node (str), a list of (neighbor_node, edge_id).
    start: str
    goal: str

    Returns: List of edge IDs (in order) that forms the shortest path from start to goal,
             or None if no path exists.
    """

    # 1) Initialize distances to infinity, except start=0
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    # 2) Keep track of how we reached each node (the node before it, and the edge ID used)
    predecessors = {node: None for node in graph}  # will store the node we came from
    edge_used = {node: None for node in graph}  # will store which edge ID led here

    # 3) Priority queue (min-heap) for the frontier
    #    The heap items are (distance_so_far, current_node)
    heap = [(0, start)]

    while heap:
        current_dist, current_node = heapq.heappop(heap)

        # If we have already found a better path, skip
        if current_dist > distances[current_node]:
            continue

        # If we've reached the goal, we can stop early
        if current_node == goal:
            break

        # Explore each neighbor of current_node
        for neighbor, edge_id in graph[current_node]:
            # Decide how you want to interpret "cost" of an edge
            # Here, we just count each edge as '1'.
            cost = 1

            distance_via_current = current_dist + cost
            if distance_via_current < distances[neighbor]:
                # Found a better path to 'neighbor'
                distances[neighbor] = distance_via_current
                predecessors[neighbor] = current_node
                edge_used[neighbor] = edge_id

                # Push updated distance & node into the heap
                heapq.heappush(heap, (distance_via_current, neighbor))

    # If goal is unreachable, distances[goal] will be infinity
    if distances[goal] == float('inf'):
        return None

    # Reconstruct path of edge IDs from goal back to start
    path_ids = []
    node = goal

    while node != start:
        # edge_used[node] is the ID of the edge that got us from predecessors[node] to node
        path_ids.append(edge_used[node])
        node = predecessors[node]

    # The path is in reverse order (goal -> ... -> start), so reverse it
    path_ids.reverse()

    return path_ids
