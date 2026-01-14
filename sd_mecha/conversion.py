import functools
import heapq
from typing import Mapping
from .extensions.merge_methods import value_to_node, get_converter_paths
from .extensions.model_configs import ModelConfig
from .recipe_nodes import RecipeNode, RecipeNodeOrValue
from .graph_finalization import open_graph, create_config_candidates


def convert(recipe: RecipeNodeOrValue, config: str | ModelConfig | RecipeNode) -> RecipeNodeOrValue:
    """
    Convert a recipe from one model config to another.

    This searches for a chain of registered conversion functions that transform `recipe`’s underlying
    config into the target config, then composes them. For example, you might need to
    convert a LoRA adapter into the base model’s format.

    Args:
        recipe:
            A `RecipeNode` or dictionary representing the input model or partial recipe.
        config (str, ModelConfig or RecipeNode):
            The desired output config, or a recipe node that has the desired config.

    Returns:
        A new recipe node describing the entire conversion.

    Raises:
        ValueError:
            If no conversion path is found.
    """
    converter_paths = get_converter_paths()

    if config is None:
        return recipe

    if isinstance(config, RecipeNode):
        with open_graph(config, fallback_ms="weight") as config_node:
            config = config_node.model_config

    tgt_config = config if isinstance(config, str) else config.identifier

    if isinstance(recipe, Mapping):
        possible_configs = sorted(create_config_candidates(recipe).stats.items(), key=lambda k: k[1].state_dict_misses)
        for possible_config in (t[0] for t in possible_configs):
            res = create_conversion_recipe(recipe, converter_paths, possible_config, tgt_config)
            if res is not None:
                return res
        raise ValueError(
            "could not infer the intended config to convert from. "
            "explicitly specifying the input config might resolve the issue"
        )

    recipe = value_to_node(recipe)
    with open_graph(recipe, fallback_ms="weight") as recipe_open:
        if recipe_open.model_config is None:
            return recipe
        src_config = recipe_open.model_config.identifier
    if src_config == "structural":
        raise ValueError(
            "recipe config is 'structural': "
            "structural recipes cannot be composed of any config conversions"
        )
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

    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    predecessors = {node: None for node in graph}  # will store the node we came from
    edge_used = {node: None for node in graph}  # will store which edge ID led here
    heap = [(0, start)]

    while heap:
        current_dist, current_node = heapq.heappop(heap)
        if current_dist > distances.get(current_node, float("inf")):
            continue

        if current_node == goal:
            break

        for neighbor, edge_id in graph.get(current_node, ()):
            distance_via_current = current_dist + 1
            if distance_via_current < distances.get(neighbor, float("inf")):
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
