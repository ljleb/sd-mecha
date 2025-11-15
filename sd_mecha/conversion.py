import functools
import heapq
import pathlib
from typing import Dict, Tuple, Any, List, Iterable, Mapping
from .extensions.merge_methods import value_to_node
from .extensions.model_configs import ModelConfig
from .extensions import merge_methods
from .recipe_nodes import RecipeNode, RecipeNodeOrValue
from sd_mecha.merging import open_input_dicts, infer_model_configs


def convert(recipe: RecipeNodeOrValue, config: str | ModelConfig | RecipeNode, model_dirs: Iterable[pathlib.Path] = ()):
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
        model_dirs (Iterable[Path], optional):
            Directories to resolve relative model paths.

    Returns:
        A new recipe node describing the entire conversion.

    Raises:
        ValueError:
            If no conversion path is found.
    """
    model_dirs = list(model_dirs)
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

    if isinstance(config, RecipeNode):
        with open_input_dicts(config, model_dirs):
            config = config.model_config

    tgt_config = config if isinstance(config, str) else config.identifier

    if isinstance(recipe, Mapping):
        possible_configs = infer_model_configs(recipe)
        for possible_config in (cfg for s in possible_configs for cfg in s):
            res = create_conversion_recipe(recipe, converter_paths, possible_config.identifier, tgt_config)
            if res is not None:
                return res
        raise ValueError(
            "could not infer the intended config to convert from. "
            "explicitly specifying the input config might resolve the issue"
        )

    recipe = value_to_node(recipe)
    with open_input_dicts(recipe, model_dirs):
        src_config = recipe.model_config.identifier
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
