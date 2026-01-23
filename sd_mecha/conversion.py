import functools
import heapq
from .extensions.merge_methods import get_converter_paths
from .extensions.model_configs import ModelConfig
from .recipe_nodes import RecipeNode, RecipeNodeOrValue
from .graph_finalization import open_graph


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
        with open_graph(config, root_only=True, solve_merge_space=False) as config_graph:
            config = config_graph.finalize(check_mandatory_keys=False, check_extra_keys=False).root.model_config

    tgt_config = config if isinstance(config, str) else config.identifier

    with open_graph(recipe, root_only=True, solve_merge_space=False) as recipe_graph:
        src_candidates = recipe_graph.root_candidates(model_config_preference=(tgt_config,)).model_config
        for src_config in src_candidates:
            res = create_conversion_recipe(recipe, converter_paths, src_config.identifier, tgt_config)
            if res is not None:
                return res
        raise ValueError("Could not infer the intended config to convert from.")


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
