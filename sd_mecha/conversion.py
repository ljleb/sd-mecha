import functools
from collections import defaultdict

from .extensions import merge_spaces
from .extensions.merge_methods import get_converter_paths
from .extensions.merge_spaces import MergeSpaceSymbol
from .extensions.model_configs import ModelConfig, resolve as resolve_model_config
from .recipe_nodes import RecipeNode, RecipeNodeOrValue
from .graph_finalization import open_graph


def convert(recipe: RecipeNodeOrValue, config: str | ModelConfig | RecipeNode) -> RecipeNodeOrValue:
    """
    Convert a recipe from one model config to another.

    Search for a chain of registered converters that accept the recipe's config and merge space.
    Each converter can change both. For example, a converter can turn adapter weights into a delta.

    Args:
        recipe (RecipeNodeOrValue):
            The model, state dict, or recipe to convert.
        config (str, ModelConfig, RecipeNode or None):
            The target model config or its registered identifier. If you pass a recipe, use its
            resolved model config as the target. For example, pass the base model to convert an
            adapter to its layout. If you pass None, return the input recipe directly.

    Returns:
        RecipeNodeOrValue: A recipe with the selected converters applied in order.
        If the input already uses the target config, return it directly.

    Raises:
        TypeError: If no registered conversion path connects the input to the target config.
    """
    if config is None:
        return recipe

    if isinstance(config, RecipeNode):
        with open_graph(config, root_only=True, solve_merge_space=False) as config_graph:
            targets = tuple(config_graph.root_candidates().model_config)
    else:
        targets = (resolve_model_config(config) if isinstance(config, str) else config,)

    with open_graph(recipe, root_only=True) as recipe_graph:
        candidates = recipe_graph.root_candidates()
        sources = tuple(candidates.model_config)
        source_stats = candidates.model_config.stats or {}
        spaces = candidates.merge_space.allowed or merge_spaces.get_all()
        for target in targets:
            ranked_sources = sorted(sources, key=lambda source: (
                source_stats[source.identifier].state_dict_misses, source.identifier != target.identifier,
            ))
            for source in ranked_sources:
                routes = _find_conversion_routes((source,), spaces, (target,))
                if routes:
                    return _apply_conversion_route(recipe, routes[0])

    source_ids = tuple(config.identifier for config in sources)
    target_ids = tuple(config.identifier for config in targets)
    raise TypeError(f"Could not convert recipe from candidate configs {source_ids} to {target_ids}.")


def _apply_conversion_route(recipe, route):
    """Build the conversion recipe by applying each merge method to the previous result.

    Args:
        recipe: The original recipe or value.
        route: The registered conversion methods in execution order.

    Returns:
        The last conversion node, or the original input if the route is empty.
    """
    return functools.reduce(lambda node, method: method(node), route, recipe)


def _find_conversion_routes(source_configs, source_spaces, target_configs, target_spaces=None):
    """Search for the routes that use the fewest registered converters.

    A search state holds a model config and a merge space. Each converter's signature
    determines which states it accepts and which state it produces.

    Args:
        source_configs: The possible model configs of the input recipe.
        source_spaces: The possible merge spaces of the input recipe.
        target_configs: The model configs that the converted recipe can use.
        target_spaces: The allowed output merge spaces. If None, any output space is valid.

    Returns:
        A tuple of up to two routes, each an ordered tuple of conversion methods.
        An empty route represents a direct match. An empty result means no route exists.
        Two routes indicate that the shortest route is ambiguous.
    """
    converters = get_converter_paths()
    target_ids = {config.identifier for config in target_configs}
    source_spaces = sorted(source_spaces, key=lambda space: space.identifier)
    frontier = {(config.identifier, space): [()] for config in source_configs for space in source_spaces}
    visited = set(frontier)
    while frontier:
        matches = {}
        for (config, space), routes in frontier.items():
            if config in target_ids and (target_spaces is None or space in target_spaces):
                matches.update(dict.fromkeys(routes))
                if len(matches) >= 2:
                    return tuple(matches)[:2]
        if matches:
            return tuple(matches)

        following = defaultdict(list)
        for (config, space), routes in frontier.items():
            for target, converter in converters.get(config, ()):
                input_space = converter.get_input_merge_spaces().args[0]
                if target is None or space not in input_space:
                    continue
                signature = converter.get_signature()
                if any(p.kind == p.KEYWORD_ONLY and p.default is p.empty for p in signature.parameters.values()):
                    continue
                result_space = signature.return_annotation.data.merge_space
                if isinstance(result_space, MergeSpaceSymbol):
                    if result_space is not input_space:
                        continue
                    result_space = space
                state = (target, result_space)
                if state in visited:
                    continue
                paths = following[state]
                for route in routes:
                    path = (*route, converter)
                    if len(paths) < 2 and path not in paths:
                        paths.append(path)
        visited.update(following)
        frontier = following
    return ()
