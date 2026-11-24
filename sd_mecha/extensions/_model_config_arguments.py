from dataclasses import dataclass, replace
from inspect import Parameter
from typing import Callable, Iterable

from . import merge_spaces, model_configs
from .model_configs import ModelConfig


ConfigMapping = Callable[[ModelConfig], ModelConfig | str | Iterable[ModelConfig | str]]


@dataclass(frozen=True)
class ModelConfigRef:
    parameter: str
    mapping: ConfigMapping | None = None

    def map(self, mapping: ConfigMapping) -> "ModelConfigRef":
        """Derive the allowed model configs from the referenced argument's config.

        Args:
            mapping: A function that receives one resolved ModelConfig. It can return a model
                config, a registered config identifier, or an iterable of either. An empty
                iterable means that the referenced config has no compatible configs.

        Returns:
            ModelConfigRef: A reference to the same parameter with the supplied mapping.
        """
        return replace(self, mapping=mapping)

    def resolve(self, config: ModelConfig) -> tuple[ModelConfig, ...]:
        """Resolve the reference to model config objects.

        Args:
            config (ModelConfig): The resolved config of the referenced argument.

        Returns:
            The distinct configs returned by the mapping. If there is no mapping,
            return a tuple containing the supplied config.
        """
        mapped = self.mapping(config) if self.mapping is not None else config
        if isinstance(mapped, (str, ModelConfig)):
            mapped = (mapped,)
        return tuple(dict.fromkeys(model_configs.resolve(cfg) if isinstance(cfg, str) else cfg for cfg in mapped))


def model_ref(parameter: str) -> ModelConfigRef:
    """
    Refer to the resolved model config of a named argument in the same merge method.

    Use the reference in the model_config argument of Parameter or Return. For example,
    model_ref("base") gives a delta parameter the config of the base. Use map() when
    the parameter accepts related configs, such as the adapter configs for that base.
    If the referenced argument converts, the reference uses its config after conversion.

    Args:
        parameter (str):
            The name of the parameter whose model config supplies the reference.
            This name must refer to a fixed parameter in the method signature.

    Returns:
        ModelConfigRef: A reference that the merge method resolves at recipe construction.
    """
    return ModelConfigRef(parameter)


def validate_config_arguments(signature) -> bool:
    """Validate config references and implicit conversion options on a merge method.

    Args:
        signature (inspect.Signature): The signature with resolved Parameter and Return annotations.

    Returns:
        bool: Whether the method uses a config reference or enables implicit conversion.

    Raises:
        TypeError: If a reference names an invalid parameter or the conversion options conflict.
    """
    parameter_data = {
        name: param.annotation.data for name, param in signature.parameters.items()
        if param.kind != Parameter.VAR_KEYWORD
    }
    convertible = [name for name, data in parameter_data.items() if data.allow_conversion]
    if len(convertible) > 1:
        raise TypeError("Only one parameter may enable implicit conversion.")

    refs = [
        data.model_config
        for data in (*parameter_data.values(), signature.return_annotation.data)
        if isinstance(data.model_config, ModelConfigRef)
    ]
    for ref in refs:
        parameter = signature.parameters.get(ref.parameter)
        if parameter is None or parameter.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            raise TypeError(f"model_ref({ref.parameter!r}) must name a non-variadic parameter.")

    for name, data in parameter_data.items():
        if not (data.allow_conversion or isinstance(data.model_config, ModelConfigRef)):
            continue
        if signature.parameters[name].kind == Parameter.VAR_POSITIONAL:
            raise TypeError(f"Config references and implicit conversion require a non-variadic parameter, got {name!r}.")
        if data.allow_conversion and data.model_config is None:
            raise TypeError(f"Parameter {name!r} must declare a model config to enable implicit conversion.")
        if data.allow_conversion and isinstance(data.model_config, ModelConfigRef):
            ref = data.model_config
            if ref.parameter == name and ref.mapping is not None:
                raise TypeError(f"A convertible parameter cannot map its own model config: {name!r}.")
    return bool(refs or convertible)


def resolve_config_arguments(signature, bound_args, input_spaces):
    """Resolve the method's config references and any requested input conversion.

    Read the referenced argument's config before checking a dependent argument. If the
    convertible argument already satisfies its config and merge space requirements, use it directly.
    Otherwise, select a unique shortest conversion path and insert it into the recipe.

    Args:
        signature (inspect.Signature): The method signature with its config references.
        bound_args (inspect.BoundArguments): The input nodes, indexed by their parameter names.
        input_spaces: A mapping from each parameter name to its merge space constraint.

    Returns:
        The arguments after conversion, the resolved configs by parameter name, and the return config.
        A None return config leaves the method's normal output inference in effect.

    Raises:
        TypeError: If a config reference or conversion route cannot be resolved unambiguously.
    """
    from ..conversion import _apply_conversion_route, _find_conversion_routes
    from ..graph_finalization import open_graph

    arguments = bound_args.arguments.copy()
    inferred = {}
    parameters = {
        name: parameter.annotation.data for name, parameter in signature.parameters.items()
        if parameter.kind != Parameter.VAR_KEYWORD
    }
    convertible = next((name for name, data in parameters.items() if data.allow_conversion), None)

    def infer(argument):
        if argument not in inferred:
            with open_graph(argument, root_only=True) as graph:
                candidates = graph.root_candidates()
                stats = candidates.model_config.stats or {}
                best = min((stat.state_dict_misses for stat in stats.values()), default=None)
                configs = tuple(
                    cfg for cfg in candidates.model_config if stats[cfg.identifier].state_dict_misses == best
                )
                spaces = candidates.merge_space.allowed or merge_spaces.get_all()
                inferred[argument] = configs, spaces
        return inferred[argument]

    configs = {}

    def resolve(ref):
        if ref.parameter == convertible and ref.parameter in configs:
            return ref.resolve(configs[ref.parameter])
        native, _ = infer(arguments[ref.parameter])
        if len(native) != 1:
            options = tuple(config.identifier for config in native)
            raise TypeError(
                f"Cannot resolve model_ref({ref.parameter!r}) from candidate configs {options}. "
                "Specify the referenced argument's model config explicitly."
            )
        config, = native
        configs[ref.parameter] = config
        return ref.resolve(config)

    # Bind the opted-in argument first so references to it see the effective config on reload.
    names = list(parameters)
    if convertible is not None:
        names.remove(convertible)
        names.insert(0, convertible)
    for name in names:
        data = parameters[name]
        if isinstance(data.model_config, ModelConfigRef):
            allowed = resolve(data.model_config)
        elif data.allow_conversion:
            allowed = (data.model_config,)
        else:
            continue

        actual, spaces = infer(arguments[name])
        expected_ids = {cfg.identifier for cfg in allowed}
        direct = tuple(cfg for cfg in actual if cfg.identifier in expected_ids)
        if data.allow_conversion:
            target_space = input_spaces[name]
            target_spaces = set(target_space)
            if isinstance(target_space, merge_spaces.MergeSpaceSymbol):
                for other, constraint in input_spaces.items():
                    if other != name and constraint is target_space:
                        if signature.parameters[other].kind == Parameter.VAR_POSITIONAL:
                            values = arguments.get(other, ())
                        else:
                            values = (arguments[other],)
                        for value in values:
                            _, candidate_spaces = infer(value)
                            target_spaces.intersection_update(candidate_spaces)
            if not target_spaces.intersection(spaces):
                direct = ()
        if len(direct) == 1:
            configs[name], = direct
            continue
        if len(direct) > 1:
            raise TypeError(f"Ambiguous model config for parameter {name!r}. Specify its model config explicitly.")

        if not data.allow_conversion:
            raise TypeError(f"Parameter {name!r} requires one of the model configs {tuple(sorted(expected_ids))}.")

        routes = _find_conversion_routes(actual, spaces, allowed, target_spaces)
        if len(routes) != 1 or not routes[0]:
            problem = "Ambiguous" if routes else "No compatible"
            actual_ids = tuple(config.identifier for config in actual)
            raise TypeError(
                f"{problem} implicit conversion for parameter {name!r} from configs {actual_ids} "
                f"in merge spaces {tuple(sorted(str(space) for space in spaces))} to configs "
                f"{tuple(sorted(expected_ids))} in merge spaces {tuple(sorted(str(space) for space in target_spaces))}. "
                "Use an explicit convert() or specify the input config and merge space."
            )
        route, = routes
        arguments[name] = _apply_conversion_route(arguments[name], route)
        configs[name] = route[-1].get_signature().return_annotation.data.model_config

    return_config = signature.return_annotation.data.model_config
    if isinstance(return_config, ModelConfigRef):
        options = resolve(return_config)
        if len(options) != 1:
            raise TypeError("A return model config reference must resolve to exactly one config.")
        return_config, = options

    return arguments, configs, return_config
