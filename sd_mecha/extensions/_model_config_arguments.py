from dataclasses import dataclass
from inspect import Parameter
from typing import Callable, Iterable

from . import merge_spaces, model_configs
from .model_configs import ModelConfig


@dataclass(frozen=True)
class ParameterInfo:
    """Model config and possible merge spaces of an argument during recipe construction."""

    model_config: ModelConfig
    merge_spaces: frozenset[merge_spaces.MergeSpace]

    @property
    def merge_space(self) -> merge_spaces.MergeSpace | None:
        """The argument's merge space if exactly one space is possible."""
        if len(self.merge_spaces) == 1:
            return next(iter(self.merge_spaces))
        return None


ParameterMapping = Callable[[ParameterInfo], ModelConfig | str | Iterable[ModelConfig | str] | type]


@dataclass(frozen=True)
class ModelConfigRef:
    parameter: str
    mapping: ParameterMapping | None = None

    def resolve(self, config: ModelConfig, spaces: frozenset[merge_spaces.MergeSpace]) -> tuple[ModelConfig, ...]:
        """Resolve the reference to model config objects.

        Args:
            config (ModelConfig): The model config to pass to the mapping.
            spaces: The possible merge spaces of the referenced argument.

        Returns:
            The distinct configs returned by the mapping. If there is no mapping,
            return a tuple containing the supplied config.
        """
        mapped = self.mapping(ParameterInfo(config, spaces)) if self.mapping is not None else config
        if isinstance(mapped, (str, ModelConfig)):
            mapped = (mapped,)
        return tuple(dict.fromkeys(model_configs.resolve(cfg) if isinstance(cfg, str) else cfg for cfg in mapped))


@dataclass(frozen=True)
class ParameterRef:
    parameter: str

    @property
    def model_config(self) -> ModelConfigRef:
        """Reference the argument's model config after any implicit input conversion."""
        return ModelConfigRef(self.parameter)

    def map(self, mapping: ParameterMapping) -> ModelConfigRef:
        """Derive a model config or a string Literal type from another argument's metadata.

        Args:
            mapping: A function that receives a ParameterInfo. When used as model_config,
                return a model config, a registered identifier, or an iterable of either.
                An empty iterable means there are no compatible configs. For Return, the
                result must identify exactly one config. In the Parameter type slot, return
                a string Literal type such as Literal["a", "b"].

        Returns:
            A reference that applies the mapping when the referenced config is known.
        """
        return ModelConfigRef(self.parameter, mapping)


def param_ref(parameter: str) -> ParameterRef:
    """
    Refer to a named parameter in the same merge method.

    Use param_ref("base").model_config in Parameter to constrain another input, or in
    Return to set the output config. Use param_ref("base").map(fn) to read the
    argument's model config and possible merge spaces and derive related configs or a
    string Literal parameter type.

    Args:
        parameter (str):
            The name of a non-variadic parameter in the method signature.

    Returns:
        ParameterRef: A reference the method resolves from the argument's metadata.
    """
    return ParameterRef(parameter)


def validate_config_arguments(signature) -> bool:
    """Validate parameter references and implicit conversion options on a merge method.

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
    type_refs = [
        (name, data.interface) for name, data in parameter_data.items()
        if isinstance(data.interface, ModelConfigRef)
    ]
    for ref in (*refs, *(ref for _, ref in type_refs)):
        parameter = signature.parameters.get(ref.parameter)
        if parameter is None or parameter.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            raise TypeError(f"param_ref({ref.parameter!r}) must name a non-variadic parameter.")

    for name, ref in type_refs:
        if signature.parameters[name].kind == Parameter.VAR_POSITIONAL:
            raise TypeError(f"A mapped parameter type requires a non-variadic parameter, got {name!r}.")
        if ref.parameter == name:
            raise TypeError(f"Parameter {name!r} cannot use a reference to itself as its type.")

    for name, data in parameter_data.items():
        if not (data.allow_conversion or isinstance(data.model_config, ModelConfigRef)):
            continue
        if signature.parameters[name].kind == Parameter.VAR_POSITIONAL:
            raise TypeError(f"Config references and implicit conversion require a non-variadic parameter, got {name!r}.")
        if isinstance(data.model_config, ModelConfigRef) and data.model_config.parameter == name:
            raise TypeError(f"Parameter {name!r} cannot reference its own model config. Use Return to derive an output config.")
        if data.allow_conversion and data.model_config is None:
            raise TypeError(f"Parameter {name!r} must declare a model config to enable implicit conversion.")
    return bool(refs or convertible)


def resolve_config_arguments(signature, bound_args, input_spaces, *, is_interface=False):
    """Resolve the method's config references and any requested input conversion.

    Read the referenced argument's config before checking a dependent argument. If the
    convertible argument already satisfies its config and merge space requirements, use it directly.
    Otherwise, select a unique shortest conversion path and insert it into the recipe.

    Args:
        signature (inspect.Signature): The method signature with its config references.
        bound_args (inspect.BoundArguments): The input nodes, indexed by their parameter names.
        input_spaces: A mapping from each parameter name to its merge space constraint.
        is_interface (bool): Whether a concrete implementation can resolve tied input configs.

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
                    cfg for cfg in candidates.model_config
                    if cfg.identifier != "structural" and stats[cfg.identifier].state_dict_misses == best
                )
                spaces = candidates.merge_space.allowed or merge_spaces.get_all()
                inferred[argument] = configs, spaces
        return inferred[argument]

    configs = {}
    deferred = set()

    def resolve_for_config(ref, config):
        if ref.mapping is None:
            return (config,)
        _, spaces = infer(arguments[ref.parameter])
        return ref.resolve(config, frozenset(spaces).intersection(input_spaces[ref.parameter]))

    def resolve(ref):
        if configs.get(ref.parameter) is not None:
            return resolve_for_config(ref, configs[ref.parameter])
        native, _ = infer(arguments[ref.parameter])
        if len(native) != 1:
            options = tuple(config.identifier for config in native)
            raise TypeError(
                f"Cannot resolve param_ref({ref.parameter!r}) from candidate configs {options}. "
                "Specify the referenced argument's model config explicitly."
            )
        config, = native
        configs[ref.parameter] = config
        return resolve_for_config(ref, config)

    # Bind the opted-in argument first so references to it see the effective config on reload.
    names = list(parameters)
    if convertible is not None:
        names.remove(convertible)
        names.insert(0, convertible)
    for name in names:
        data = parameters[name]
        ref = data.model_config
        if not data.allow_conversion and not isinstance(ref, ModelConfigRef):
            continue

        actual, spaces = infer(arguments[name])
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

        source_configs_by_target = {}
        if isinstance(ref, ModelConfigRef):
            native, _ = infer(arguments[ref.parameter])
            if data.allow_conversion and ref.mapping is None and target_spaces.intersection(spaces):
                # Ordinary additions can still share and infer a config in the graph.
                if not native or not actual or (len(native) > 1 and set(native).intersection(actual)):
                    configs[name] = None
                    deferred.add(ref.parameter)
                    continue
            if data.allow_conversion and is_interface and ref.mapping is not None and len(native) > 1:
                for source in native:
                    for target in resolve_for_config(ref, source):
                        source_configs_by_target.setdefault(target, []).append(source)
                allowed = tuple(source_configs_by_target)
            else:
                allowed = resolve(ref)
        else:
            allowed = (ref,)

        expected_ids = {cfg.identifier for cfg in allowed}
        direct = tuple(cfg for cfg in actual if cfg.identifier in expected_ids)
        if data.allow_conversion and not target_spaces.intersection(spaces):
            direct = ()
        if len(direct) == 1:
            config, = direct
        elif len(direct) > 1:
            raise TypeError(f"Ambiguous model config for parameter {name!r}. Specify its model config explicitly.")
        elif not data.allow_conversion:
            raise TypeError(f"Parameter {name!r} requires one of the model configs {tuple(sorted(expected_ids))}.")
        else:
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
            config = route[-1].get_signature().return_annotation.data.model_config
        configs[name] = config
        if source_configs_by_target:
            choices = tuple(source_configs_by_target[config])
            if len(choices) != 1:
                raise TypeError(f"Ambiguous model config for parameter {ref.parameter!r}. Specify its model config explicitly.")
            configs[ref.parameter], = choices

    return_config = signature.return_annotation.data.model_config
    if isinstance(return_config, ModelConfigRef):
        if return_config.mapping is None and return_config.parameter in deferred:
            return_config = None
        else:
            options = resolve(return_config)
            if len(options) != 1:
                raise TypeError("A return model config reference must resolve to exactly one config.")
            return_config, = options

    return arguments, configs, return_config
