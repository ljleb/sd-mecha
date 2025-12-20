import inspect
import fuzzywuzzy.process
from typing import Callable, Dict, List, Optional
from sd_mecha.extensions.merge_methods import _ensure_parameter, _ensure_return, MergeMethod


class MergeStrategy:
    def __init__(self, identifier: str, fn: Callable):
        self.identifier = identifier
        self.candidates = []

        signature = inspect.signature(fn)
        for param in signature.parameters.values():
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                raise RuntimeError(f"Keyword-only parameter '{param.name}' is not allowed in a merge strategy.")
        self.signature = signature.replace(
            parameters=[
                p.replace(annotation=_ensure_parameter(p.annotation, p.name))
                for p in signature.parameters.values()
            ],
            return_annotation=_ensure_return(signature.return_annotation),
        )

    def register_candidate(self, candidate: MergeMethod):
        candidate_signature = candidate.get_signature()
        new_parameters = []

        for (contract_name, contract_param), (candidate_name, candidate_param) in zip(
            self.signature.parameters.items(), candidate_signature.parameters.items(),
        ):
            contract_param: inspect.Parameter
            candidate_param: inspect.Parameter
            if candidate_param.kind != contract_param.kind:
                raise RuntimeError(f"Expected parameter '{candidate_name}' to be {contract_param.kind} but is {candidate_param.kind}.")
            if candidate_name != contract_name:
                raise RuntimeError(f"Expected parameter '{candidate_name}' to be named '{contract_name}'.")

            candidate_data = candidate_param.annotation.data
            contract_data = contract_param.annotation.data
            if candidate_data.interface != contract_data.interface:
                raise TypeError(f"Expected parameter '{candidate_name}' to have type {contract_data.interface} but got {candidate_data.interface}.")
            if contract_data.merge_space is not None and candidate_data.merge_space != contract_data.merge_space:
                raise TypeError(f"Expected parameter '{candidate_name}' to use merge space(s) {contract_data.merge_space} but got {candidate_data.merge_space}.")
            if contract_data.model_config is not None and candidate_data.model_config != contract_data.model_config:
                raise TypeError(f"Expected parameter '{candidate_name}' to use model config {contract_data.model_config} but got {candidate_data.model_config}.")
            if contract_param.default == inspect.Parameter.empty and candidate_param.default != inspect.Parameter.empty:
                raise TypeError(f"Expected parameter '{candidate_name}' to have no default value.")

            new_parameters.append(candidate_param.replace(
                default=candidate_param.default if candidate_param.default != inspect.Parameter.empty else contract_param.default,
            ))

        candidate_data = candidate_signature.return_annotation.data
        contract_data = self.signature.return_annotation.data
        if candidate_data.interface != contract_data.interface:
            raise TypeError(f"Expected return type {contract_data.interface} but got {candidate_data.interface}.")
        if contract_data.merge_space is not None and candidate_data.merge_space != contract_data.merge_space:
            raise TypeError(f"Expected return merge space {contract_data.merge_space} but got {candidate_data.merge_space}.")
        if contract_data.model_config is not None and candidate_data.model_config != contract_data.model_config:
            raise TypeError(f"Expected return model config {contract_data.model_config} but got {candidate_data.model_config}.")

        candidate_signature = candidate_signature.replace(parameters=new_parameters)
        self.candidates.append((candidate, candidate_signature))


def apply_strategy(strategy: str | MergeStrategy, *args, **kwargs):
    if isinstance(strategy, str):
        strategy = resolve(strategy)

    args_model_configs = [getattr(v, "model_config", None) for v in args]
    kwargs_model_configs = {k: getattr(v, "model_config", None) for k, v in kwargs.items()}

    args_merge_spaces = [getattr(v, "merge_space", None) for v in args]
    kwargs_merge_spaces = {k: getattr(v, "merge_space", None) for k, v in kwargs.items()}

    for candidate, candidate_signature in strategy.candidates:
        try:
            model_configs = candidate_signature.bind(*args_model_configs, **kwargs_model_configs).arguments
            merge_spaces = candidate_signature.bind(*args_merge_spaces, **kwargs_merge_spaces).arguments
            for parameter_name, argument_config in model_configs.items():
                argument_merge_space = merge_spaces.get(parameter_name, None)
                contract_data = candidate_signature.parameters[parameter_name].annotation.data
                if contract_data.model_config is not None and argument_config is not None and contract_data.model_config != argument_config:
                    raise TypeError
                if contract_data.merge_space is not None and argument_merge_space is not None and argument_merge_space not in contract_data.merge_space:
                    raise TypeError

            bound_args = candidate_signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            return candidate.create_recipe(bound_args)

        except TypeError:
            pass

    raise TypeError(f"No strategy candidates matched the given arguments: {strategy.identifier}(*{args}, **{kwargs})")


def merge_strategy(
    fn: Optional[Callable] = None, *,
    identifier: Optional[str] = None,
) -> MergeStrategy | Callable[[Callable], MergeStrategy]:
    if fn is None:
        return lambda fn: __merge_strategy_impl(fn, identifier=identifier)
    return __merge_strategy_impl(fn, identifier=identifier)


def __merge_strategy_impl(
    fn: Callable,
    identifier: Optional[str],
) -> MergeStrategy:
    if identifier is None:
        identifier = fn.__name__
    fn_object = MergeStrategy(identifier, fn)

    if identifier in _strategies_registry:
        raise ValueError(f"Another merge strategy named {identifier} is already registered.")

    if not inspect.isfunction(fn):
        raise ValueError("merge_strategy() must be applied to a function")

    _strategies_registry[identifier] = fn_object
    return fn_object


_strategies_registry: Dict[str, MergeStrategy] = {}


def resolve(identifier: str) -> MergeStrategy:
    try:
        return _strategies_registry[identifier]
    except KeyError as e:
        suggestion = fuzzywuzzy.process.extractOne(str(e), _strategies_registry.keys())[0]
        raise ValueError(f"unknown merge method interface: {e}. Nearest match is '{suggestion}'")


def get_all() -> List[MergeStrategy]:
    return list(_strategies_registry.values())
