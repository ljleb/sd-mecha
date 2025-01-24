import abc
import dataclasses
import itertools
import fuzzywuzzy.process
import inspect
import pathlib
import torch
import typing
from sd_mecha import extensions
from sd_mecha.recipe_nodes import RecipeNode, ModelRecipeNode, MergeRecipeNode, LiteralRecipeNode, RecipeVisitor
from sd_mecha.extensions.merge_space import MergeSpace, MergeSpaceSymbol, AnyMergeSpace
from sd_mecha.extensions.model_config import ModelConfig
from types import SimpleNamespace
from typing import Optional, Callable, Dict, Tuple, List, Iterable, Any, Generic, TypeVar, Mapping


NonDictLiteralValue = str | int | float | bool
RecipeNodeOrValue = RecipeNode | pathlib.Path | NonDictLiteralValue | dict


T = TypeVar('T', torch.Tensor, str, int, float | bool)
class StateDict(Mapping[str, T], Generic[T], abc.ABC):
    @property
    @abc.abstractmethod
    def model_config(self) -> ModelConfig:
        pass

    @abc.abstractmethod
    def keys(self) -> Iterable[str]:
        pass


@dataclasses.dataclass
class ParameterData:
    interface: type
    merge_space: Optional[AnyMergeSpace]
    model_config: Optional[ModelConfig]


def Parameter(interface, merge_space: Optional[str | Iterable[str] | AnyMergeSpace] = None, model_config: Optional[str | ModelConfig] = None) -> type[Any]:
    supported_types = [StateDict] + list(T.__constraints__)
    if type(None) in (typing.get_args(interface) or ()):
        interface = typing.get_args(interface)[0]

    if not isinstance(interface, TypeVar) and not any(issubclass(typing.get_origin(interface) or interface, supported_type) for supported_type in supported_types):
        raise TypeError(f"type {interface} should be one of {', '.join(map(lambda x: x.__name__, supported_types))}")

    if isinstance(merge_space, str):
        merge_space = (merge_space,)
    if isinstance(merge_space, Iterable):
        merge_space = {
            extensions.merge_space.resolve(m) if isinstance(m, str) else m
            for m in merge_space
        }

    if isinstance(model_config, str):
        model_config = extensions.model_config.resolve(model_config)

    return type(Parameter.__name__, (), {
        "data": ParameterData(interface, merge_space, model_config)
    })


def Return(interface, merge_space: Optional[str | MergeSpace | MergeSpaceSymbol] = None, model_config: Optional[str | ModelConfig] = None) -> type[Any]:
    if not isinstance(interface, TypeVar):
        supported_types = list(T.__constraints__)
        if not any(issubclass(typing.get_origin(interface) or interface, supported_type) for supported_type in supported_types):
            raise TypeError(f"type {interface} should be one of {', '.join(map(str, supported_types))}")

    if isinstance(merge_space, (str, MergeSpace)):
        merge_space = (merge_space,)
    if isinstance(merge_space, Iterable):
        merge_space = {
            extensions.merge_space.resolve(m) if isinstance(m, str) else m
            for m in merge_space
        }

    if isinstance(model_config, str):
        model_config = extensions.model_config.resolve(model_config)

    return type(Return.__name__, (), {
        "data": ParameterData(interface, merge_space, model_config)
    })


P = TypeVar('P')
@dataclasses.dataclass
class FunctionArgs(Generic[P]):
    args: List[P]
    vararg: P | SimpleNamespace  # using SimpleNamespace as "empty" because P can be Optional
    kwargs: Dict[str, P]

    def as_dict(self, varargs_count=0) -> Dict[int | str, P]:
        varargs_count = self._get_varargs_count(varargs_count)
        args_dict = {i: v for i, v in enumerate(self.args)}
        vararg_dict = {
            i: self.vararg
            for i in range(len(args_dict), len(args_dict) + varargs_count)
        }
        return args_dict | vararg_dict | self.kwargs

    def args_varags(self, varargs_count=1) -> List[P]:
        varargs_count = self._get_varargs_count(varargs_count)
        varargs = [self.vararg]*varargs_count
        return self.args + varargs

    def _get_varargs_count(self, varargs_count):
        return varargs_count*int(self.has_varargs())

    def has_varargs(self):
        return self.vararg != FunctionArgs.EMPTY_VARARGS


FunctionArgs.EMPTY_VARARGS = SimpleNamespace()


class MergeMethod:
    def __init__(self, f: Callable, identifier: str):
        self.__wrapped__ = f
        self.identifier = identifier
        self.has_varkwargs = True
        self.__validate_f()

    def __validate_f(self):
        spec = inspect.getfullargspec(self.__wrapped__)
        hints = typing.get_type_hints(self.__wrapped__)
        names = self.get_param_names()
        params = self.get_params()
        defaults = self.get_default_args()
        merge_spaces = self.get_input_merge_spaces()
        input_configs = self.get_input_configs()

        if spec.varkw is None:
            self.has_varkwargs = False

        for param_idx in params.as_dict(1):
            is_default_arg = (
                isinstance(param_idx, int) and
                len(params.args) - len(defaults.args) <= param_idx < len(params.args)
            )
            is_default_kwarg = isinstance(param_idx, str) and param_idx in (spec.kwonlydefaults or {})
            if is_default_arg or is_default_kwarg:
                param_merge_space = merge_spaces.as_dict(1)[param_idx]
                if param_merge_space != {extensions.merge_space.resolve("param")}:
                    param_name = names.args_varags()[param_idx] if isinstance(param_idx, int) else param_idx
                    raise TypeError(f"The merge space for '{param_name}' should be 'param' since it has a default value.")

        input_configs_are_explicit = all(config is not None for config in input_configs.as_dict(1).values())
        if input_configs_are_explicit and self.get_return_config(input_configs.args_varags(), input_configs.kwargs) is None:
            raise TypeError("Cannot infer the model config to return from the input model configs")

        return_data = self.__get_return_data(hints.get("return"))
        if isinstance(return_data.merge_space, MergeSpaceSymbol):
            if not any(k.merge_space for k in params.as_dict(1).values()):
                raise RuntimeError("when using a merge space symbol as output, it must also be used by at least one input parameter")

    def merge_key(
        self,
        input_args: Tuple[torch.Tensor | StateDict, ...],
        input_kwargs: Dict[str, torch.Tensor | StateDict],
        key: str,
        cache: Optional[dict],
    ):
        args, kwargs = self.__get_args_kwargs(input_args, input_kwargs, key, cache)
        return self.__wrapped__(*args, **kwargs)

    def __get_args_kwargs(
        self,
        input_args: Tuple[Any, ...],
        input_kwargs: Dict[str, float],
        key: str,
        cache: Optional[dict],
    ) -> Tuple[Tuple[torch.Tensor, ...], Dict]:
        if self.has_varkwargs:
            input_kwargs |= {
                "key": key,
                "cache": cache,
            }
        return input_args, input_kwargs

    def __call__(self, *args, **kwargs):
        return self.create_recipe(*args, **kwargs)

    def create_recipe(self, *args, **kwargs):
        params = self.get_param_names()
        defaults = self.get_default_args()
        max_args = len(params.args) if not params.has_varargs() else float("inf")
        min_args = max_args - len(defaults.args)

        if not (min_args <= len(args) <= max_args):
            raise TypeError(f"Expected from {min_args} to {max_args} arguments, received {len(args)} arguments")

        for k in kwargs:
            if k not in params.kwargs:
                raise TypeError(f"Unexpected keyword-argument '{k}'")

        for k in params.kwargs:
            if k not in itertools.chain(kwargs.keys() | defaults.kwargs.keys()):
                raise TypeError(f"Missing keyword-argument '{k}'")

        varargs_count = len(args) - len(params.args)
        input_configs = self.get_input_configs()
        default_config = self.get_return_config(input_configs.args_varags(varargs_count), input_configs.kwargs)
        merge_spaces = self.get_input_merge_spaces()

        def arg_to_node(k: int | str, arg: Any, expected_type: type):
            nonlocal default_config
            merge_space = merge_spaces.as_dict(varargs_count)[k]
            config = input_configs.as_dict(varargs_count)[k]
            if config is None:
                config = default_config
            return value_to_node(arg, expected_type).accept(InferModelConfigVisitor(config, merge_space))

        input_types = self.get_input_types()
        args = tuple(
            arg_to_node(i, arg, input_types.args[i])
            for i, (arg, k) in enumerate(zip(args, params.args_varags(varargs_count)))
        )
        kwargs = {
            k: arg_to_node(k, arg, input_types.kwargs[k])
            for k, arg in kwargs.items()
        }

        return MergeRecipeNode(self, args, kwargs)

    def get_input_types(self) -> FunctionArgs[type]:
        params = self.get_params()
        arg_types = [
            param.interface
            for param in params.args
        ]
        vararg_type = params.vararg.interface if params.has_varargs() else FunctionArgs.EMPTY_VARARGS
        kwarg_types = {
            k: param.interface
            for k, param in params.kwargs.items()
        }
        return FunctionArgs(arg_types, vararg_type, kwarg_types)

    def get_return_merge_space(self, merge_space_args: List[MergeSpace], merge_space_kwargs: Dict[str, MergeSpace]) -> MergeSpace:
        names = self.get_param_names()
        num_varargs = max(0, len(merge_space_args) - len(names.args))
        input_merge_spaces = self.get_input_merge_spaces().as_dict(num_varargs)

        resolved_input_spaces = {}
        arg_tuples = enumerate(merge_space_args)
        kwarg_tuples = ((k, v) for k, v in merge_space_kwargs.items())
        for idx, merge_space_arg in itertools.chain(arg_tuples, kwarg_tuples):
            name = names.args_varags(num_varargs)[idx] if isinstance(idx, int) else idx
            merge_space_param = input_merge_spaces[idx]
            is_symbol = isinstance(merge_space_param, MergeSpaceSymbol)
            valid_merge_spaces = merge_space_param.merge_spaces if is_symbol else merge_space_param
            if merge_space_arg not in valid_merge_spaces:
                valid_str_merge_spaces = tuple(m.identifier for m in valid_merge_spaces)
                raise TypeError(f"parameter '{name}' of method {self.identifier} expects a merge space in {valid_str_merge_spaces} but got {merge_space_arg.identifier}")
            if not is_symbol:
                continue

            if (resolved_input_space := resolved_input_spaces.get(merge_space_param)) is not None:
                # occurrence of already seen type var
                if merge_space_arg != resolved_input_space:
                    raise TypeError(f"parameter '{name}' of method {self.identifier} was resolved to {resolved_input_space.identifier} but got {merge_space_arg.identifier}")
            else:
                resolved_input_spaces[merge_space_param] = merge_space_arg

        type_hints = typing.get_type_hints(self.__wrapped__)
        merge_space_param = self.__get_return_data(type_hints.get("return")).merge_space
        if isinstance(merge_space_param, MergeSpaceSymbol):
            return resolved_input_spaces[merge_space_param]
        if merge_space_param is None:
            raise RuntimeError(f"could not infer merge space of method '{self.identifier}'")
        return next(iter(merge_space_param))  # get the only value

    def get_input_merge_spaces(self) -> FunctionArgs[AnyMergeSpace]:
        params = self.get_params()
        names = self.get_param_names()
        defaults = self.get_default_args()

        def get_merge_space_or_default(merge_space, has_default):
            if merge_space is None:
                if has_default:
                    merge_space = {extensions.merge_space.resolve("param")}
                else:
                    merge_space = extensions.merge_space.get_all()
            return merge_space

        merge_spaces = []
        for i in range(len(names.args)):
            merge_space = params.args[i].merge_space
            merge_space = get_merge_space_or_default(merge_space, i >= len(names.args) - len(defaults.args))
            merge_spaces.append(merge_space)

        varargs_merge_space = FunctionArgs.EMPTY_VARARGS
        if params.has_varargs():
            varargs_merge_space = params.vararg.merge_space
            if varargs_merge_space is None:
                varargs_merge_space = extensions.merge_space.get_all()

        kw_merge_spaces = {}
        for name in names.kwargs:
            merge_space = params.kwargs[name].merge_space
            merge_space = get_merge_space_or_default(merge_space, name in defaults.kwargs)
            kw_merge_spaces[name] = merge_space

        return FunctionArgs(merge_spaces, varargs_merge_space, kw_merge_spaces)

    def get_return_config(self, arg_configs: List[Optional[ModelConfig]], kwarg_configs: Dict[str, Optional[ModelConfig]]) -> ModelConfig:
        type_hints = typing.get_type_hints(self.__wrapped__)
        names = self.get_param_names()
        num_varargs = max(0, len(arg_configs) - len(names.args))
        input_configs = self.get_input_configs().as_dict(num_varargs)
        default_config = self.__get_return_data(type_hints.get("return")).model_config

        arg_tuples = enumerate(arg_configs)
        kwarg_tuples = ((k, kwarg_configs.get(k)) for k in kwarg_configs)
        for param, arg_config in itertools.chain(arg_tuples, kwarg_tuples):
            if arg_config is None:
                continue

            param_config = input_configs[param]
            if param_config is None:
                param_config = default_config
            if param_config is None:
                param_config = arg_config
                default_config = arg_config
            if param_config.identifier != arg_config.identifier:
                raise ValueError(f"Recipe received an incompatible input: expected model config {param_config.identifier} but instead got {arg_config.identifier}")

        return default_config

    def get_input_configs(self) -> FunctionArgs[Optional[ModelConfig]]:
        params = self.get_params()
        return FunctionArgs(
            [arg.model_config for arg in params.args],
            params.vararg.model_config if params.has_varargs() else FunctionArgs.EMPTY_VARARGS,
            {k: v.model_config for k, v in params.kwargs.items()},
        )

    def get_param_names(self) -> FunctionArgs[str]:
        spec = inspect.getfullargspec(self.__wrapped__)
        return FunctionArgs(
            spec.args or [],
            spec.varargs or FunctionArgs.EMPTY_VARARGS,
            spec.kwonlyargs or {},
        )

    def get_default_args(self) -> FunctionArgs[Any]:
        spec = inspect.getfullargspec(self.__wrapped__)
        return FunctionArgs(
            spec.defaults or [],
            FunctionArgs.EMPTY_VARARGS,
            spec.kwonlydefaults or {},
        )

    def get_params(self) -> FunctionArgs[ParameterData]:
        names = self.get_param_names()
        annotations = typing.get_type_hints(self.__wrapped__)

        return FunctionArgs(
            [self.__get_parameter_data(annotations[k]) for k in names.args],
            self.__get_parameter_data(annotations[names.vararg]) if names.has_varargs() else FunctionArgs.EMPTY_VARARGS,
            {k: self.__get_parameter_data(annotations[k]) for k in names.kwargs},
        )

    @staticmethod
    def __get_parameter_data(hint: type):
        hint_args = [arg for arg in (typing.get_args(hint) or ()) if arg is not type(None)]
        if hint_args:
            hint = hint_args[0]

        if isinstance(getattr(hint, "data", None), ParameterData):
            if hint.__name__ != Parameter.__name__:
                raise RuntimeError("all parameters type should be sd_mecha.Parameter()")
            return hint.data
        return Parameter(hint).data

    @staticmethod
    def __get_return_data(hint: type):
        if isinstance(getattr(hint, "data", None), ParameterData):
            if hint.__name__ != Return.__name__:
                raise RuntimeError("the return type should be sd_mecha.Return()")
            return hint.data
        return Return(hint).data

    def get_identifier(self) -> str:
        return self.identifier


@dataclasses.dataclass
class InferModelConfigVisitor(RecipeVisitor):
    default_model_config: ModelConfig
    default_merge_space: type

    def visit_literal(self, node: LiteralRecipeNode):
        model_config = node.model_config if node.model_config is not None else self.default_model_config
        return LiteralRecipeNode(node.value, model_config)

    def visit_model(self, node: ModelRecipeNode):
        node_merge_space = node.merge_space
        default_merge_spaces = extensions.merge_space.get_identifiers(self.default_merge_space)
        # allow to infer merge space 'param' for i.e. approximated fisher diagonal
        if len(default_merge_spaces) == 1 and default_merge_spaces[0] == "param":
            node_merge_space = default_merge_spaces[0]
        return ModelRecipeNode(
            node.path if node.state_dict is None else node.state_dict,
            node.model_config,
            node_merge_space,
        )

    def visit_merge(self, node: MergeRecipeNode):
        return node


F = TypeVar("F", bound=Callable)


def make_recipe(
    f: Optional[F] = None, *,
    identifier: Optional[str] = None,
    register: bool = True,
    is_conversion: bool = False,
) -> F:
    if f is None:
        return lambda f: __convert_to_recipe_impl(f, identifier=identifier, register=register, is_conversion=is_conversion)
    return __convert_to_recipe_impl(f, identifier=identifier, register=register, is_conversion=is_conversion)


def __convert_to_recipe_impl(
    fn: Callable, *,
    identifier: Optional[str] = None,
    register: bool,
    is_conversion: bool,
):
    if identifier is None:
        identifier = fn.__name__
    merge_method = MergeMethod(fn, identifier)

    if register:
        _merge_methods_registry[identifier] = merge_method
        if is_conversion:
            _conversion_registry[identifier] = validate_config_conversion(merge_method)
    return merge_method


def validate_config_conversion(merge_method: MergeMethod):
    params = merge_method.get_param_names()
    args_varargs = params.args if params.args else params.args_varags(1)
    assert len(args_varargs) == 1, f"the merge method should be able to take exactly 1 positional argument"
    configs = merge_method.get_input_configs()
    input_config = configs.args if configs.args else configs.args_varags(1)[0]
    assert input_config is not None, f"the input ModelConfig['identifier...'] is missing. It should be appended to the type annotation of `{args_varargs[0]}`"
    return merge_method


_merge_methods_registry = {}
_conversion_registry = {}


def resolve(identifier: str) -> MergeMethod:
    try:
        return _merge_methods_registry[identifier]
    except KeyError as e:
        suggestion = fuzzywuzzy.process.extractOne(str(e), _merge_methods_registry.keys())[0]
        raise ValueError(f"unknown merge method: {e}. Nearest match is '{suggestion}'")


def get_all() -> List[MergeMethod]:
    return list(_merge_methods_registry.values())


def get_all_converters() -> List[MergeMethod]:
    return list(_conversion_registry.values())


def value_to_node(node_or_value: RecipeNodeOrValue, expected_type: type = None) -> RecipeNode:
    if isinstance(node_or_value, RecipeNode):
        return node_or_value

    if not isinstance(node_or_value, RecipeNodeOrValue):
        raise TypeError(f"type of 'node_or_value' should be one of {typing.get_args(RecipeNodeOrValue)}, not {type(node_or_value)}")

    numeric = int | float

    # verify dict value type consistency
    if isinstance(node_or_value, Mapping) and node_or_value:
        actual_type = type(next(iter(node_or_value.values())))
        if isinstance(actual_type, numeric):
            actual_type = numeric
        if not issubclass(actual_type, NonDictLiteralValue):
            raise TypeError(f"unsupported type found in input dict: {actual_type} (supported types are {typing.get_args(NonDictLiteralValue)})")
        if not all(isinstance(v, actual_type) for v in node_or_value.values()):
            bad_type = next(iter(type(v) for v in node_or_value.values() if not isinstance(v, actual_type)))
            raise TypeError(f"inconsistent types found in input dict: {actual_type} and {bad_type}")

    try:
        if issubclass(typing.get_origin(expected_type) or expected_type, StateDict):
            expected_type = next(iter(typing.get_args(expected_type) or (T,)))
    except TypeError:
        pass

    if isinstance(node_or_value, NonDictLiteralValue | Mapping):
        return LiteralRecipeNode(node_or_value)

    if isinstance(expected_type, TypeVar) and isinstance(node_or_value, pathlib.Path) or issubclass(expected_type, torch.Tensor):
        try:
            return ModelRecipeNode(node_or_value)
        except TypeError as e:
            base_error_message = f"No implicit conversion exists from {type(node_or_value)} to Dict[Tensor]"
            if isinstance(node_or_value, str):
                raise TypeError(f"{base_error_message}. Consider using pathlib.Path instead if a model path was intended") from e
            if not isinstance(node_or_value, pathlib.Path):
                raise TypeError(base_error_message) from e

    raise TypeError(f"No implicit conversion exists from {type(node_or_value)} to {expected_type}")
