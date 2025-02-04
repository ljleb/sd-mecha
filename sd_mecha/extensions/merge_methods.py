import abc
import dataclasses
import itertools
import fuzzywuzzy.process
import inspect
import pathlib
import torch
import typing
from sd_mecha.recipe_nodes import RecipeNode, ModelRecipeNode, MergeRecipeNode, LiteralRecipeNode, RecipeVisitor, NonDictLiteralValue, RecipeNodeOrValue
from . import merge_spaces, model_configs
from .merge_spaces import MergeSpace, MergeSpaceSymbol, AnyMergeSpace
from .model_configs import ModelConfig
from types import SimpleNamespace
from typing import Optional, Callable, Dict, Tuple, List, Iterable, Any, Generic, TypeVar, Mapping
from ..typing_ import is_subclass


T = TypeVar('T', torch.Tensor, *typing.get_args(NonDictLiteralValue))


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
    interface: type | TypeVar
    merge_space: Optional[AnyMergeSpace]
    model_config: Optional[ModelConfig]


@dataclasses.dataclass
class ParameterType:
    data: ParameterData


def Parameter(interface: type | TypeVar, merge_space: Optional[str | Iterable[str] | AnyMergeSpace] = None, model_config: Optional[str | ModelConfig] = None) -> type[Any]:
    supported_types = [StateDict] + list(T.__constraints__)
    if type(None) in (typing.get_args(interface) or ()):
        interface = typing.get_args(interface)[0]

    if not isinstance(interface, TypeVar) and not any(issubclass(typing.get_origin(interface) or interface, supported_type) for supported_type in supported_types):
        raise TypeError(f"type {interface} should be one of {', '.join(map(lambda x: x.__name__, supported_types))}")

    if isinstance(merge_space, str):
        merge_space = (merge_space,)
    if isinstance(merge_space, Iterable):
        merge_space = {
            merge_spaces.resolve(m) if isinstance(m, str) else m
            for m in merge_space
        }

    if isinstance(model_config, str):
        model_config = model_configs.resolve(model_config)

    return type(Parameter.__name__, (ParameterType,), {
        "data": ParameterData(interface, merge_space, model_config)
    })


@dataclasses.dataclass
class ReturnType:
    data: ParameterData


def Return(interface: type | TypeVar, merge_space: Optional[str | MergeSpace | MergeSpaceSymbol] = None, model_config: Optional[str | ModelConfig] = None) -> type[Any]:
    if not isinstance(interface, TypeVar):
        supported_types = list(T.__constraints__)
        if not any(issubclass(typing.get_origin(interface) or interface, supported_type) for supported_type in supported_types):
            raise TypeError(f"type {interface} should be one of {', '.join(map(str, supported_types))}")

    if isinstance(merge_space, (str, MergeSpace)):
        merge_space = (merge_space,)
    if isinstance(merge_space, Iterable):
        merge_space = {
            merge_spaces.resolve(m) if isinstance(m, str) else m
            for m in merge_space
        }

    if isinstance(model_config, str):
        model_config = model_configs.resolve(model_config)

    return type(Return.__name__, (ReturnType,), {
        "data": ParameterData(interface, merge_space, model_config)
    })


P = TypeVar('P')
@dataclasses.dataclass
class FunctionArgs(Generic[P]):
    args: List[P]
    vararg: P | SimpleNamespace  # using SimpleNamespace as "empty" because P can be Optional
    kwargs: Dict[str, P]

    def as_dict(self, args_count=None) -> Dict[int | str, P]:
        varargs_count = self._get_varargs_count(args_count)
        args_dict = {i: v for i, v in enumerate(self.args)}
        vararg_dict = {
            i: self.vararg
            for i in range(len(args_dict), len(args_dict) + varargs_count)
        }
        return args_dict | vararg_dict | self.kwargs

    def args_varags(self, args_count=None) -> List[P]:
        varargs_count = self._get_varargs_count(args_count)
        varargs = [self.vararg]*varargs_count
        return self.args + varargs

    def _get_varargs_count(self, args_count):
        n_args = len(self.args)
        if args_count is None:
            args_count = n_args + 1

        return (args_count - n_args) * int(self.has_varargs())

    def has_varargs(self):
        return self.vararg != FunctionArgs.EMPTY_VARARGS


FunctionArgs.EMPTY_VARARGS = SimpleNamespace()


class MergeMethod:
    def __init__(self, f: Callable, identifier: str):
        self.__wrapped__ = f
        self.__f_spec = inspect.getfullargspec(self.__wrapped__)
        self.__f_hints = typing.get_type_hints(self.__wrapped__)
        self.identifier = identifier
        self.has_varkwargs = True
        self.default_merge_space = MergeSpaceSymbol(*merge_spaces.get_all())
        self.__validate_f()

    def __validate_f(self):
        names = self.get_param_names()
        params = self.get_params()  # validates param type annotations
        defaults = self.get_default_args()
        input_merge_spaces = self.get_input_merge_spaces()
        input_configs = self.get_input_configs()

        if self.__f_spec.varkw is None:
            self.has_varkwargs = False

        for param_idx in params.as_dict():
            is_default_arg = (
                isinstance(param_idx, int) and
                len(params.args) - len(defaults.args) <= param_idx < len(params.args)
            )
            is_default_kwarg = isinstance(param_idx, str) and param_idx in (self.__f_spec.kwonlydefaults or {})
            param_name = names.as_dict()[param_idx]
            if is_default_arg or is_default_kwarg:
                param_merge_space = input_merge_spaces.as_dict()[param_idx]
                if param_merge_space != {merge_spaces.resolve("param")}:
                    raise TypeError(f"The merge space for '{param_name}' should be 'param' since it has a default value.")

        input_configs_are_explicit = all(config is not None for config in input_configs.as_dict().values())
        if input_configs_are_explicit and self.get_return_config(input_configs.args_varags(), input_configs.kwargs) is None:
            raise TypeError("Cannot infer the model config to return from the input model configs")

        return_data = self.__get_return_data(self.__f_hints.get("return"))  # validates return type annotation
        if isinstance(return_data.merge_space, MergeSpaceSymbol):
            if not any(k.merge_space for k in params.as_dict().values()):
                raise RuntimeError("when using a merge space symbol as output, it must also be used by at least one input parameter")

    def __repr__(self):
        return f"<merge method '{self.identifier}'>"

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

        first_arg_as_kwarg = min(itertools.chain(
            (i for i, k in enumerate(kwargs) if k in params.args),
            (float('inf'),)
        ))
        args = [
            args[i] if i < first_arg_as_kwarg
            else kwargs.pop(params.args[i]) if params.args[i] in kwargs
            else defaults.args[i]
            for i in range(len(params.args))
        ]

        max_args = len(params.args) if not params.has_varargs() else float("inf")
        min_args = len(params.args) - len(defaults.args)
        n_args = len(args)
        if not (min_args <= n_args <= max_args):
            raise TypeError(f"Expected from {min_args} to {max_args} arguments, received {n_args} arguments")

        for k in kwargs:
            if k not in params.kwargs:
                raise TypeError(f"Unexpected keyword-argument '{k}'")

        for k in params.kwargs:
            if k not in itertools.chain(kwargs.keys(), defaults.kwargs.keys()):
                raise TypeError(f"Missing keyword-argument '{k}'")

        default_args = defaults.args[n_args - min_args:]
        input_configs = self.get_input_configs()
        input_configs_dict = input_configs.as_dict(n_args)
        default_config = self.get_return_config(input_configs.args_varags(n_args), input_configs.kwargs)
        input_merge_spaces_dict = self.get_input_merge_spaces().as_dict(n_args)

        def arg_to_node(k: int | str, arg: Any, expected_type: type):
            nonlocal default_config
            merge_space = input_merge_spaces_dict[k]
            config = input_configs_dict[k]
            if config is None:
                config = default_config
            return value_to_node(arg, expected_type).accept(InferModelConfigVisitor(config, merge_space))

        input_types = self.get_input_types()
        args = tuple(
            arg_to_node(i, arg, input_types.args[i] if i < len(input_types.args) else input_types.vararg)
            for i, (arg, k) in enumerate(zip(itertools.chain(args, default_args), params.args_varags(n_args)))
        )
        kwargs = {
            k: arg_to_node(k, arg, input_types.kwargs[k])
            for k, arg in (defaults.kwargs | kwargs).items()
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
        n_args = len(merge_space_args)
        input_merge_spaces = self.get_input_merge_spaces().as_dict(n_args)

        resolved_input_spaces = {}
        arg_tuples = enumerate(merge_space_args)
        kwarg_tuples = ((k, v) for k, v in merge_space_kwargs.items())
        for idx, merge_space_arg in itertools.chain(arg_tuples, kwarg_tuples):
            name = names.args_varags(n_args)[idx] if isinstance(idx, int) else idx
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

        merge_space_param = self.__get_return_data(self.__f_hints.get("return")).merge_space
        if isinstance(merge_space_param, MergeSpaceSymbol):
            return resolved_input_spaces[merge_space_param]
        if merge_space_param is None:
            if self.default_merge_space in resolved_input_spaces:
                return resolved_input_spaces[self.default_merge_space]
            raise RuntimeError(f"could not infer merge space of method '{self.identifier}'")
        return next(iter(merge_space_param))  # get the only value

    def get_input_merge_spaces(self) -> FunctionArgs[AnyMergeSpace]:
        params = self.get_params()
        names = self.get_param_names()
        defaults = self.get_default_args()

        def get_merge_space_or_default(merge_space, has_default):
            if merge_space is None:
                if has_default:
                    merge_space = {merge_spaces.resolve("param")}
                else:
                    merge_space = self.default_merge_space
            return merge_space

        args_merge_spaces = []
        for i in range(len(names.args)):
            merge_space = params.args[i].merge_space
            merge_space = get_merge_space_or_default(merge_space, i >= len(names.args) - len(defaults.args))
            args_merge_spaces.append(merge_space)

        varargs_merge_space = FunctionArgs.EMPTY_VARARGS
        if params.has_varargs():
            varargs_merge_space = params.vararg.merge_space
            if varargs_merge_space is None:
                varargs_merge_space = self.default_merge_space

        kwargs_merge_spaces = {}
        for name in names.kwargs:
            merge_space = params.kwargs[name].merge_space
            merge_space = get_merge_space_or_default(merge_space, name in defaults.kwargs)
            kwargs_merge_spaces[name] = merge_space

        return FunctionArgs(args_merge_spaces, varargs_merge_space, kwargs_merge_spaces)

    def get_return_config(self, arg_configs: List[Optional[ModelConfig]], kwarg_configs: Dict[str, Optional[ModelConfig]]) -> ModelConfig:
        input_configs = self.get_input_configs().as_dict(len(arg_configs))
        default_config = self.__get_return_data(self.__f_hints.get("return")).model_config

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
        return FunctionArgs(
            self.__f_spec.args or [],
            self.__f_spec.varargs or FunctionArgs.EMPTY_VARARGS,
            {k: k for k in self.__f_spec.kwonlyargs} or {},
        )

    def get_default_args(self) -> FunctionArgs[Any]:
        return FunctionArgs(
            self.__f_spec.defaults or [],
            FunctionArgs.EMPTY_VARARGS,
            self.__f_spec.kwonlydefaults or {},
        )

    def get_params(self) -> FunctionArgs[ParameterData]:
        names = self.get_param_names()

        return FunctionArgs(
            [self.__get_parameter_data(self.__f_hints.get(k), k) for k in names.args],
            self.__get_parameter_data(self.__f_hints.get(names.vararg), names.vararg) if names.has_varargs() else FunctionArgs.EMPTY_VARARGS,
            {k: self.__get_parameter_data(self.__f_hints.get(k), k) for k in names.kwargs},
        )

    @staticmethod
    def __get_parameter_data(hint: type, param_name: str):
        hint_args = [arg for arg in (typing.get_args(hint) or ()) if arg is not type(None)]
        if hint_args:
            hint = hint_args[0]

        if hint is Parameter:
            raise TypeError(f"the type of parameter '{param_name}' should be `sd_mecha.Parameter(...)`, not `sd_mecha.Parameter` (note the lack of parentheses)")

        if not inspect.isclass(hint) or not issubclass(hint, ParameterType):
            raise TypeError(f"the type of parameter '{param_name}' should be 'sd_mecha.Parameter(...)', not '{getattr(hint, '__name__', hint)}'")
        return hint.data

    @staticmethod
    def __get_return_data(hint: type):
        if hint is Return:
            raise TypeError(f"the return type should be 'sd_mecha.Return(...)', not 'sd_mecha.Return' (note the lack of parentheses)")

        if not inspect.isclass(hint) or not issubclass(hint, ReturnType):
            raise TypeError(f"the return type should be 'sd_mecha.Return(...)', not '{getattr(hint, '__name__', hint)}'")
        return hint.data

    def get_identifier(self) -> str:
        return self.identifier


@dataclasses.dataclass
class InferModelConfigVisitor(RecipeVisitor):
    default_model_config: ModelConfig
    param_merge_space: type

    def visit_literal(self, node: LiteralRecipeNode):
        if isinstance(node.value, dict) and node.model_config is None:
            from sd_mecha.recipe_merger import infer_model_configs
            possible_configs = infer_model_configs(node.value)
            if self.default_model_config in possible_configs:
                model_config = self.default_model_config
            elif self.default_model_config is None and len(possible_configs) == 1:
                model_config = next(iter(possible_configs))
            else:
                raise ValueError("Cannot implicitly infer the model config of a dict literal (did you forget to use 'sd_mecha.convert'?)")
        else:
            model_config = node.model_config if node.model_config is not None else self.default_model_config
        return LiteralRecipeNode(node.value, model_config=model_config)

    def visit_model(self, node: ModelRecipeNode):
        node_merge_space = node.merge_space
        param_merge_spaces = merge_spaces.get_identifiers(self.param_merge_space)
        # allow to infer merge space 'param' for i.e. approximated fisher diagonal
        if len(param_merge_spaces) == 1 and param_merge_spaces[0] in ["weight", "param"]:
            node_merge_space = param_merge_spaces[0]
        return ModelRecipeNode(
            node.path if node.state_dict is None else node.state_dict,
            model_config=node.model_config,
            merge_space=node_merge_space,
        )

    def visit_merge(self, node: MergeRecipeNode):
        return node


F = TypeVar("F", bound=Callable)


def merge_method(
    f: Optional[F] = None, *,
    identifier: Optional[str] = None,
    register: bool = True,
    is_conversion: bool = False,
) -> F:
    if f is None:
        return lambda f: __recipe_impl(f, identifier=identifier, register=register, is_conversion=is_conversion)
    return __recipe_impl(f, identifier=identifier, register=register, is_conversion=is_conversion)


def __recipe_impl(
    fn: Callable, *,
    identifier: Optional[str] = None,
    register: bool,
    is_conversion: bool,
):
    if identifier is None:
        identifier = fn.__name__
    fn_object = MergeMethod(fn, identifier)

    if register:
        _merge_methods_registry[identifier] = fn_object
        if is_conversion:
            _conversion_registry[identifier] = validate_config_conversion(fn_object)
    elif is_conversion:
        raise ValueError("A conversion recipe must be registered")

    return fn_object


def validate_config_conversion(merge_method: MergeMethod):
    params = merge_method.get_param_names()
    args_varargs = params.args if params.args else params.args_varags()
    assert len(args_varargs) == 1, f"the merge method should be able to take exactly 1 positional argument"
    configs = merge_method.get_input_configs()
    input_config = configs.args if configs.args else configs.args_varags()[0]
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
        if is_subclass(actual_type, numeric):
            actual_type = numeric
        if is_subclass(actual_type, RecipeNode):
            actual_type = RecipeNode
        if not is_subclass(actual_type, NonDictLiteralValue | RecipeNode):
            raise TypeError(f"unsupported type found in input dict: {actual_type} (supported types are {typing.get_args(NonDictLiteralValue)})")
        if not all(isinstance(v, actual_type) for v in node_or_value.values()):
            bad_type = next(iter(type(v) for v in node_or_value.values() if not isinstance(v, actual_type)))
            raise TypeError(f"inconsistent types found in input dict: {actual_type} and {bad_type}")
        if is_subclass(actual_type, RecipeNode):
            first_node = next(iter(node_or_value.values()))
            if not all(r.model_config == first_node.model_config for r in node_or_value.values()):
                raise RuntimeError("inconsistent model configs found in recipe dict")
            if not all(r.merge_space == first_node.merge_space for r in node_or_value.values()):
                raise RuntimeError("inconsistent output merge space found in recipe dict")

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
