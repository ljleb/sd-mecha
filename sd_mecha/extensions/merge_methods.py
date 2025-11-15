import abc
import dataclasses
import itertools
import fuzzywuzzy.process
import inspect
import pathlib
import torch
import typing
from sd_mecha.recipe_nodes import RecipeNode, ModelRecipeNode, MergeRecipeNode, LiteralRecipeNode, NonDictLiteralValue, RecipeNodeOrValue, PythonLiteralValue
from . import merge_spaces, model_configs
from .merge_spaces import MergeSpace, MergeSpaceSymbol, AnyMergeSpace
from .model_configs import ModelConfig
from types import SimpleNamespace
from typing import Optional, Callable, Dict, Tuple, List, Iterable, Any, Generic, TypeVar, Mapping
from ..typing_ import is_subclass


T = TypeVar('T', *typing.get_args(NonDictLiteralValue))


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
    merge_space: Optional[MergeSpace | AnyMergeSpace]
    model_config: Optional[ModelConfig]


@dataclasses.dataclass
class ParameterType:
    data: ParameterData


def Parameter(
    interface: type[NonDictLiteralValue | StateDict[NonDictLiteralValue]] | TypeVar,
    merge_space: Optional[MergeSpace | str | Iterable[MergeSpace | str] | MergeSpaceSymbol] = None,
    model_config: Optional[ModelConfig | str] = None,
) -> type[Any]:
    """
    Describe a parameter to a merge method with its type, optional merge space, and optional model config.

    `Parameter` is used in function type hints to specify how `@merge_method` should handle each
    argument. For example, `Parameter(Tensor, "weight")` means "this argument is a tensor
    in weight space."

    Args:
        interface (type):
            The Python or Torch type of this parameter.
        merge_space (str or Iterable[str], optional):
            Which merge space(s) are valid for this parameter (e.g., "weight", "delta").
        model_config (str or ModelConfig, optional):
            A specific model config or config identifier if it must match a certain architecture.

    Returns:
        A special type annotation object used by `@merge_method` to interpret function arguments.
    """
    supported_types = [StateDict] + list(T.__constraints__)
    if type(None) in (typing.get_args(interface) or ()):
        interface = typing.get_args(interface)[0]

    if not isinstance(interface, TypeVar) and not any(issubclass(typing.get_origin(interface) or interface, supported_type) for supported_type in supported_types):
        raise TypeError(f"type {interface} should be one of {', '.join(map(lambda x: x.__name__, supported_types))}")

    if isinstance(merge_space, (str, MergeSpace)):
        merge_space = (merge_space,)
    if isinstance(merge_space, Iterable):
        merge_space = {
            merge_spaces.resolve(m) if isinstance(m, str) else m
            for m in merge_space
        }

    if isinstance(model_config, str):
        model_config = model_configs.resolve(model_config)

    if getattr(model_config, "identifier", None) == "structural":
        raise ValueError("merge methods cannot convert 'structural' model configs")

    return type(Parameter.__name__, (ParameterType,), {
        "data": ParameterData(interface, merge_space, model_config)
    })


@dataclasses.dataclass
class ReturnType:
    data: ParameterData


def Return(
    interface: type[NonDictLiteralValue] | TypeVar,
    merge_space: Optional[MergeSpace | str | MergeSpaceSymbol] = None,
    model_config: Optional[ModelConfig | str] = None,
) -> type[Any]:
    """
    Describe the return type of a merge method, optionally including its merge space and model config.

    Args:
        interface (type):
            The Python or Torch type (e.g., `torch.Tensor`) returned by the merge method.
        merge_space (MergeSpace, str or MergeSpaceSymbol, optional):
            The single merge space valid for the return, or a symbol that depends on the input spaces.
        model_config (ModelConfig or str, optional):
            The model config that the returned tensor or dictionary should belong to.

    Returns:
        A type annotation object used by `@merge_method` for the return signature.
    """
    if not isinstance(interface, TypeVar):
        supported_types = list(T.__constraints__)
        if not any(issubclass(typing.get_origin(interface) or interface, supported_type) for supported_type in supported_types):
            raise TypeError(f"type {interface} should be one of {', '.join(map(str, supported_types))}")

    if isinstance(merge_space, str):
        merge_space = merge_spaces.resolve(merge_space)

    if isinstance(model_config, str):
        model_config = model_configs.resolve(model_config)

    if getattr(model_config, "identifier", None) == "structural":
        raise ValueError("merge methods cannot convert 'structural' model configs")

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

    def args_varargs(self, args_count=None) -> List[P]:
        varargs_count = self._get_varargs_count(args_count)
        varargs = [self.vararg]*varargs_count
        return self.args + varargs

    def _get_varargs_count(self, args_count):
        n_args = len(self.args)
        if args_count is None:
            args_count = n_args + 1

        return max(0, args_count - n_args) * int(self.has_varargs())

    def has_varargs(self):
        return self.vararg != FunctionArgs.EMPTY_VARARGS


FunctionArgs.EMPTY_VARARGS = SimpleNamespace()


class MergeMethod:
    def __init__(self, fn: Callable, identifier: str):
        self.__wrapped__ = fn
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
        if input_configs_are_explicit and self.get_return_config(input_configs.args_varargs(), input_configs.kwargs) is None:
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
        first_default_arg = len(params.args) - len(defaults.args)

        first_arg_as_kwarg = min((
            *(params.args.index(k) for k in kwargs if k in params.args),
            float('inf'),
        ))

        def ensure_positive(v: int):
            if v < 0:
                raise RuntimeError
            return v

        max_args = len(params.args) if not params.has_varargs() else float("inf")
        min_args = len(params.args) - len(defaults.args)
        n_args = len(args) + len([kwargs[k] for k in params.args if k in kwargs])
        if not (min_args <= n_args <= max_args):
            raise TypeError(f"Expected from {min_args} to {max_args} arguments, received {n_args} arguments")

        args = [
            args[i] if i < min(first_arg_as_kwarg, len(args))
            else kwargs.pop(params.args[i]) if params.args[i] in kwargs
            else defaults.args[ensure_positive(i - first_default_arg)]
            for i in range(len(params.args))
        ] + list(args[len(params.args):])

        for k in kwargs:
            if k not in params.kwargs:
                raise TypeError(f"Unexpected keyword-argument '{k}'")

        for k in params.kwargs:
            if k not in (*kwargs.keys(), *defaults.kwargs.keys()):
                raise TypeError(f"Missing keyword-argument '{k}'")

        input_types = self.get_input_types()
        args = tuple(
            value_to_node(arg, arg_type)
            for arg, k, arg_type in zip(args, params.args_varargs(n_args), input_types.args_varargs(n_args))
        )
        kwargs = {
            k: value_to_node(arg, input_types.kwargs[k])
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

        for idx, merge_space_arg in (*zip(names.args_varargs(n_args), merge_space_args), *merge_space_kwargs.items()):
            if merge_space_arg is None:
                raise ValueError(f"merge space of parameter {idx} cannot be None")

        input_merge_spaces = self.get_input_merge_spaces().as_dict(n_args)

        resolved_input_spaces = {}
        arg_tuples = enumerate(merge_space_args)
        kwarg_tuples = ((k, v) for k, v in merge_space_kwargs.items())
        for idx, merge_space_arg in itertools.chain(arg_tuples, kwarg_tuples):
            name = names.args_varargs(n_args)[idx] if isinstance(idx, int) else idx
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

        merge_space_param: MergeSpace | MergeSpaceSymbol = self.__get_return_data(self.__f_hints.get("return")).merge_space
        if isinstance(merge_space_param, MergeSpaceSymbol):
            return resolved_input_spaces[merge_space_param]
        if merge_space_param is None:
            if self.default_merge_space in resolved_input_spaces:
                return resolved_input_spaces[self.default_merge_space]
            any_input_merge_space = next(iter(input_merge_spaces.values()))
            if all(v == any_input_merge_space for v in input_merge_spaces.values()):
                if isinstance(any_input_merge_space, set):
                    if len(any_input_merge_space) != 1:
                        raise RuntimeError(f"could not infer merge space of method '{self.identifier}'")
                    any_input_merge_space = next(iter(any_input_merge_space))
                return any_input_merge_space
            raise RuntimeError(f"could not infer merge space of method '{self.identifier}'")
        return merge_space_param

    def get_input_merge_spaces(self) -> FunctionArgs[AnyMergeSpace]:
        params = self.get_params()
        names = self.get_param_names()
        defaults = self.get_default_args()

        def get_merge_space_or_default(param, has_default):
            merge_space = param.merge_space
            if merge_space is None:
                if has_default or is_subclass(param.interface, PythonLiteralValue):
                    merge_space = {merge_spaces.resolve("param")}
                else:
                    merge_space = self.default_merge_space
            return merge_space

        args_merge_spaces = []
        for i in range(len(names.args)):
            param = params.args[i]
            merge_space = get_merge_space_or_default(param, i >= len(names.args) - len(defaults.args))
            args_merge_spaces.append(merge_space)

        varargs_merge_space = FunctionArgs.EMPTY_VARARGS
        if params.has_varargs():
            varargs_merge_space = params.vararg.merge_space
            if varargs_merge_space is None:
                varargs_merge_space = self.default_merge_space

        kwargs_merge_spaces = {}
        for name in names.kwargs:
            param = params.kwargs[name]
            merge_space = get_merge_space_or_default(param, name in defaults.kwargs)
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


F = TypeVar("F", bound=Callable)


def merge_method(
    fn: Optional[F] = None, *,
    identifier: Optional[str] = None,
    register: bool = True,
    is_conversion: bool = False,
) -> MergeRecipeNode | Callable[[F], MergeRecipeNode]:
    """
    Decorator to define a custom merge method.

    This converts the decorated function into a `MergeMethod` object that can be used in
    recipe graphs. The type hints in the function signature determine which arguments are
    considered "weight" or "param" merges, and so on.

    Args:
        fn (callable, optional):
            The function to decorate. If omitted, the decorator can be used with named
            arguments (e.g. `@merge_method(is_conversion=True)`).
        identifier (str, optional):
            An explicit name to register this method under. By default, uses the function’s name.
        register (bool):
            If True (default), registers the merge method globally so it can be accessed by `merge_methods.resolve()`.
        is_conversion (bool):
            If True, marks this merge method as a config-conversion function. That means `convert()`
            will consider it as an implicit transition when converting between different model configs.

    Returns:
        A `MergeMethod` object or a decorator.
    """
    if fn is None:
        return lambda fn: __recipe_impl(fn, identifier=identifier, register=register, is_conversion=is_conversion)
    return __recipe_impl(fn, identifier=identifier, register=register, is_conversion=is_conversion)


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
    args_varargs = params.args if params.args else params.args_varargs()
    assert len(args_varargs) == 1, f"the merge method should be able to take exactly 1 positional argument"
    configs = merge_method.get_input_configs()
    input_config = configs.args if configs.args else configs.args_varargs()[0]
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
    """
    Convert a literal or path into a `RecipeNode`, if it isn't one already.

    This helper is primarily used internally when constructing recipe graphs. For instance,
    if you pass a float to a merge method where a tensor is expected, `value_to_node` wraps
    that float in a node that will automatically convert it into a tensor when broadcasting
    over state dict keys.

    Args:
        node_or_value:
            The input that should become a `RecipeNode`.
        expected_type (type, optional):
            The target type that the value should be converted to.

    Returns:
        A `RecipeNode` instance, if conversion is successful.
    """
    if isinstance(node_or_value, RecipeNode):
        return node_or_value

    if not isinstance(node_or_value, RecipeNodeOrValue):
        raise TypeError(f"type of 'node_or_value' should be one of {typing.get_args(RecipeNodeOrValue)}, not {type(node_or_value)}")

    if expected_type is None:
        if isinstance(node_or_value, Mapping):
            expected_type = next(iter(node_or_value.values()))
        else:
            expected_type = type(node_or_value)

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
