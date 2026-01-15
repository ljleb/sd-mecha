import abc
import contextlib
import dataclasses
import threading
from collections import defaultdict
import fuzzywuzzy.process
import inspect
import pathlib
import torch
import typing
import textwrap
import ast
from sd_mecha.recipe_nodes import ClosedModelRecipeNode, RecipeNode, MergeRecipeNode, LiteralRecipeNode, NonDictLiteralValue, RecipeNodeOrValue, PythonLiteralValue
from . import merge_spaces, model_configs
from .merge_spaces import MergeSpace, MergeSpaceSymbol, AnyMergeSpace
from .model_configs import ModelConfig
from types import SimpleNamespace
from typing import Optional, Callable, Dict, Tuple, List, Iterable, Any, Generic, TypeVar, Mapping, Sequence
from ..keys_map import KeyMapBuilder, KeyMap, KeyRelation
from ..typing_ import is_subclass, is_instance


T = TypeVar('T', *typing.get_args(NonDictLiteralValue))


class StateDict(Mapping[str, T], Generic[T], abc.ABC):
    @property
    @abc.abstractmethod
    def model_config(self) -> ModelConfig:
        pass

    @abc.abstractmethod
    def keys(self) -> Iterable[str]:
        pass

    @abc.abstractmethod
    def __contains__(self, item):
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
    if type(None) in (typing.get_args(interface) or ()):
        interface = typing.get_args(interface)[0]

    if not isinstance(interface, TypeVar):
        supported_types = [StateDict] + list(T.__constraints__)
        if not any(issubclass(typing.get_origin(interface) or interface, supported_type) for supported_type in supported_types):
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
    interface: type[NonDictLiteralValue | StateDict[NonDictLiteralValue]] | TypeVar,
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
        supported_types = [StateDict] + list(T.__constraints__)
        if not any(issubclass(typing.get_origin(interface) or interface, supported_type) for supported_type in supported_types):
            raise TypeError(f"type {interface} should be one of {', '.join(map(lambda x: x.__name__, supported_types))}")

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
    def __init__(self, fn_or_cls: Callable | type, identifier: str):
        self.__wrapped__ = fn_or_cls
        self.wrapped_is_class = inspect.isclass(fn_or_cls)
        if self.wrapped_is_class:
            if not hasattr(fn_or_cls, "__call__"):
                raise TypeError("class merge methods must define a __call__ method")
            fn = fn_or_cls.__call__
        else:
            fn = fn_or_cls

        try:
            # merge method is overloaded if an interface is registered for it
            self.__interface = _module_state.interfaces_registry[identifier]
        except KeyError:
            self.__interface = None

        self.default_merge_space = MergeSpaceSymbol(*merge_spaces.get_all())

        signature = inspect.signature(fn)
        return_annotation = _ensure_return(signature.return_annotation, self.default_merge_space)
        self.__f_signature = signature.replace(
            parameters=[
                p.replace(annotation=_ensure_parameter(p.annotation, p.name, p.default, self.default_merge_space, return_annotation.data.interface) if p.annotation != p.empty else p)
                for p in signature.parameters.values()
            ],
            return_annotation=return_annotation
        )

        if self.wrapped_is_class:
            if not isinstance(inspect.getattr_static(fn, "__call__"), (classmethod, staticmethod)):
                self_arg = next(iter(p.name for p in self.__f_signature.parameters.values()))
                self.__f_signature = self.__f_signature.replace(parameters=[v for v in self.__f_signature.parameters.values() if v.name != self_arg])

        self.identifier = identifier
        self.has_varkwargs = any(p.kind == p.VAR_KEYWORD for p in self.__f_signature.parameters.values())

        self.__validate()

    def __validate(self):
        names = self.get_param_names().as_dict()
        params = self.get_params()  # validates param type annotations
        defaults = self.get_default_args()
        input_merge_spaces = self.get_input_merge_spaces()
        input_configs = self.get_input_configs()

        for param_idx in params.as_dict():
            is_default_arg = (
                isinstance(param_idx, int) and
                len(params.args) - len(defaults.args) <= param_idx < len(params.args)
            )
            is_default_kwarg = isinstance(param_idx, str) and self.__f_signature.parameters[param_idx].default != self.__f_signature.empty
            param_name = names[param_idx]
            if is_default_arg or is_default_kwarg:
                param_merge_space = input_merge_spaces.as_dict()[param_idx]
                if param_merge_space != {merge_spaces.resolve("param")}:
                    raise TypeError(f"The merge space for '{param_name}' should be 'param' since it has a default value.")

        return_data = self.__f_signature.return_annotation.data

        if isinstance(return_data.merge_space, MergeSpaceSymbol):
            if not any(p.merge_space == return_data.merge_space for p in params.as_dict().values()):
                raise RuntimeError("When using a merge space symbol as output, it must also be used by at least one input parameter.")

        configs_involved = (set(getattr(config, "identifier", None) for config in input_configs.as_dict().values()) | {getattr(return_data.model_config, "identifier", None)}).difference({None})
        is_conversion_implicitly = len(configs_involved) > 1
        is_return_dict = is_subclass(return_data.interface, StateDict)
        is_map_keys_defined = self.wrapped_is_class and isinstance(inspect.getattr_static(self.__wrapped__, "map_keys", None), (staticmethod, classmethod))
        if self.__interface is None:
            if (is_conversion_implicitly or is_return_dict) and not is_map_keys_defined:
                raise RuntimeError("A merge method that converts configs must be a class merge method and define a static member 'map_keys(builder)'")
        else:
            if is_map_keys_defined:
                raise RuntimeError("A merge method interface cannot define 'map_keys'.")

    def instantiate(self):
        if self.wrapped_is_class:
            return self.__wrapped__()
        return None

    def key_map(self, args_configs, kwargs_configs, return_config) -> KeyMap:
        input_configs = self.__f_signature.bind(*args_configs, **kwargs_configs).arguments
        input_configs = {
            p.name: input_configs.get(p.name) if input_configs.get(p.name) is not None else p.annotation.data.model_config if p.annotation.data.model_config is not None else return_config
            for p in self.__f_signature.parameters.values()
            if p.kind != p.VAR_KEYWORD
        }
        if self.wrapped_is_class and hasattr(self.__wrapped__, "map_keys"):
            builder = KeyMapBuilder(input_configs, return_config)
            self.__wrapped__.map_keys(builder)
            res = builder.build()
        else:
            res = KeyMap({
                (key,): KeyRelation(
                    (key,),
                    {input_name: (key,) for input_name in input_configs},
                )
                for key in return_config.keys()
            })
        return res

    def __eq__(self, other):
        return isinstance(other, MergeMethod) and self.identifier == other.identifier

    def __hash__(self):
        return hash(self.identifier)

    def __repr__(self):
        return f"<merge method '{self.identifier}'>"

    def merge_key(
        self,
        input_args: Sequence[NonDictLiteralValue | StateDict[NonDictLiteralValue]],
        input_kwargs: Mapping[str, NonDictLiteralValue | StateDict[NonDictLiteralValue]],
        key: str,
        key_relation: KeyRelation,
        cache: Optional[dict],
        context: Optional[Any],
    ) -> NonDictLiteralValue | Mapping[str, NonDictLiteralValue]:
        args, kwargs = self.__get_args_kwargs(input_args, input_kwargs, key, key_relation, cache)
        fn = self.__wrapped__
        if self.wrapped_is_class:
            assert context is not None, f"class merge method {self.identifier} received self=None"
            return context(*args, **kwargs)
        return fn(*args, **kwargs)

    def __get_args_kwargs(
        self,
        input_args: Sequence[NonDictLiteralValue | StateDict[NonDictLiteralValue]],
        input_kwargs: Mapping[str, float],
        key: str,
        key_relation: KeyRelation,
        cache: Optional[dict],
    ) -> Tuple[Sequence[NonDictLiteralValue | StateDict[NonDictLiteralValue]], Mapping]:
        if self.has_varkwargs:
            input_kwargs |= {
                "key": key,
                "key_relation": key_relation,
                "cache": cache,
            }
        return input_args, input_kwargs

    def __call__(self, *args, **kwargs) -> MergeRecipeNode:
        bound_args = self.__f_signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return self.create_recipe(bound_args)

    def create_recipe(self, bound_args: inspect.BoundArguments) -> MergeRecipeNode:
        types = self.get_input_types()
        args = tuple(
            value_to_node(arg, arg_type)
            for arg, arg_type in zip(bound_args.args, types.args_varargs(len(bound_args.args)))
        )
        kwargs = {
            k: value_to_node(arg, types.kwargs[k])
            for k, arg in bound_args.kwargs.items()
        }

        if self.__interface is not None:
            return self.__interface.dispatch(*args, **kwargs)

        return MergeRecipeNode(self, self.__f_signature.bind(*args, **kwargs))

    def get_signature(self):
        return self.__f_signature

    def get_return_type(self) -> type:
        return self.__f_signature.return_annotation

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

    def get_input_configs(self) -> FunctionArgs[Optional[ModelConfig]]:
        params = self.get_params()
        return FunctionArgs(
            [arg.model_config for arg in params.args],
            params.vararg.model_config if params.has_varargs() else FunctionArgs.EMPTY_VARARGS,
            {k: v.model_config for k, v in params.kwargs.items()},
        )

    def get_param_names(self) -> FunctionArgs[str]:
        return FunctionArgs(
            [p.name for p in self.__f_signature.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)],
            tuple((*(p.name for p in self.__f_signature.parameters.values() if p.kind == p.VAR_POSITIONAL), FunctionArgs.EMPTY_VARARGS))[0],
            {p.name: p.name for p in self.__f_signature.parameters.values() if p.kind == p.KEYWORD_ONLY},
        )

    def get_default_args(self) -> FunctionArgs[Any]:
        return FunctionArgs(
            [p.default for p in self.__f_signature.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default != p.empty],
            tuple((*(p for p in self.__f_signature.parameters.values() if p.kind == p.VAR_POSITIONAL), FunctionArgs.EMPTY_VARARGS))[0],
            {p.name: p.default for p in self.__f_signature.parameters.values() if p.kind == p.KEYWORD_ONLY and p.default != p.empty},
        )

    def get_params(self) -> FunctionArgs[ParameterData]:
        names = self.get_param_names()

        return FunctionArgs(
            [self.__f_signature.parameters[k].annotation.data for k in names.args],
            self.__f_signature.parameters[names.vararg].annotation.data if names.has_varargs() else FunctionArgs.EMPTY_VARARGS,
            {k: self.__f_signature.parameters[k].annotation.data for k in names.kwargs},
        )

    def get_identifier(self) -> str:
        return self.identifier


class MergeMethodInterface:
    def __init__(self, identifier: str, fn: Callable):
        self.identifier = identifier
        self.candidates = []

        self.default_merge_space = MergeSpaceSymbol(*merge_spaces.get_all())

        signature = inspect.signature(fn)
        for param in signature.parameters.values():
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                raise RuntimeError(f"Keyword-only parameter '{param.name}' is not allowed in a merge strategy.")
        return_annotation = _ensure_return(signature.return_annotation, self.default_merge_space)
        self.signature = signature.replace(
            parameters=[
                p.replace(annotation=_ensure_parameter(p.annotation, p.name, p.default, self.default_merge_space, return_annotation.data.interface))
                for p in signature.parameters.values()
            ],
            return_annotation=return_annotation,
        )

    def register_implementation(self, candidate: MergeMethod):
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

    def dispatch(self, *args, **kwargs):
        from .. import open_graph

        argument_graphs = {}
        with contextlib.ExitStack() as stack:
            for candidate, candidate_signature in self.candidates:
                try:
                    bound_args: inspect.BoundArguments = candidate_signature.bind(*args, **kwargs)

                    for parameter_name, argument_node in bound_args.arguments.copy().items():
                        contract_data = candidate_signature.parameters[parameter_name].annotation.data
                        contract_model_config = contract_data.model_config
                        contract_merge_spaces = contract_data.merge_space

                        if argument_node not in argument_graphs:
                            argument_graph = argument_graphs[argument_node] = stack.enter_context(open_graph(argument_node, root_only=True))
                        else:
                            argument_graph = argument_graphs[argument_node]

                        argument_candidates = argument_graph.root_candidates(
                            model_config=contract_model_config,
                            merge_space_preference=contract_merge_spaces,
                        )
                        mc_satisfied = bool(argument_candidates.model_config)
                        ms_satisfied = contract_merge_spaces is None or any(ms in contract_merge_spaces for ms in argument_candidates.merge_space)
                        if not (mc_satisfied and ms_satisfied):
                            raise TypeError

                    bound_args.apply_defaults()
                    return candidate.create_recipe(bound_args)

                except TypeError:
                    pass

        raise TypeError(f"No candidate matched the given arguments: {self.identifier}(*{args}, **{kwargs})")


def _ensure_parameter(hint: type, param_name: str, default: Any, default_merge_space: MergeSpaceSymbol, return_interface: type):
    # remove outer `| None`
    hint_args = [arg for arg in (typing.get_args(hint) or ()) if arg is not type(None)]
    if hint_args:
        hint = hint_args[0]

    if hint is Parameter:
        raise TypeError(f"the type of parameter '{param_name}' should be `sd_mecha.Parameter(...)`, not `sd_mecha.Parameter` (note the lack of parentheses)")

    if not inspect.isclass(hint) or not issubclass(hint, ParameterType):
        hint = Parameter(hint)

    if hint.data.merge_space is None:
        interface = hint.data.interface
        if is_subclass(hint.data.interface, StateDict):
            interface = (typing.get_args(interface) or (T,))[0]

        if is_subclass(return_interface, StateDict):
            return_interface = (typing.get_args(return_interface) or (T,))[0]

        if default is inspect.Parameter.empty and is_subclass(interface, return_interface):
            hint.data.merge_space = default_merge_space
        else:
            hint.data.merge_space = {"param"}

    return hint


def _ensure_return(hint: type, default_merge_space: MergeSpaceSymbol):
    if hint is Return:
        raise TypeError(f"the return type should be 'sd_mecha.Return(...)', not 'sd_mecha.Return' (note the lack of parentheses)")

    if not inspect.isclass(hint) or not issubclass(hint, ReturnType):
        hint = Return(hint)

    if hint.data.merge_space is None:
        hint.data.merge_space = default_merge_space

    return hint


F = TypeVar("F", bound=Callable | type)


def merge_method(
    fn: Optional[F] = None, *,
    identifier: Optional[str] = None,
    register: bool = True,
    is_conversion: bool = False,
    implements: Optional[str | MergeMethod] = None,
    is_interface: bool = False,
) -> MergeMethod | Callable[[F], MergeMethod]:
    """
    Decorator to define a custom merge method.

    This converts the decorated function into a `MergeMethod` object that can be used to create
    recipe graphs. Use sd_mecha.Parameter(...) as the type hint of parameters to add constraints to the inputs
    (i.e. merge space, model config).

    Args:
        fn (callable, optional):
            The function to convert to a merge method object.
        identifier (str, optional):
            An explicit name to register this method under. By default, uses `fn.__name__`.
        register (bool):
            If True (default), registers the merge method globally so it can be accessed by `merge_methods.resolve()`.
        is_conversion (bool):
            If True, marks this merge method as a config-conversion function. That means `convert()`
            will consider it as an implicit transition when converting between different model configs.
        implements (str | MergeMethod):
            The interface to implement. The signature of the merge method must match that of the interface.
        is_interface (bool):
            If True, marks this merge method can be overloaded with multiple implementations.
            The appropriate candidate implementation will be resolved during recipe node creation.

    Returns:
        A `MergeMethod` object or a decorator returning such an object.

    Raises:
        ValueError: another merge method with this identifier was already registered.
        ValueError: register is False, but is_conversion is True or dispatcher is not None.
    """
    if fn is None:
        return lambda fn: _merge_method_impl(fn, identifier=identifier, register=register, is_conversion=is_conversion, implements=implements, is_interface=is_interface)
    return _merge_method_impl(fn, identifier=identifier, register=register, is_conversion=is_conversion, implements=implements, is_interface=is_interface)


def _merge_method_impl(
    fn: F, *,
    identifier: Optional[str],
    register: bool,
    is_conversion: bool,
    implements: Optional[str | MergeMethod],
    is_interface: bool,
) -> MergeMethod:
    global _module_state

    if identifier is None:
        identifier = fn.__name__

    if isinstance(implements, MergeMethod):
        implements = implements.identifier

    if not register:
        if is_conversion:
            raise ValueError("A conversion merge method must be registered.")
        if implements is not None:
            raise ValueError("A merge method overload must be registered.")
        if is_interface:
            raise ValueError("A merge method interface must be registered.")

    if is_interface and implements is not None:
        raise ValueError("An merge method interface cannot overload a merge method interface.")

    if implements is not None and implements not in _module_state.interfaces_registry:
        raise ValueError(f"The provided merge method interface {implements} is not an interface.")

    with _module_state.registry_lock:
        if register:
            if identifier in _module_state.merge_methods_registry:
                raise ValueError(f"Another merge method named {identifier} is already registered.")

        module_state_copy = _module_state.copy()
        try:
            if is_interface:
                _register_interface(fn, identifier)
            fn_object = MergeMethod(fn, identifier)
            _module_state.merge_methods_registry[identifier] = fn_object
            if is_conversion:
                _register_config_converter(fn_object)
            if implements is not None:
                _module_state.interfaces_registry[implements].register_implementation(fn_object)
        except BaseException:
            _module_state = module_state_copy
            raise

    return fn_object


@dataclasses.dataclass
class ModuleState:
    registry_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    merge_methods_registry: Dict[str, MergeMethod] = dataclasses.field(default_factory=dict)
    conversion_registry: Dict[str, MergeMethod] = dataclasses.field(default_factory=dict)
    converter_paths: Dict[str, List[Tuple[str, MergeMethod]]] = dataclasses.field(default_factory=lambda: defaultdict(list))
    interfaces_registry: Dict[str, MergeMethodInterface] = dataclasses.field(default_factory=dict)

    def copy(self):
        return ModuleState(
            self.registry_lock,
            self.merge_methods_registry.copy(),
            self.conversion_registry.copy(),
            self.converter_paths.copy(),
            self.interfaces_registry.copy(),
        )


_module_state = ModuleState()


def resolve(identifier: str) -> MergeMethod:
    try:
        return _module_state.merge_methods_registry[identifier]
    except KeyError as e:
        suggestion = fuzzywuzzy.process.extractOne(str(e), _module_state.merge_methods_registry.keys())[0]
        raise KeyError(f"unknown merge method: {e}. Nearest match is '{suggestion}'")


def get_all() -> List[MergeMethod]:
    return list(_module_state.merge_methods_registry.values())


def get_all_converters() -> List[MergeMethod]:
    return list(_module_state.conversion_registry.values())


def get_converter_paths() -> Dict[str, List[Tuple[str, MergeMethod]]]:
    return _module_state.converter_paths.copy()


def _register_config_converter(converter: MergeMethod):
    validate_config_converter(converter)
    input_configs = converter.get_input_configs()
    return_config = converter.get_signature().return_annotation.data.model_config
    src_config = input_configs.args[0].identifier
    tgt_config = return_config.identifier if return_config is not None else None

    _module_state.conversion_registry[converter.identifier] = converter
    _module_state.converter_paths[src_config].append((tgt_config, converter))


def _register_interface(
    fn: Callable,
    identifier: Optional[str]
) -> MergeMethodInterface:
    global _module_state

    if identifier is None:
        identifier = fn.__name__
    fn_object = MergeMethodInterface(identifier, fn)

    if identifier in _module_state.interfaces_registry:
        raise KeyError(f"Another merge method interface named {identifier} is already registered.")

    if not _is_empty_body_fn(fn):
        raise ValueError("register() must be applied to an empty function (the body must be either nothing, `pass`, `...`, `return` or `return None`. docstrings are allowed)")

    _module_state.interfaces_registry[identifier] = fn_object
    return fn_object


def validate_config_converter(merge_method: MergeMethod):
    params = merge_method.get_param_names()
    args_varargs = params.args if params.args else params.args_varargs()
    assert len(args_varargs) == 1, f"the merge method should be able to take exactly 1 positional argument"
    configs = merge_method.get_input_configs()
    input_config = configs.args if configs.args else configs.args_varargs()[0]
    assert input_config is not None, f"the input model config is missing. It should be declared in the type of `{args_varargs[0]}`"
    return merge_method


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

    if not is_instance(node_or_value, RecipeNodeOrValue):
        raise TypeError(f"type of 'node_or_value' should be one of {typing.get_args(RecipeNodeOrValue)}, not {type(node_or_value)}")

    if expected_type is None:
        if isinstance(node_or_value, Mapping):
            expected_type = next(iter(node_or_value.values()))
        else:
            expected_type = type(node_or_value)

        if issubclass(expected_type, pathlib.Path):
            expected_type = torch.Tensor

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

    if isinstance(node_or_value, NonDictLiteralValue):
        res = LiteralRecipeNode({"key": node_or_value}, model_config="singleton-mecha")
        res = _module_state.merge_methods_registry["convert_singleton"](res)
        return res

    if isinstance(node_or_value, dict):
        return LiteralRecipeNode(node_or_value)

    if isinstance(node_or_value, pathlib.Path) and (isinstance(expected_type, TypeVar) or issubclass(expected_type, torch.Tensor)):
        try:
            return ClosedModelRecipeNode(node_or_value)
        except TypeError as e:
            base_error_message = f"No implicit conversion exists from {type(node_or_value)} to Dict[Tensor]"
            if isinstance(node_or_value, str):
                raise TypeError(f"{base_error_message}. Consider using pathlib.Path instead if a model path was intended") from e
            if not isinstance(node_or_value, pathlib.Path):
                raise TypeError(base_error_message) from e

    raise TypeError(f"No implicit conversion exists from {type(node_or_value)} to {expected_type}")


def _is_empty_body_fn(func) -> bool:
    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):  # no source available (builtins, C-ext, interactive, etc.)
        return False

    src = textwrap.dedent(src)
    node = ast.parse(src)

    # Find the first function/async function node in that source block
    fn = next(
        (n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None
    )
    if fn is None:
        return False

    # Ignore leading docstring if present
    body = fn.body[:]
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    # Empty after docstring => "empty"
    if not body:
        return True

    # Exactly one statement: `pass`, `ellipsis`, `return` or `return None`
    if len(body) == 1:
        stmt = body[0]
        if isinstance(stmt, ast.Pass):
            return True
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(getattr(stmt, "value", None), ast.Constant)
            and stmt.value.value is Ellipsis
        ):
            return True
        if (
            isinstance(stmt, ast.Return)
            and (
                stmt.value is None or
                isinstance(stmt.value, ast.Constant) and stmt.value.value is None
            )
        ):
            return True

    return False
