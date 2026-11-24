import abc
import contextlib
import dataclasses
import threading
import warnings
from collections import defaultdict
import fuzzywuzzy.process
import inspect
import pathlib
import torch
import typing
import textwrap
import ast
from copy import copy
from sd_mecha.recipe_nodes import ClosedModelRecipeNode, RecipeNode, MergeRecipeNode, LiteralRecipeNode, NonDictLiteralValue, RecipeNodeOrValue, PythonLiteralValue
from . import merge_spaces, model_configs
from .merge_spaces import MergeSpace, MergeSpaceSymbol, AnyMergeSpace
from .model_configs import ModelConfig
from ._model_config_arguments import ModelConfigRef, ParameterInfo, param_ref, resolve_config_arguments, validate_config_arguments
from types import MappingProxyType, SimpleNamespace, UnionType
from typing import Optional, Callable, Dict, Tuple, List, Iterable, Any, Generic, TypeVar, Mapping, Sequence
from ..keys_map import Clause, KeyMapBuilder, KeyMap, KeyRelation, RealizedKeyRelation
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
    interface: type | TypeVar | ModelConfigRef
    merge_space: Optional[MergeSpace | AnyMergeSpace]
    model_config: Optional[ModelConfig | ModelConfigRef]
    allow_conversion: bool = False


@dataclasses.dataclass
class ParameterType:
    data: ParameterData


def Parameter(
    annotation: type[NonDictLiteralValue | StateDict[NonDictLiteralValue]] | TypeVar | ModelConfigRef,
    merge_space: Optional[MergeSpace | str | Iterable[MergeSpace | str] | MergeSpaceSymbol] = None,
    model_config: Optional[ModelConfig | ModelConfigRef | str] = None,
    *,
    allow_conversion: bool = False,
) -> type[Any]:
    """
    Describe the value, merge space, and model config that a merge method accepts.

    Use Parameter in the signature of a function or class decorated with @merge_method.
    For example, Parameter(Tensor, "weight") accepts a tensor in weight space.

    Args:
        annotation (type):
            The Python or Torch type passed to the method for each output key. Use Tensor
            for a single tensor or StateDict[Tensor] to read the input's mapped keys by name.
            Use typing.Literal["fast", "accurate"] for a string from a fixed set of choices
            in "param" space by default. Wrap it in Optional to also accept None. Pass a
            string, per-key values, or a recipe that declares only accepted return choices.
            Use param_ref("name").map(fn) to derive a string Literal type from another
            argument's model config and merge space on a standalone merge method.
            Plain str parameters are deprecated.
        merge_space (str, MergeSpace, iterable or MergeSpaceSymbol, optional):
            The allowed merge space or spaces. For example, "weight" selects checkpoint
            weights and "delta" selects weight differences. If omitted, use the method's default.
        model_config (ModelConfig, ModelConfigRef or str, optional):
            The required model config or its registered identifier. Use param_ref("name").model_config
            to read another input's config. Use param_ref("name").map(fn) to derive configs from
            that input's model config and possible merge spaces.
        allow_conversion (bool):
            If True, use a direct match first, then search for a unique shortest route to the
            required config and merge space. A route can change the input's merge space.
            When True, model_config is required. At most one parameter per method can enable this.

    Returns:
        A type annotation that @merge_method uses to build the recipe and plan its input keys.
    """
    value_type = _state_dict_value_type(annotation)
    literal_choices = _string_literal_choices(value_type)
    if isinstance(annotation, ModelConfigRef):
        if annotation.mapping is None:
            raise TypeError("Use param_ref(name).map(fn) to derive a string Literal parameter type.")
        if merge_space is None:
            merge_space = "param"
    elif literal_choices is not None:
        _validate_literal_annotation(value_type, literal_choices, "Parameter")
        if merge_space is None:
            merge_space = "param"
    else:
        if type(None) in typing.get_args(annotation):
            annotation = typing.get_args(annotation)[0]
        if annotation is str:
            warnings.warn(
                "Using str as a merge method parameter type is deprecated and will be removed in sd-mecha 1.3.0. "
                "Use typing.Literal for fixed string choices.",
                DeprecationWarning,
                stacklevel=2,
            )
        if not isinstance(annotation, TypeVar):
            supported_types = [StateDict] + list(T.__constraints__)
            if not any(issubclass(typing.get_origin(annotation) or annotation, supported_type) for supported_type in supported_types):
                raise TypeError(f"type {annotation} should be one of {', '.join(map(lambda x: x.__name__, supported_types))}")

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
        "data": ParameterData(annotation, merge_space, model_config, allow_conversion)
    })


@dataclasses.dataclass
class ReturnType:
    data: ParameterData


def Return(
    annotation: type[NonDictLiteralValue | StateDict[NonDictLiteralValue]] | TypeVar,
    merge_space: Optional[MergeSpace | str | MergeSpaceSymbol] = None,
    model_config: Optional[ModelConfig | ModelConfigRef | str] = None,
) -> type[Any]:
    """
    Describe the return type of a merge method, optionally including its merge space and model config.

    Args:
        annotation (type):
            The Python or Torch type (e.g., `torch.Tensor`) returned by the merge method.
            Use typing.Literal["fast", "accurate"] to declare possible string results.
            Wrap it in Optional to also allow None, or in StateDict for multiple output keys.
            Literal results use "param" space by default.
        merge_space (MergeSpace, str or MergeSpaceSymbol, optional):
            The output merge space. Use a shared symbol to tie the output space to an input.
        model_config (ModelConfig, ModelConfigRef or str, optional):
            The output model config or its registered identifier. Use param_ref("name").model_config
            to read an input's config after any implicit conversion. The reference must resolve to
            exactly one config, including when you use param_ref("name").map(fn) to derive it.

    Returns:
        A type annotation object used by `@merge_method` for the return signature.
    """
    value_type = _state_dict_value_type(annotation)
    literal_choices = _string_literal_choices(value_type)
    if literal_choices is not None:
        _validate_literal_annotation(value_type, literal_choices, "Return")
        if merge_space is None:
            merge_space = "param"
    elif not isinstance(annotation, TypeVar):
        supported_types = [StateDict] + list(T.__constraints__)
        if not any(issubclass(typing.get_origin(annotation) or annotation, supported_type) for supported_type in supported_types):
            raise TypeError(f"type {annotation} should be one of {', '.join(map(lambda x: x.__name__, supported_types))}")

    if isinstance(merge_space, str):
        merge_space = merge_spaces.resolve(merge_space)

    if isinstance(model_config, str):
        model_config = model_configs.resolve(model_config)

    if getattr(model_config, "identifier", None) == "structural":
        raise ValueError("merge methods cannot convert 'structural' model configs")

    return type(Return.__name__, (ReturnType,), {
        "data": ParameterData(annotation, merge_space, model_config)
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
    def __init__(
        self,
        fn_or_cls: Callable | type,
        identifier: str,
        default_merge_space: MergeSpaceSymbol,
        interface: Optional["MergeMethodInterface"],
        reuse_outputs: bool,
        cache_factory: Callable[..., Any],
        globals_: Optional[Dict[str, Any]],
        locals_: Optional[Dict[str, Any]],
    ):
        self.__wrapped__ = fn_or_cls
        self.wrapped_is_class = inspect.isclass(fn_or_cls)
        if self.wrapped_is_class:
            if not hasattr(fn_or_cls, "__call__"):
                raise TypeError("class merge methods must define a __call__ method")
            fn = fn_or_cls.__call__
        else:
            fn = fn_or_cls

        self.default_merge_space = default_merge_space
        self.interface = interface

        signature = inspect.signature(fn, globals=globals_, locals=locals_, eval_str=True)
        return_annotation = _ensure_return(signature.return_annotation, self.default_merge_space)
        return_type = return_annotation.data.interface
        self._literal_return_choices = _literal_value_choices(return_type)
        self._literal_return_is_mapping = typing.get_origin(return_type) is StateDict
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
        self.reuse_outputs = reuse_outputs
        self.cache_factory = cache_factory
        self._has_config_arguments = validate_config_arguments(self.__f_signature)
        self._parameter_type_refs = {
            name: param.annotation.data.interface
            for name, param in self.__f_signature.parameters.items()
            if param.kind != inspect.Parameter.VAR_KEYWORD and isinstance(param.annotation.data.interface, ModelConfigRef)
        }
        if self.interface is not None and self._parameter_type_refs:
            raise TypeError("Mapped parameter types are only supported on standalone merge methods.")

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
                raise TypeError("When using a merge space symbol as output, it must also be used by at least one input parameter.")

        configs_involved = (set(getattr(config, "identifier", None) for config in input_configs.as_dict().values()) | {getattr(return_data.model_config, "identifier", None)}).difference({None})
        is_conversion_implicitly = len(configs_involved) > 1
        is_return_dict = _string_literal_choices(return_data.interface) is None and is_subclass(return_data.interface, StateDict)
        is_map_keys_defined = self.wrapped_is_class and isinstance(inspect.getattr_static(self.__wrapped__, "map_keys", None), (staticmethod, classmethod))
        if self.interface is None:
            if (is_conversion_implicitly or is_return_dict) and not is_map_keys_defined:
                raise TypeError("A merge method that converts configs must be a class merge method and define a static member 'map_keys(builder)'")
        else:
            if is_map_keys_defined:
                raise TypeError("A merge method interface cannot define 'map_keys'.")

        for reserved_name in ("key", "key_relation", "cache", "output_reused"):
            if reserved_name in names.values():
                raise TypeError(f"A merge method cannot define a parameter named {reserved_name}.")

    def instantiate(self):
        if self.wrapped_is_class:
            return self.__wrapped__()
        return None

    def create_cache(self, *args, **kwargs):
        """Create a cache by forwarding arguments to the method's cache factory."""
        return self.cache_factory(*args, **kwargs)

    def build_key_map(
        self, args_configs: Sequence[ModelConfig], kwargs_configs: Mapping[str, ModelConfig], return_config: ModelConfig,
        *, output_merge_space: Optional[MergeSpace] = None,
    ) -> KeyMap[KeyRelation]:
        return self._build_key_map(args_configs, kwargs_configs, return_config, output_merge_space=output_merge_space)

    @property
    def _has_custom_key_map(self) -> bool:
        return self.wrapped_is_class and hasattr(self.__wrapped__, "map_keys")

    @property
    def _can_map_first_input_keys(self) -> bool:
        return self._has_custom_key_map and hasattr(self.__wrapped__, "_map_first_input_keys")

    def _build_key_map(
        self,
        args_configs: Sequence[ModelConfig],
        kwargs_configs: Mapping[str, ModelConfig],
        return_config: ModelConfig,
        *,
        output_merge_space: Optional[MergeSpace] = None,
        output_keys: Optional[Iterable[str]] = None,
        first_input_keys: Optional[Iterable[str]] = None,
    ) -> KeyMap[KeyRelation]:
        input_configs = self.__f_signature.bind(*args_configs, **kwargs_configs).arguments
        input_configs = {
            p.name: input_configs.get(p.name) if input_configs.get(p.name) is not None else p.annotation.data.model_config if p.annotation.data.model_config is not None else return_config
            for p in self.__f_signature.parameters.values()
            if p.kind != p.VAR_KEYWORD
        }
        if self._has_custom_key_map:
            builder = KeyMapBuilder(MappingProxyType(input_configs), return_config, _output_merge_space=output_merge_space)
            if first_input_keys is not None and self._can_map_first_input_keys:
                config_keys = next(iter(input_configs.values())).keys()
                available_keys = (key for key in first_input_keys if key in config_keys)
                self.__wrapped__._map_first_input_keys(builder, available_keys)
            else:
                self.__wrapped__.map_keys(builder)
            res = builder.build()
        else:
            config_keys = return_config.keys()
            keys = config_keys if output_keys is None else (key for key in output_keys if key in config_keys)
            res = KeyMap(MappingProxyType({
                (key,): KeyRelation(
                    (key,),
                    (Clause({input_name: (key,) for input_name in input_configs}),),
                )
                for key in keys
            }))
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
        key_relation: RealizedKeyRelation,
        cache: Optional[dict],
        context: Optional[Any],
        output_reused: bool,
    ) -> NonDictLiteralValue | Mapping[str, NonDictLiteralValue]:
        args, kwargs = self.__get_args_kwargs(input_args, input_kwargs, key, key_relation, cache, output_reused)
        if self.wrapped_is_class:
            assert context is not None, f"class merge method {self.identifier} received self=None"
        fn = context if self.wrapped_is_class else self.__wrapped__
        result = fn(*args, **kwargs)
        if self._literal_return_choices is not None:
            if self._literal_return_is_mapping:
                if not isinstance(result, dict):
                    raise TypeError(f"Merge method {self.identifier!r} must return a dict of literal values.")
                values = result.items()
            else:
                values = ((key, result),)
            for output_key, value in values:
                _validate_literal_choice(
                    value, self._literal_return_choices, output_key,
                    subject=f"output {output_key!r} of merge method {self.identifier!r}",
                )
        return result

    def __get_args_kwargs(
        self,
        input_args: Sequence[NonDictLiteralValue | StateDict[NonDictLiteralValue]],
        input_kwargs: Mapping[str, float],
        key: str,
        key_relation: RealizedKeyRelation,
        cache: Optional[dict],
        output_reused: bool,
    ) -> Tuple[Sequence[NonDictLiteralValue | StateDict[NonDictLiteralValue]], Mapping]:
        if self.has_varkwargs:
            input_kwargs |= {
                "key": key,
                "key_relation": key_relation,
                "cache": cache,
                "output_reused": output_reused,
            }
        return input_args, input_kwargs

    def __call__(self, *args, **kwargs) -> MergeRecipeNode:
        bound_args = self.__f_signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return self.create_recipe(bound_args)

    def create_recipe(self, bound_args: inspect.BoundArguments) -> MergeRecipeNode:
        params = self.get_params()
        names = self.get_param_names()
        args = tuple(
            _parameter_value_to_node(arg, param, name)
            for arg, param, name in zip(
                bound_args.args, params.args_varargs(len(bound_args.args)), names.args_varargs(len(bound_args.args)),
            )
        )
        kwargs = {
            k: _parameter_value_to_node(arg, params.kwargs[k], k)
            for k, arg in bound_args.kwargs.items()
        }

        bound_args = self.__f_signature.bind(*args, **kwargs)
        if self._has_config_arguments:
            input_spaces = self.get_input_merge_spaces()
            spaces = dict(zip(names.args, input_spaces.args)) | input_spaces.kwargs
            if names.has_varargs():
                spaces[names.vararg] = input_spaces.vararg
            arguments, configs, return_config = resolve_config_arguments(
                self.__f_signature, bound_args, spaces, is_interface=self.interface is not None,
            )
            for name, argument in arguments.items():
                if argument is bound_args.arguments[name]:
                    continue
                choices = _literal_value_choices(self.__f_signature.parameters[name].annotation.data.interface)
                if choices is not None:
                    _validate_literal_argument(argument, choices, name)
            bound_args = inspect.BoundArguments(self.__f_signature, arguments)
            if self.interface is not None:
                return self.interface._dispatch(bound_args.args, bound_args.kwargs, configs, return_config)
            method = self._with_model_configs(configs, return_config)
            bound_args = method.__f_signature.bind(*bound_args.args, **bound_args.kwargs)
            method, bound_args = method._resolve_parameter_types(bound_args)
            return MergeRecipeNode(method, bound_args)

        if self.interface is not None:
            return self.interface.dispatch(*args, **kwargs)

        method, bound_args = self._resolve_parameter_types(bound_args)
        return MergeRecipeNode(method, bound_args)

    def _resolve_parameter_types(self, bound_args):
        """Specialize and validate mapped literals whose source metadata is known."""
        if not self._parameter_type_refs:
            return self, bound_args

        parameters = dict(self.__f_signature.parameters)
        remaining = {}
        for name, ref in self._parameter_type_refs.items():
            source = bound_args.arguments[ref.parameter]
            if source.model_config is None or source.merge_space is None:
                remaining[name] = ref
                continue
            info = ParameterInfo(source.model_config, frozenset((source.merge_space,)))
            mapped = ref.mapping(info)
            choices = _string_literal_choices(mapped)
            if choices is None:
                raise TypeError(f"The type mapping for parameter {name!r} must return a string Literal type.")
            param = parameters[name]
            data = param.annotation.data
            annotation = Parameter(
                mapped,
                data.merge_space,
                data.model_config,
                allow_conversion=data.allow_conversion,
            )
            _validate_literal_argument(bound_args.arguments[name], choices, name)
            parameters[name] = param.replace(annotation=annotation)

        if len(remaining) == len(self._parameter_type_refs):
            return self, bound_args
        method = copy(self)
        method.__f_signature = self.__f_signature.replace(parameters=parameters.values())
        method._parameter_type_refs = remaining
        return method, inspect.BoundArguments(method.__f_signature, bound_args.arguments)

    def _with_model_configs(self, configs, return_config):
        """Copy the method with concrete model configs for one recipe call.

        Args:
            configs: The resolved input configs, keyed by parameter name.
            return_config: The resolved output config, or None to retain the declaration.

        Returns:
            A MergeMethod with the resolved config annotations for key planning.
        """
        def specialize(annotation, config, name):
            declared = annotation.data.model_config
            if declared is not None and not isinstance(declared, ModelConfigRef) and declared != config:
                raise TypeError(f"{name} requires {declared.identifier!r}, got {config.identifier!r}.")
            return type(annotation.__name__, (annotation,), {
                "data": dataclasses.replace(annotation.data, model_config=config),
            })

        method = copy(self)
        signature = self.__f_signature
        return_annotation = signature.return_annotation
        if return_config is not None or isinstance(return_annotation.data.model_config, ModelConfigRef):
            return_annotation = specialize(return_annotation, return_config, "Return value")
        method.__f_signature = signature.replace(
            parameters=[
                param.replace(annotation=specialize(param.annotation, configs[name], f"Parameter {name!r}"))
                if name in configs else param
                for name, param in signature.parameters.items()
            ],
            return_annotation=return_annotation,
        )
        method._has_config_arguments = False
        method.__validate()
        return method

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

    def get_input_configs(self) -> FunctionArgs[Optional[ModelConfig | ModelConfigRef]]:
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
    def __init__(
        self,
        identifier: str,
        fn: Callable,
        default_merge_space: MergeSpaceSymbol,
        globals_: Optional[Dict[str, Any]],
        locals_: Optional[Dict[str, Any]],
    ):
        self.identifier = identifier
        self.candidates = []

        self.default_merge_space = default_merge_space

        signature = inspect.signature(fn, globals=globals_, locals=locals_, eval_str=True)
        for param in signature.parameters.values():
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                raise RuntimeError(f"Keyword-only parameter '{param.name}' is not allowed in a merge strategy.")
        return_annotation = _ensure_return(signature.return_annotation, self.default_merge_space)
        self.signature = signature.replace(
            parameters=[
                p.replace(annotation=_ensure_parameter(p.annotation, p.name, p.default, self.default_merge_space, return_annotation.data.interface))
                for p in signature.parameters.values()
                if p.kind != inspect.Parameter.VAR_KEYWORD
            ],
            return_annotation=return_annotation,
        )

    def register_implementation(self, candidate: MergeMethod):
        if candidate._parameter_type_refs:
            raise TypeError("Mapped parameter types are only supported on standalone merge methods.")
        if candidate._has_config_arguments:
            raise TypeError("Declare config references and implicit conversion on the merge method interface.")
        candidate_signature = candidate.get_signature()
        for name, parameter in self.signature.parameters.items():
            if name not in candidate_signature.parameters and _literal_value_choices(parameter.annotation.data.interface) is not None:
                raise TypeError(f"Implementation {candidate.identifier!r} must declare parameter {name!r} of {self.identifier!r}.")
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
            if not _same_state_dict_interface(candidate_data.interface, contract_data.interface):
                raise TypeError(f"Expected parameter '{candidate_name}' to have type {contract_data.interface} but got {candidate_data.interface}.")
            if contract_data.merge_space is not None and candidate_data.merge_space != contract_data.merge_space:
                raise TypeError(f"Expected parameter '{candidate_name}' to use merge space(s) {contract_data.merge_space} but got {candidate_data.merge_space}.")
            if (contract_data.model_config is not None and not isinstance(contract_data.model_config, ModelConfigRef)
                    and candidate_data.model_config != contract_data.model_config):
                raise TypeError(f"Expected parameter '{candidate_name}' to use model config {contract_data.model_config} but got {candidate_data.model_config}.")
            if contract_param.default == inspect.Parameter.empty and candidate_param.default != inspect.Parameter.empty:
                raise TypeError(f"Expected parameter '{candidate_name}' to have no default value.")

            new_parameters.append(candidate_param.replace(
                default=candidate_param.default if candidate_param.default != inspect.Parameter.empty else contract_param.default,
            ))

        candidate_data = candidate_signature.return_annotation.data
        contract_data = self.signature.return_annotation.data
        contract_choices = _literal_value_choices(contract_data.interface)
        if contract_choices is not None:
            candidate_choices = candidate._literal_return_choices
            if candidate_choices is None or not set(candidate_choices).issubset(contract_choices):
                raise TypeError(f"Return type must declare a Literal subset of {contract_choices}.")
        elif not _same_state_dict_interface(candidate_data.interface, contract_data.interface):
            raise TypeError(f"Expected return type {contract_data.interface} but got {candidate_data.interface}.")
        if contract_data.merge_space is not None and candidate_data.merge_space != contract_data.merge_space:
            raise TypeError(f"Expected return merge space {contract_data.merge_space} but got {candidate_data.merge_space}.")
        if (contract_data.model_config is not None and not isinstance(contract_data.model_config, ModelConfigRef)
                and candidate_data.model_config != contract_data.model_config):
            raise TypeError(f"Expected return model config {contract_data.model_config} but got {candidate_data.model_config}.")

        candidate_signature = candidate_signature.replace(parameters=new_parameters)
        self.candidates.append((candidate, candidate_signature))

    def dispatch(self, *args, **kwargs):
        return self._dispatch(args, kwargs, {}, None)

    def _dispatch(self, args, kwargs, configs, return_config):
        argument_graphs = {}
        failures: list[tuple[str, str]] = []

        with contextlib.ExitStack() as stack:
            for candidate, candidate_signature in self.candidates:
                try:
                    if configs or return_config is not None:
                        candidate = candidate._with_model_configs(configs, return_config)
                        signature = candidate.get_signature()
                        candidate_signature = candidate_signature.replace(parameters=[
                            parameter.replace(annotation=signature.parameters[name].annotation)
                            for name, parameter in candidate_signature.parameters.items()
                        ])
                    bound_args = self._try_candidate(
                        candidate,
                        candidate_signature,
                        argument_graphs,
                        stack,
                        args,
                        kwargs,
                    )
                    return candidate.create_recipe(bound_args)
                except CandidateDispatchError as e:
                    failures.append((candidate.identifier, str(e)))
                except Exception as e:
                    failures.append((candidate.identifier, f"unexpected error: {e}"))
        if failures:
            lines = [
                f"No candidate matched the given arguments for interface '{self.identifier}'.",
                "Attempted candidates:",
            ]
            lines.extend(f"  - {candidate_id}: {reason}" for candidate_id, reason in failures)
            raise TypeError("\n".join(lines))

        raise TypeError(f"No candidate matched the given arguments: {self.identifier}(*{args}, **{kwargs})")

    def _try_candidate(
        self,
        candidate: MergeMethod,
        candidate_signature: inspect.Signature,
        argument_graphs: dict,
        stack: contextlib.ExitStack,
        args,
        kwargs,
    ) -> inspect.BoundArguments:
        from sd_mecha import open_graph

        try:
            bound_args: inspect.BoundArguments = candidate_signature.bind(*args, **kwargs)
        except TypeError as e:
            raise CandidateDispatchError(candidate.identifier, f"signature mismatch: {e}") from e

        argument_nodes = [
            (name, node)
            for name, value in bound_args.arguments.items()
            for node in (value if candidate_signature.parameters[name].kind == inspect.Parameter.VAR_POSITIONAL else (value,))
        ]
        for parameter_name, argument_node in argument_nodes:
            contract_data = candidate_signature.parameters[parameter_name].annotation.data
            contract_model_config = contract_data.model_config
            contract_merge_spaces = contract_data.merge_space

            try:
                if argument_node not in argument_graphs:
                    argument_graph = argument_graphs[argument_node] = stack.enter_context(
                        open_graph(argument_node, root_only=True)
                    )
                else:
                    argument_graph = argument_graphs[argument_node]

                argument_candidates = argument_graph.root_candidates(
                    model_config=contract_model_config,
                    merge_space_preference=contract_merge_spaces,
                )
            except Exception as e:
                raise CandidateDispatchError(
                    candidate.identifier,
                    f"parameter '{parameter_name}': failed to analyze argument graph: {e}"
                ) from e

            # A broadcast constant can get its config later from the recipe that uses it.
            mc_satisfied = bool(argument_candidates.model_config) or (
                contract_model_config is None and _literal_value_choices(contract_data.interface) is not None
            )
            ms_satisfied = contract_merge_spaces is None or (
                    argument_candidates.merge_space is not None and
                    any(ms in contract_merge_spaces for ms in argument_candidates.merge_space)
            )

            if not mc_satisfied or not ms_satisfied:
                details = []
                if contract_model_config is not None and not mc_satisfied:
                    details.append(f"model_config requirement not satisfied ({contract_model_config.identifier})")
                if contract_merge_spaces is not None and not ms_satisfied:
                    want = sorted(ms.identifier for ms in contract_merge_spaces)
                    got = (
                        sorted(ms.identifier for ms in argument_candidates.merge_space)
                        if argument_candidates.merge_space is not None else
                        None
                    )
                    details.append(f"merge_space requirement not satisfied (wanted one of {want}, got {got})")

                raise CandidateDispatchError(
                    candidate.identifier,
                    f"parameter '{parameter_name}': " + "; ".join(details)
                )

        bound_args.apply_defaults()
        return bound_args


class CandidateDispatchError(TypeError):
    def __init__(self, candidate_identifier: str, message: str):
        super().__init__(message)
        self.candidate_identifier = candidate_identifier


def _state_dict_value_type(annotation):
    if typing.get_origin(annotation) is StateDict:
        return typing.get_args(annotation)[0]
    return annotation


def _same_state_dict_interface(i1, i2):
    return _state_dict_value_type(i1) == _state_dict_value_type(i2)


def _string_literal_choices(annotation):
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return typing.get_args(annotation)
    if origin in (typing.Union, UnionType):
        args = typing.get_args(annotation)
        if len(args) == 2 and type(None) in args:
            literal = next(arg for arg in args if arg is not type(None))
            if typing.get_origin(literal) is typing.Literal:
                return (*typing.get_args(literal), None)
    return None


def _literal_value_choices(annotation):
    return _string_literal_choices(_state_dict_value_type(annotation))


def _validate_literal_annotation(annotation, choices, role):
    string_choices = choices if typing.get_origin(annotation) is typing.Literal else choices[:-1]
    if not string_choices or not all(isinstance(choice, str) for choice in string_choices):
        raise TypeError(f"{role} Literal annotations must contain one or more string choices.")


def _unwrap_scalar_literal(value):
    node = value
    if isinstance(value, MergeRecipeNode) and value.merge_method is _module_state.merge_methods_registry.get("convert_singleton"):
        node = value.bound_args.args[0]
    if (isinstance(node, LiteralRecipeNode) and getattr(node.model_config, "identifier", None) == "singleton-mecha"
            and node.value_dict.keys() == {"key"}):
        return node.value_dict["key"]
    return value


def _validate_literal_choice(value, choices, name, *, subject=None):
    if (value is None or isinstance(value, str)) and value in choices:
        return value
    subject = subject or f"parameter {name!r}"
    if not isinstance(value, str):
        raise TypeError(f"{subject[0].upper()}{subject[1:]} requires a string from {choices}.")
    raise ValueError(f"Invalid value {value!r} for {subject}. Expected one of {choices}.")


def _validate_literal_argument(value, choices, name):
    literal = _unwrap_scalar_literal(value)
    if isinstance(literal, LiteralRecipeNode):
        literal = literal.value_dict
    if isinstance(literal, dict):
        for key, item in literal.items():
            _validate_literal_argument(item, choices, f"{name}[{key!r}]")
    elif isinstance(literal, MergeRecipeNode):
        method = literal.merge_method
        output_choices = method._literal_return_choices
        if output_choices is None:
            if method is _module_state.merge_methods_registry.get("convert_singleton"):
                _validate_literal_argument(literal.bound_args.args[0], choices, name)
                return
            raise TypeError(f"Parameter {name!r} requires a string Literal return type, but {method.identifier!r} does not declare one.")
        if not set(output_choices).issubset(choices):
            raise TypeError(f"Parameter {name!r} accepts {choices}, but {method.identifier!r} can return {output_choices}.")
    elif isinstance(literal, RecipeNode):
        raise TypeError(f"Parameter {name!r} requires a recipe with a string Literal return type or explicit string values.")
    else:
        _validate_literal_choice(literal, choices, name)


def _parameter_value_to_node(value, param, name):
    choices = _literal_value_choices(param.interface)
    if choices is not None:
        _validate_literal_argument(value, choices, name)
    elif not isinstance(param.interface, ModelConfigRef):
        return value_to_node(value, param.interface)

    if isinstance(value, LiteralRecipeNode) and _unwrap_scalar_literal(value) is not value:
        return _module_state.merge_methods_registry["convert_singleton"](value)
    if isinstance(value, dict):
        return LiteralRecipeNode(value)
    return value_to_node(value, str)


def _ensure_parameter(hint: type, param_name: str, default: Any, default_merge_space: MergeSpaceSymbol, return_interface: type):
    # remove outer `| None`
    hint_args = [arg for arg in (typing.get_args(hint) or ()) if arg is not type(None)]
    if hint_args and _literal_value_choices(hint) is None:
        hint = hint_args[0]

    if hint is Parameter:
        raise TypeError(f"the type of parameter '{param_name}' should be `sd_mecha.Parameter(...)`, not `sd_mecha.Parameter` (note the lack of parentheses)")

    if not inspect.isclass(hint) or not issubclass(hint, ParameterType):
        hint = Parameter(hint)

    choices = _literal_value_choices(hint.data.interface)
    if choices is not None and default is not inspect.Parameter.empty:
        _validate_literal_argument(default, choices, param_name)

    if hint.data.merge_space is None:
        interface = hint.data.interface
        if is_subclass(hint.data.interface, StateDict):
            interface = (typing.get_args(interface) or (T,))[0]

        if _string_literal_choices(return_interface) is None and is_subclass(return_interface, StateDict):
            return_interface = (typing.get_args(return_interface) or (T,))[0]
        if _string_literal_choices(return_interface) is not None:
            return_interface = str

        if default is inspect.Parameter.empty and is_subclass(interface, return_interface):
            hint.data.merge_space = default_merge_space
        else:
            hint.data.merge_space = {merge_spaces.resolve("param")}

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
    reuse_outputs: bool = True,
    cache_factory: Callable[..., Any] = type(None),
    globals: Optional[dict[str, Any]] = None,
    locals: Optional[dict[str, Any]] = None,
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
            The interface to implement. Match its signature, including all string Literal
            input choices. A string Literal return type may declare a subset of the
            interface's output choices.
        is_interface (bool):
            If True, marks this merge method can be overloaded with multiple implementations.
            The appropriate candidate implementation will be resolved during recipe node creation.
        reuse_outputs (bool):
            If True, marks this merge method as computationally heavy.
        cache_factory (Callable[..., Any]):
            Constructor for the method's cache. create_cache() forwards arguments to it.
            Give factory arguments defaults so callers can create a cache with no arguments.
            Defaults to a factory that returns None.

    Returns:
        A `MergeMethod` object or a decorator returning such an object.

    Raises:
        ValueError: another merge method with this identifier was already registered.
        ValueError: register is False, but is_conversion is True or dispatcher is not None.
    """
    if fn is None:
        return lambda fn: _merge_method_impl(
            fn,
            identifier=identifier,
            register=register,
            is_conversion=is_conversion,
            implements=implements,
            is_interface=is_interface,
            reuse_outputs=reuse_outputs,
            cache_factory=cache_factory,
            globals_=globals,
            locals_=locals,
        )
    return _merge_method_impl(
        fn,
        identifier=identifier,
        register=register,
        is_conversion=is_conversion,
        implements=implements,
        is_interface=is_interface,
        reuse_outputs=reuse_outputs,
        cache_factory=cache_factory,
        globals_=globals,
        locals_=locals,
    )


def _merge_method_impl(
    fn: F, *,
    identifier: Optional[str],
    register: bool,
    is_conversion: bool,
    implements: Optional[str | MergeMethod],
    is_interface: bool,
    reuse_outputs: bool,
    cache_factory: Callable[..., Any],
    globals_: Optional[dict[str, Any]],
    locals_: Optional[dict[str, Any]],
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

        if implements is not None:
            default_merge_space = module_state_copy.merge_methods_registry[implements].default_merge_space
        else:
            default_merge_space = MergeSpaceSymbol(*merge_spaces.get_all())

        if is_interface:
            _register_interface(
                fn,
                identifier,
                default_merge_space,
                module_state_copy,
                globals_,
                locals_,
            )

        fn_object = MergeMethod(
            fn,
            identifier,
            default_merge_space,
            module_state_copy.interfaces_registry.get(identifier),
            reuse_outputs,
            cache_factory,
            globals_,
            locals_,
        )
        module_state_copy.merge_methods_registry[identifier] = fn_object

        if is_conversion:
            _register_config_converter(fn_object, module_state_copy)
        if implements is not None:
            module_state_copy.interfaces_registry[implements].register_implementation(fn_object)

        _module_state = module_state_copy

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
            defaultdict(list, ((k, v.copy()) for k, v in self.converter_paths.items())),
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


def _register_config_converter(converter: MergeMethod, module_state: ModuleState):
    validate_config_converter(converter)
    input_configs = converter.get_input_configs()
    return_config = converter.get_signature().return_annotation.data.model_config
    src_config = input_configs.args[0].identifier
    tgt_config = return_config.identifier if return_config is not None else None

    module_state.conversion_registry[converter.identifier] = converter
    module_state.converter_paths[src_config].append((tgt_config, converter))


def _register_interface(
    fn: Callable,
    identifier: Optional[str],
    default_merge_space: MergeSpaceSymbol,
    module_state: ModuleState,
    globals_: Optional[Dict[str, Any]],
    locals_: Optional[Dict[str, Any]],
) -> MergeMethodInterface:
    if identifier is None:
        identifier = fn.__name__
    fn_object = MergeMethodInterface(identifier, fn, default_merge_space, globals_, locals_)

    if identifier in module_state.interfaces_registry:
        raise KeyError(f"Another merge method interface named {identifier} is already registered.")

    if not _is_empty_body_fn(fn):
        raise ValueError("register() must be applied to an empty function (the body must be either nothing, `pass`, `...`, `return` or `return None`. docstrings are allowed)")

    module_state.interfaces_registry[identifier] = fn_object
    return fn_object


def validate_config_converter(merge_method: MergeMethod):
    if merge_method._has_config_arguments:
        raise TypeError("Registered converters must declare concrete configs and cannot opt into implicit conversion.")
    params = merge_method.get_param_names()
    args_varargs = params.args if params.args else params.args_varargs()
    assert len(args_varargs) == 1, f"the merge method should be able to take exactly 1 positional argument"
    configs = merge_method.get_input_configs()
    input_config = configs.args[0] if configs.args else configs.args_varargs()[0]
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
            expected_type = type(next(iter(node_or_value.values())))
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
