import abc
import dataclasses
import itertools
import fuzzywuzzy.process
import inspect
import pathlib
import torch
import typing
from sd_mecha import extensions
from sd_mecha.extensions import merge_space
from sd_mecha.recipe_nodes import RecipeNode, ModelRecipeNode, MergeRecipeNode, LiteralRecipeNode, RecipeVisitor
from sd_mecha.extensions.merge_space import MergeSpace, MergeSpaceSymbolBase, AnyMergeSpaceBase
from sd_mecha.extensions.model_config import ModelConfig, ModelConfigTag
from types import UnionType, SimpleNamespace
from typing import Optional, Callable, Dict, Tuple, List, Set, Iterable, Any, Generic, TypeVar, Mapping


RecipeNodeOrLiteral = RecipeNode | pathlib.Path | str | int | float | dict


T = TypeVar('T', torch.Tensor, str)
class StateDict(Mapping[str, T], Generic[T], abc.ABC):
    @property
    @abc.abstractmethod
    def model_config(self) -> ModelConfig:
        pass

    @abc.abstractmethod
    def keys(self) -> Iterable[str]:
        pass


P = TypeVar('P')
@dataclasses.dataclass
class FunctionArgs(Generic[P]):
    args: List[P]
    vararg: P | SimpleNamespace  # using SimpleNamespace as "empty" because P can be Optional[X]
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


_common_valid_type_permutations = tuple(
    perm
    for t in T.__constraints__
    for perm in (
        (t,),
        (t, AnyMergeSpaceBase),
    )
)
_valid_return_type_permutations = _common_valid_type_permutations + tuple(
    perm
    for t in T.__constraints__
    for perm in (
        (t, ModelConfigTag),
        (t, ModelConfigTag, AnyMergeSpaceBase),
        (t, AnyMergeSpaceBase, ModelConfigTag),
    )
)
_valid_param_type_permutations = _common_valid_type_permutations + tuple(
    perm
    for t in T.__constraints__
    for perm in (
        (StateDict[t],),
        (StateDict[t], ModelConfigTag),
        (StateDict[t], AnyMergeSpaceBase),
        (StateDict[t], ModelConfigTag, AnyMergeSpaceBase),
        (StateDict[t], AnyMergeSpaceBase, ModelConfigTag),
    )
)


class MergeMethod:
    def __init__(self, f: Callable, identifier: str, volatile_params: Iterable[str]):
        self.__wrapped__ = f
        self.identifier = identifier
        self.volatile_params = volatile_params
        self.__validate_f()

    def __validate_f(self):
        spec = inspect.getfullargspec(self.__wrapped__)
        hints = typing.get_type_hints(self.__wrapped__)
        merge_spaces = self.get_input_merge_spaces()
        input_configs = self.get_input_configs()

        for volatile_hyper in self.volatile_params:
            if volatile_hyper in spec.kwonlydefaults:
                continue
            elif volatile_hyper in spec.kwonlyargs:
                raise TypeError(f"Keyword-only parameter '{volatile_hyper}' was marked as volatile but it does not have a default value")

        if spec.varkw is None:
            raise TypeError(f"for forward compatibility reasons, **kwargs must be included in the function parameters")

        def type_to_str(a: type):
            if typing.get_origin(a) is UnionType or issubclass(a, AnyMergeSpaceBase):
                return "MergeSpace[...]"
            elif issubclass(a, ModelConfigTag):
                return "ModelConfig[...]"
            else:
                return a.__name__

        def validate_type_permutation(parameter_name, parameter_types, type_perms):
            is_valid_type_combination = any(
                len(valid_type_combination) == len(parameter_types) and
                all(issubclass(t, p) for t, p in zip(parameter_types, valid_type_combination))
                for valid_type_combination in type_perms
            )
            if not is_valid_type_combination:
                raise TypeError(
                    f"The type annotation for '{parameter_name}' is invalid. \n"
                    f"Got: {' | '.join(map(type_to_str, parameter_types))}\n"
                    "Valid choices are:\n\t" + "\n\t".join(" | ".join(map(type_to_str, type_comb)) for type_comb in type_perms)
                )

        for param_idx in itertools.chain(range(len(spec.args) + int(spec.varargs is not None)), ["return"] + spec.kwonlyargs):
            param_name = spec.args[param_idx] if isinstance(param_idx, int) else param_idx
            if param_name in self.volatile_params:
                continue

            if isinstance(param_idx, int) and param_idx >= len(spec.args) - len(spec.defaults):
                if merge_spaces.args[param_idx] not in (MergeSpace["param"], None):
                    raise TypeError(f"The merge space for '{param_name}' should be 'param' since it has a default value.")

            param_type = hints[param_name]
            union_types = typing.get_args(param_type) if typing.get_origin(param_type) is UnionType else (param_type,)
            type_perms = _valid_param_type_permutations if param_name != "return" else _valid_return_type_permutations
            validate_type_permutation(param_name, union_types, type_perms)

        input_configs_are_explicit = all(config is not None for config in input_configs.as_dict().values())
        if input_configs_are_explicit and self.get_return_config(input_configs.args_varags(), input_configs.kwargs) is None:
            raise TypeError("Cannot infer the model config to return from the input model configs")

    def merge_key(
        self,
        input_args: Tuple[torch.Tensor | StateDict, ...],
        input_kwargs: Dict[str, torch.Tensor | StateDict, ...],
        key: str,
    ):
        args, kwargs = self.__get_args_kwargs(input_args, input_kwargs, key)
        return self.__wrapped__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.create_recipe(*args, **kwargs)

    def __get_args_kwargs(
        self,
        input_args: Tuple[Any, ...],
        input_kwargs: Dict[str, float],
        key: str,
    ) -> Tuple[Tuple[torch.Tensor, ...], Dict]:
        return input_args, input_kwargs | {
            "key": key,
        }

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
        merge_spaces = self.get_input_merge_spaces()
        default_config = self.get_return_config(input_configs.args_varags(varargs_count), input_configs.kwargs)

        def arg_to_node(k: int | str, arg: Any, expected_type: type):
            nonlocal default_config
            merge_space = merge_spaces.as_dict(varargs_count)[k]
            return value_to_node(arg, expected_type).accept(InferModelConfigVisitor(default_config, merge_space))

        input_types = self.get_input_types()
        args = (
            arg_to_node(i, arg, input_types.args[i]) if k not in self.volatile_params else arg
            for i, (arg, k) in enumerate(zip(args, params.args_varags(varargs_count)))
        )
        kwargs = {
            k: arg_to_node(k, arg, input_types.kwargs[k]) if k not in self.volatile_params else arg
            for k, arg in kwargs.items()
        }

        return MergeRecipeNode(self, *args, **kwargs)

    def get_input_types(self) -> FunctionArgs[type]:
        params = self.get_param_names()
        type_hints = typing.get_type_hints(self.__wrapped__)

        def get_empirical_type(annotation):
            if annotation is None or typing.get_origin(annotation) is not UnionType:
                return annotation
            return typing.get_args(annotation)[0]  # validation ensures the input type is index 0

        arg_types = [
            get_empirical_type(annotation)
            for k, annotation in type_hints.items()
            if k in params.args
        ]
        vararg_type = get_empirical_type(type_hints.get(params.vararg)) if params.has_varargs() else FunctionArgs.EMPTY_VARARGS
        kwarg_types = {
            k: get_empirical_type(annotation)
            for k, annotation in type_hints.items()
            if k in params.kwargs
        }
        return FunctionArgs(arg_types, vararg_type, kwarg_types)

    def get_return_merge_space(self, merge_space_args: List[str], merge_space_kwargs: Dict[str, str]) -> str:
        type_hints = typing.get_type_hints(self.__wrapped__)
        params = self.get_param_names()

        resolved_input_spaces = {}
        arg_tuples = zip(params.args_varags(max(0, len(merge_space_args) - len(params.args))), merge_space_args)
        kwarg_tuples = ((k, v) for k, v in merge_space_kwargs.items())
        for param_name, merge_space_arg in itertools.chain(arg_tuples, kwarg_tuples):
            annotation = type_hints.get(param_name)
            if annotation:
                merge_space_param, key, _ = self.__extract_merge_space(annotation, param_name)

                if key in resolved_input_spaces:
                    # occurrence of already seen type var
                    if merge_space_arg != resolved_input_spaces[key]:
                        raise TypeError(f"parameter '{param_name}' of method {self.identifier} expects {resolved_input_spaces[key]} but got {merge_space_arg}")
                elif merge_space_arg in merge_space_param:
                    resolved_input_spaces[key] = merge_space_arg
                else:
                    raise TypeError(f"parameter '{param_name}' of method {self.identifier} expects {merge_space_param} but got {merge_space_arg}")

        merge_space_param, key, _ = self.__extract_merge_space(type_hints.get("return"), "return")
        if key in resolved_input_spaces:
            return resolved_input_spaces[key]
        else:
            return next(iter(merge_space_param))

    def get_input_merge_spaces(self) -> FunctionArgs[type]:
        type_hints = typing.get_type_hints(self.__wrapped__)
        params = self.get_param_names()
        defaults = self.get_default_args()

        merge_spaces = []
        for i, param in enumerate(params.args):
            annotation = type_hints.get(param)
            merge_space, _, explicit = self.__extract_merge_space(annotation, param)
            if not explicit and i >= len(params.args) - len(defaults.args):
                merge_space = "param",
            merge_spaces.append(MergeSpace[tuple(merge_space)])

        varargs_merge_space = FunctionArgs.EMPTY_VARARGS
        if params.has_varargs():
            annotation = type_hints.get(params.vararg)
            varargs_merge_space, _, explicit = self.__extract_merge_space(annotation, params.vararg)
            varargs_merge_space = MergeSpace[tuple(varargs_merge_space)]

        kw_merge_spaces = {}
        for param in params.kwargs:
            annotation = type_hints.get(param)
            merge_space, _, explicit = self.__extract_merge_space(annotation, param)
            if not explicit and param in defaults.kwargs:
                merge_space = "param",
            kw_merge_spaces[param] = MergeSpace[tuple(merge_space)]

        return FunctionArgs(merge_spaces, varargs_merge_space, kw_merge_spaces)

    def __extract_merge_space(self, annotation: type, param_name: str) -> Tuple[List[str], str, bool]:
        if typing.get_origin(annotation) is UnionType:
            type_args = typing.get_args(annotation)
        elif annotation is not None:
            type_args = (annotation,)
        else:
            type_args = ()

        key = param_name
        for type_arg in type_args:
            if issubclass(type_arg, MergeSpaceSymbolBase):
                key = type_arg.__name__
                break

        merge_space_ids = extensions.merge_space.get_identifiers(annotation)
        explicit = True
        if not merge_space_ids:
            merge_space_ids = extensions.merge_space.get_all()
            explicit = False
        return merge_space_ids, key, explicit

    def get_return_config(self, arg_configs: List[Optional[ModelConfig]], kwarg_configs: Dict[str, Optional[ModelConfig]]) -> ModelConfig:
        type_hints = typing.get_type_hints(self.__wrapped__)
        params = self.get_param_names()
        default_config = self.__extract_model_config(type_hints.get("return"))

        arg_tuples = zip(params.args_varags(max(0, len(arg_configs) - len(params.args))), arg_configs)
        kwarg_tuples = ((k, kwarg_configs.get(k)) for k in kwarg_configs)
        for param, arg_config in itertools.chain(arg_tuples, kwarg_tuples):
            if arg_config is None:
                continue

            annotation = type_hints.get(param)
            param_config = self.__extract_model_config(annotation)
            if param_config is None:
                param_config = default_config
            if param_config is None:
                param_config = arg_config
                default_config = arg_config
            if param_config.identifier != arg_config.identifier:
                raise ValueError(f"Recipe received an incompatible input: expected model config {param_config.identifier} but instead got {arg_config.identifier}")

        return default_config

    def get_input_configs(self) -> FunctionArgs[Optional[ModelConfig]]:
        type_hints = typing.get_type_hints(self.__wrapped__)
        param_names = self.get_param_names()

        arg_configs = []
        for param_name in param_names.args:
            annotation = type_hints.get(param_name)
            config = self.__extract_model_config(annotation)
            arg_configs.append(config)

        vararg_model_config = FunctionArgs.EMPTY_VARARGS
        if param_names.has_varargs():
            annotation = type_hints.get(param_names.vararg)
            vararg_model_config = self.__extract_model_config(annotation)

        kwarg_configs = {}
        for param_name in param_names.kwargs:
            annotation = type_hints.get(param_name)
            config = self.__extract_model_config(annotation)
            kwarg_configs[param_name] = config

        res = FunctionArgs(arg_configs, vararg_model_config, kwarg_configs)
        return res
        # default_config = self.get_return_config(res.args_varags(), res.kwargs)

    def __extract_model_config(self, annotation) -> Optional[ModelConfig]:
        if typing.get_origin(annotation) is UnionType:
            type_args = typing.get_args(annotation)
        elif annotation is not None:
            type_args = (annotation,)
        else:
            type_args = ()

        for type_arg in type_args:
            if issubclass(type_arg, ModelConfigTag):
                return type_arg.config

        return None

    def get_param_names(self) -> FunctionArgs[str]:
        spec = inspect.getfullargspec(self.__wrapped__)
        return FunctionArgs(
            spec.args,
            spec.varargs or FunctionArgs.EMPTY_VARARGS,
            dict(zip(spec.kwonlyargs, spec.kwonlyargs)),
        )

    def get_default_args(self) -> FunctionArgs[Any]:
        spec = inspect.getfullargspec(self.__wrapped__)
        return FunctionArgs(list(spec.defaults or ()), FunctionArgs.EMPTY_VARARGS, spec.kwonlydefaults or {})

    def get_volatile_names(self) -> Set[str]:
        return set(self.volatile_params)

    def get_identifier(self) -> str:
        return self.identifier


@dataclasses.dataclass
class InferModelConfigVisitor(RecipeVisitor):
    default_model_config: ModelConfig
    default_merge_space: type

    def visit_literal(self, node: ModelRecipeNode):
        model_config = node.model_config if node.model_config is not None else self.default_model_config
        return LiteralRecipeNode(node.state_dict, model_config)

    def visit_model(self, node: ModelRecipeNode):
        node_merge_space = node.merge_space
        default_merge_spaces = merge_space.get_identifiers(self.default_merge_space)
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


def convert_to_recipe(
    f: Optional[Callable] = None, *,
    identifier: Optional[str] = None,
    volatile_hypers: Iterable[str] = (),
    register: bool = True,
):
    if f is None:
        return lambda f: __convert_to_recipe_impl(f, identifier=identifier, volatile_hypers=volatile_hypers, register=register)
    return __convert_to_recipe_impl(f, identifier=identifier, volatile_hypers=volatile_hypers, register=register)


def __convert_to_recipe_impl(
    fn: Callable, *,
    identifier: Optional[str] = None,
    volatile_hypers: Iterable[str],
    register: bool,
):
    if identifier is None:
        identifier = fn.__name__
    merge_method = MergeMethod(fn, identifier, volatile_hypers)

    if register:
        _merge_methods_registry[identifier] = merge_method
    return merge_method


def implicit_config_conversion(merge_method: MergeMethod):
    params = merge_method.get_param_names()
    args_varargs = params.args if params.args else params.args_varags(1)
    assert len(args_varargs) == 1, f"the merge method should be able to take exactly 1 positional argument"
    configs = merge_method.get_input_configs()
    input_config = configs.args if configs.args else configs.args_varags(1)[0]
    assert input_config is not None, f"the input ModelConfig['identifier...'] is missing. It should be appended to the type annotation of `{args_varargs[0]}`"
    return merge_method


_merge_methods_registry = {}


def resolve(identifier: str) -> MergeMethod:
    try:
        return _merge_methods_registry[identifier]
    except KeyError as e:
        suggestion = fuzzywuzzy.process.extractOne(str(e), _merge_methods_registry.keys())[0]
        raise ValueError(f"unknown merge method: {e}. Nearest match is '{suggestion}'")


def get_all() -> List[MergeMethod]:
    return list(_merge_methods_registry.values())


def value_to_node(node_or_value: RecipeNodeOrLiteral, expected_value_type: type = torch.Tensor) -> RecipeNode:
    if isinstance(node_or_value, RecipeNode):
        return node_or_value

    if issubclass(expected_value_type, StateDict):
        expected_value_type = typing.get_args(expected_value_type)[0]

    if isinstance(node_or_value, expected_value_type):
        return LiteralRecipeNode(node_or_value)

    if issubclass(expected_value_type, torch.Tensor):
        if isinstance(node_or_value, int | float):
            return LiteralRecipeNode(node_or_value)
        return ModelRecipeNode(node_or_value)

    if issubclass(expected_value_type, str):
        if isinstance(node_or_value, dict):
            return LiteralRecipeNode(node_or_value)
        raise TypeError(f"No implicit conversion exist from str to torch.Tensor")

    raise TypeError(f"Type {expected_value_type} is not yet supported in merge methods")
