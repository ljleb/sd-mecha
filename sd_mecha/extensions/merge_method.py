import abc
import fuzzywuzzy.process
import inspect
import pathlib
import textwrap
import torch
import typing
from sd_mecha import extensions
from sd_mecha.hypers import Hyper
from sd_mecha.recipe_nodes import RecipeNode, ModelRecipeNode, MergeRecipeNode
from sd_mecha.extensions.merge_space import MergeSpace, MergeSpaceBase, MergeSpaceSymbolBase
from sd_mecha.extensions.model_config import ModelConfig, ModelConfigTag
from types import UnionType
from typing import Optional, Callable, Dict, Tuple, Union, List, Set, Iterable


RecipeNodeOrPath = RecipeNode | str | pathlib.Path


class StateDict(typing.Mapping[str, torch.Tensor], abc.ABC):
    @property
    @abc.abstractmethod
    def model_config(self) -> ModelConfig:
        pass

    @abc.abstractmethod
    def keys(self) -> Iterable[str]:
        pass


AnyMergeSpaceBase = MergeSpaceBase | MergeSpaceSymbolBase
_common_valid_type_permutations = (
    (torch.Tensor,),
    (torch.Tensor, AnyMergeSpaceBase),
)
_valid_return_type_permutations = _common_valid_type_permutations + (
    (torch.Tensor, ModelConfigTag),
    (torch.Tensor, ModelConfigTag, AnyMergeSpaceBase),
    (torch.Tensor, AnyMergeSpaceBase, ModelConfigTag),
)
_valid_param_type_permutations = _common_valid_type_permutations + (
    (StateDict,),
    (StateDict, ModelConfigTag),
    (StateDict, AnyMergeSpaceBase),
    (StateDict, ModelConfigTag, AnyMergeSpaceBase),
    (StateDict, AnyMergeSpaceBase, ModelConfigTag),
)


class MergeMethod:
    create_recipe: Callable

    def __init__(self, f: Callable, identifier: str, volatile_hypers: Iterable[str]):
        self.__wrapped__ = f
        self.identifier = identifier
        self.volatile_hypers = volatile_hypers
        self.__validate_f()

    def __validate_f(self):
        spec = inspect.getfullargspec(self.__wrapped__)
        hints = typing.get_type_hints(self.__wrapped__)
        if spec.defaults:
            params_with_default = spec.args[-len(spec.defaults):]
            raise TypeError(f"Default arguments are not supported for positional parameters. To declare hyperparameters, they must be keyword-only ({params_with_default})")

        for volatile_hyper in self.volatile_hypers:
            if volatile_hyper in spec.kwonlydefaults:
                continue
            elif volatile_hyper in spec.kwonlyargs:
                raise TypeError(f"Keyword-only parameter '{volatile_hyper}' was marked as volatile but it does not have a default value")

        if spec.varkw is None:
            raise TypeError(f"**kwargs must be included in the function parameters")

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

        for model_name in spec.args + ["return"]:
            model_type = hints[model_name]
            union_types = typing.get_args(model_type) if typing.get_origin(model_type) is UnionType else (model_type,)
            type_perms = _valid_param_type_permutations if model_name != "return" else _valid_return_type_permutations
            validate_type_permutation(model_name, union_types, type_perms)

        input_configs = self.get_input_configs()
        input_configs = input_configs[0] + [input_configs[1]]*int(spec.varargs is not None)
        input_configs_are_explicit = all(config is not None for config in input_configs)
        if input_configs_are_explicit and self.get_return_config(input_configs) is None:
            raise TypeError("Cannot infer the model config to return from the input model configs")

        for hyper in spec.kwonlyargs:
            if hyper in self.volatile_hypers:
                continue

            if hints[hyper] == Hyper | None:
                raise TypeError(f"The type annotation of the keyword-only parameter '{hyper}' is invalid (Optional[Hyper]). It should be`sd_mecha.Hyper`")
            if hints[hyper] != Hyper:
                raise TypeError(f"The type annotation of the keyword-only parameter '{hyper}' is invalid ({hints[hyper]}). It should be`sd_mecha.Hyper`")

    def merge_key(self, inputs: Tuple[torch.Tensor, ...], hypers: Dict[str, Hyper], key: str, device: str, dtype: torch.dtype):
        args, kwargs = self.__get_args_kwargs(inputs, hypers, key, device, dtype)
        return self.__wrapped__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.create_recipe(*args, **kwargs)

    def __get_args_kwargs(
        self,
        inputs: Tuple[torch.Tensor, ...],
        hypers: Dict[str, float],
        key: str,
        device: str,
        dtype: Optional[torch.dtype],
    ) -> Tuple[Tuple[torch.Tensor, ...], Dict]:
        if dtype is None:
            to_args = device,
        else:
            to_args = device, dtype

        for k in hypers:
            if k not in self.get_hyper_names():
                raise ValueError(f"method {self.identifier} does not have a hyperparameter '{k}'")

        merge_method_args = tuple(
            v.to(*to_args)
            for v in inputs
        )
        return merge_method_args, hypers | {
            "device": device,
            "dtype": dtype,
            "key": key,
        }

    def get_input_types(self) -> List[type]:
        return [
            annotation if annotation is None or typing.get_origin(annotation) is not UnionType else
            typing.get_args(annotation)[0]  # validation ensures the input type is index 0
            for k, annotation in typing.get_type_hints(self.__wrapped__).items()
            if k in self.get_input_names()
        ]

    def get_return_merge_space(self, merge_spaces_args: List[str]) -> str:
        type_hints = typing.get_type_hints(self.__wrapped__)
        model_names = self.get_input_names()
        varargs_name = self.get_input_varargs_name()
        if varargs_name is not None:
            model_names.extend([varargs_name] * max(0, len(merge_spaces_args) - len(model_names)))

        resolved_input_spaces = {}
        for param_name, merge_space_arg in zip(model_names, merge_spaces_args):
            annotation = type_hints.get(param_name)
            if annotation:
                merge_space_param, key = self.__extract_merge_space(annotation, param_name)

                if key in resolved_input_spaces:
                    # occurrence of already seen type var
                    if merge_space_arg != resolved_input_spaces[key]:
                        raise TypeError(f"parameter '{param_name}' of method {self.identifier} expects {resolved_input_spaces[key]} but got {merge_space_arg}")
                elif merge_space_arg in merge_space_param:
                    resolved_input_spaces[key] = merge_space_arg
                else:
                    raise TypeError(f"parameter '{param_name}' of method {self.identifier} expects {merge_space_param} but got {merge_space_arg}")

        merge_space_param, key = self.__extract_merge_space(type_hints.get("return"), "return")
        if key in resolved_input_spaces:
            return resolved_input_spaces[key]
        else:
            return next(iter(merge_space_param))

    def get_input_merge_spaces(self) -> Tuple[List[type], Optional[type]]:
        type_hints = typing.get_type_hints(self.__wrapped__)
        model_names = self.get_input_names()
        merge_spaces = []
        for param in model_names:
            annotation = type_hints.get(param)
            if annotation:
                merge_space, _ = self.__extract_merge_space(annotation, param)
                merge_spaces.append(MergeSpace[tuple(merge_space)])

        varargs_name = self.get_input_varargs_name()
        varargs_merge_space = None
        if varargs_name:
            annotation = type_hints.get(varargs_name)
            if annotation:
                varargs_merge_space, _ = self.__extract_merge_space(annotation, varargs_name)
                varargs_merge_space = MergeSpace[tuple(varargs_merge_space)]

        return merge_spaces, varargs_merge_space

    def __extract_merge_space(self, annotation: type, param_name: str) -> Tuple[List[str], str]:
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
        if not merge_space_ids:
            merge_space_ids = extensions.merge_space.get_all()
        return merge_space_ids, key

    def get_return_config(self, arg_configs: List[ModelConfig]) -> ModelConfig:
        type_hints = typing.get_type_hints(self.__wrapped__)
        params = self.get_input_names()
        varargs_name = self.get_input_varargs_name()
        if varargs_name is not None:
            params.extend([varargs_name] * max(0, len(arg_configs) - len(params)))

        return_type_hint = type_hints.get("return")
        default_config = self.__extract_model_config(return_type_hint)

        for param, arg_config in zip(params, arg_configs):
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

    def get_input_configs(self) -> Tuple[List[Optional[ModelConfig]], Optional[ModelConfig]]:
        type_hints = typing.get_type_hints(self.__wrapped__)
        model_names = self.get_input_names()

        model_configs = []
        for param in model_names:
            annotation = type_hints.get(param)
            if annotation:
                model_config = self.__extract_model_config(annotation)
                model_configs.append(model_config)

        varargs_name = self.get_input_varargs_name()
        varargs_model_config = None
        if varargs_name:
            annotation = type_hints.get(varargs_name)
            if annotation:
                varargs_model_config = self.__extract_model_config(annotation)

        return model_configs, varargs_model_config

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

    def get_input_names(self) -> List[str]:
        return inspect.getfullargspec(self.__wrapped__).args

    def get_hyper_names(self) -> Set[str]:
        return set(inspect.getfullargspec(self.__wrapped__).kwonlyargs)

    def get_volatile_hyper_names(self) -> Set[str]:
        return set(self.volatile_hypers)

    def get_default_hypers(self) -> Dict[str, Hyper]:
        return inspect.getfullargspec(self.__wrapped__).kwonlydefaults or {}

    def get_input_varargs_name(self) -> Optional[str]:
        return inspect.getfullargspec(self.__wrapped__).varargs

    def get_identifier(self) -> str:
        return self.identifier


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
    default_hypers = merge_method.get_default_hypers()

    model_params = "".join(
        f"{model_name}: {MergeRecipeNode.__name__}, "
        for model_name in merge_method.get_input_names()
    )
    hyper_params = "".join(
        f"{hyper_name}: {Hyper.__name__} = default_hypers[\"{hyper_name}\"], "
        if hyper_name in default_hypers else
        f"{hyper_name}: {Hyper.__name__}, "
        for hyper_name in merge_method.get_hyper_names()
    )

    model_args = "".join(
        f"{path_to_node.__name__}({model_name}), "
        for model_name in merge_method.get_input_names()
    )
    hyper_args = "".join(
        f"{hyper_name}={hyper_name}, "
        for hyper_name in merge_method.get_hyper_names() - merge_method.get_volatile_hyper_names()
    )
    volatile_hyper_args = "".join(
        f"{hyper_name}={hyper_name}, "
        for hyper_name in merge_method.get_volatile_hyper_names()
    )

    fn_locals = {}
    fn_globals = globals() | {
        "merge_method": merge_method,
        "default_hypers": default_hypers,
        "dtype": torch.dtype,  # `torch.dtype.__name__`
    }
    exec(textwrap.dedent(f"""
        def {fn.__name__}(
            {model_params}
            *args: {MergeRecipeNode.__name__},
            {hyper_params}
            device: {Optional.__name__}[{str.__name__}] = None,
            dtype: {Optional.__name__}[{torch.dtype.__name__}] = None,
        ):
            return {MergeRecipeNode.__name__}(
                merge_method,
                {model_args}
                *({path_to_node.__name__}(arg) for arg in args),
                hypers=dict({hyper_args}),
                volatile_hypers=dict({volatile_hyper_args}),
                device=device,
                dtype=dtype,
            )
    """), fn_globals, fn_locals)
    merge_method.create_recipe = fn_locals[fn.__name__]
    if register:
        _merge_methods_registry[identifier] = merge_method
    return merge_method


def config_converter(merge_method: MergeMethod):
    assert len(merge_method.get_input_names()) == 1, f"the merge method should take exactly 1 positional argument"
    input_param = merge_method.get_input_names()[0]
    input_config = merge_method.get_input_configs()[0][0]
    assert input_config is not None, f"the input ModelConfig['identifier...'] is missing. It should be appended to the type annotation of `{input_param}`"
    output_config = merge_method.get_return_config([input_config])
    assert output_config is not None, f"the output ModelConfig['identifier...'] is missing. It should be specified as part of the return type annotation"
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


def path_to_node(node_or_path: RecipeNodeOrPath) -> RecipeNode:
    if isinstance(node_or_path, (str, pathlib.Path)):
        return ModelRecipeNode(node_or_path)
    return node_or_path
