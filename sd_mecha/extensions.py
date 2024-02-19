import functools
import inspect
import pathlib
import textwrap

import torch
from sd_mecha.recipe_nodes import MergeSpace, RecipeNode, ModelRecipeNode, MergeRecipeNode
from sd_mecha.hypers import Hyper
from typing import Optional, Callable, Dict, Tuple, TypeVar, Generic, get_type_hints, get_origin, Union, get_args, List, Set, Iterable


RecipeNodeOrPath = RecipeNode | str | pathlib.Path
T = TypeVar("T")


class LiftFlag(Generic[T]):
    def __init__(self):
        raise TypeError


class MergeMethod:
    def __init__(self, f: Callable, volatile_hypers: Iterable[str]):
        self.__f = f
        self.__name = f.__name__
        self.__volatile_hypers = volatile_hypers
        self.__validate_f()

    def __validate_f(self):
        spec = inspect.getfullargspec(self.__f)
        if spec.defaults:
            params_with_default = spec.args[-len(spec.defaults):]
            raise TypeError(f"Default arguments are not supported for positional parameters. To declare hyperparameters, they must be keyword-only ({params_with_default})")

        for volatile_hyper in self.__volatile_hypers:
            if volatile_hyper not in spec.kwonlydefaults:
                raise TypeError(f"Keyword-only parameter '{volatile_hyper}' was marked as volatile but it missing or does not have a default value")

        if spec.varkw is None:
            raise TypeError(f"**kwargs must be included in the function parameters")

    def __call__(self, inputs: Tuple[torch.Tensor, ...], hypers: Dict[str, Hyper], device, dtype):
        args, kwargs = self.__get_args_kwargs(inputs, hypers, device, dtype)
        return self.__f(*args, **kwargs)

    def __get_args_kwargs(
        self,
        inputs: Tuple[torch.Tensor, ...],
        hypers: Dict[str, float],
        device: str,
        dtype: Optional[torch.dtype],
    ) -> Tuple[Tuple[torch.Tensor, ...], Dict]:
        if dtype is None:
            to_args = device,
        else:
            to_args = device, dtype

        merge_method_args = tuple(
            v.to(*to_args)
            for v in inputs
        )
        return merge_method_args, hypers

    def get_return_merge_space(self, merge_spaces_args: List[MergeSpace]) -> MergeSpace:
        type_hints = get_type_hints(self.__f)
        model_names = self.get_model_names()
        varargs_name = self.get_model_varargs_name()
        if varargs_name is not None:
            model_names.extend([varargs_name]*(len(merge_spaces_args) - len(model_names)))

        resolved_input_spaces = {}
        for param, merge_space_arg in zip(model_names, merge_spaces_args):
            annotation = type_hints.get(param)
            if annotation:
                merge_space_param, key = self.__extract_lift_flag(annotation, param)

                if key in resolved_input_spaces:
                    # occurrence of already seen type var
                    if merge_space_arg != resolved_input_spaces[key]:
                        raise TypeError(f"parameter '{param}' of method {self.__name} expects {resolved_input_spaces[key]} but got {merge_space_arg}")
                elif merge_space_arg & merge_space_param:
                    resolved_input_spaces[key] = merge_space_arg
                else:
                    raise TypeError(f"parameter '{param}' expects {merge_space_param} but got {merge_space_arg}")

        merge_space_param, key = self.__extract_lift_flag(type_hints.get("return"), None)
        if key in resolved_input_spaces:
            return resolved_input_spaces[key]
        else:
            return merge_space_param

    def __extract_lift_flag(self, annotation, param) -> Tuple[MergeSpace, object]:
        if get_origin(annotation) is Union:
            for arg in get_args(annotation):
                if get_origin(arg) is LiftFlag:
                    return get_args(arg)[0], param
                elif isinstance(arg, TypeVar):
                    return get_args(arg.__bound__)[0], arg

    def get_model_names(self) -> List[str]:
        return inspect.getfullargspec(self.__f).args

    def get_hyper_names(self) -> Set[str]:
        return set(inspect.getfullargspec(self.__f).kwonlyargs)

    def get_volatile_hyper_names(self) -> Set[str]:
        return set(self.__volatile_hypers)

    def get_default_hypers(self) -> Dict[str, Hyper]:
        return inspect.getfullargspec(self.__f).kwonlydefaults or {}

    def get_model_varargs_name(self) -> Optional[str]:
        return inspect.getfullargspec(self.__f).varargs

    def get_name(self) -> str:
        return self.__name


def convert_to_recipe(
    f: Optional[Callable] = None, *,
    volatile_hypers: Iterable[str] = (),
):
    if f is None:
        return lambda f: __convert_to_recipe_impl(f, volatile_hypers=volatile_hypers)
    return __convert_to_recipe_impl(f, volatile_hypers=volatile_hypers)


def __convert_to_recipe_impl(
    f: Callable, *,
    volatile_hypers: Iterable[str],
):
    merge_method = MergeMethod(f, volatile_hypers)
    default_hypers = merge_method.get_default_hypers()

    model_params = "".join(
        f"{model_name}: {MergeRecipeNode.__name__}, "
        for model_name in merge_method.get_model_names()
    )
    hyper_params = "".join(
        f"{hyper_name}: {Hyper.__name__} = default_hypers[\"{hyper_name}\"], "
        if hyper_name in default_hypers else
        f"{hyper_name}: {Hyper.__name__}, "
        for hyper_name in merge_method.get_hyper_names()
    )

    model_args = "".join(
        f"{path_to_node.__name__}({model_name}), "
        for model_name in merge_method.get_model_names()
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
        def {f.__name__}(
            {model_params}
            *args: {MergeRecipeNode.__name__},
            {hyper_params}
            device: {Optional.__name__}[{str.__name__}] = None,
            dtype: {Optional.__name__}[{torch.dtype.__name__}] = None,
            **kwargs,
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
    res = fn_locals[f.__name__]
    res.__wrapped__ = f
    methods_registry[f.__name__] = res
    return res


methods_registry = {}


@functools.cache
def path_to_node(a: RecipeNodeOrPath) -> RecipeNode:
    if isinstance(a, (str, pathlib.Path)):
        return ModelRecipeNode(a)
    return a


def clear_model_paths_cache():
    path_to_node.cache_clear()
