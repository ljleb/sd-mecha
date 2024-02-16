import inspect
import pathlib
import textwrap

import torch
from sd_mecha.recipe_nodes import MergeSpace, RecipeNode, ModelRecipeNode, SymbolicRecipeNode
from sd_mecha.weight import ModelParameter
from typing import Optional, Callable, Dict, Tuple, TypeVar, Generic, get_type_hints, get_origin, Union, get_args

RecipeNodeOrModel = RecipeNode | str | pathlib.Path


T = TypeVar("T")


class LiftFlag(Generic[T]):
    def __init__(self):
        raise TypeError


class MergeMethod:
    def __init__(self, f: Callable):
        self.__f = f
        self.__name = f.__name__
        self.__validate_f()

    def __validate_f(self):
        spec = inspect.getfullargspec(self.__f)
        positional_extra = set(spec.args) - {"a", "b", "c"}
        if positional_extra:
            raise TypeError(f"Unsupported positional parameters: {positional_extra}")

        kwonlyargs_extra = set(spec.kwonlyargs) - {"alpha", "beta", "cache"}
        if kwonlyargs_extra:
            raise TypeError(f"Unsupported keyword only parameters: {kwonlyargs_extra}")

        if spec.varkw is None:
            raise TypeError(f"**kwargs must be specified")

    def __call__(self, inputs, hyper_parameters, device, dtype, cache):
        kwargs = self.__get_kwargs(inputs, hyper_parameters, device, dtype, cache)

        # pix2pix and inpainting models
        # todo: verify whether we want to slice, merge and then concat this key instead of ignore it
        if (a_size := kwargs["a"].size()) != (b_size := kwargs["b"].size()):
            if a_size[1] > b_size[1]:
                return kwargs["a"]
            else:
                return kwargs["b"]

        return self.__f(**kwargs)

    def __get_kwargs(
        self,
        inputs: Dict[str, torch.Tensor],
        hyper_parameters: Dict[str, float],
        device: str,
        dtype: Optional[torch.dtype],
        cache: Optional[Dict],
    ) -> Dict:
        if self.requests_model_c() and "c" not in inputs:
            raise ValueError

        if dtype is None:
            to_args = device,
        else:
            to_args = device, dtype

        merge_method_kwargs = {
            **{
                k: v.to(*to_args)
                for k, v in inputs.items()
            },
            **hyper_parameters,
        }
        if self.requests_cache():
            merge_method_kwargs["cache"] = cache

        return merge_method_kwargs

    def get_return_merge_space(self, a: MergeSpace, b: MergeSpace, c: Optional[MergeSpace] = None) -> MergeSpace:
        type_hints = get_type_hints(self.__f)

        resolved_input_spaces = {}

        for param, merge_space_arg in zip(('a', 'b', 'c'), (a, b, c)):
            if merge_space_arg is None:
                if self.requests_model_c():
                    raise ValueError("Missing argument c")
                continue

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

    def requests_alpha(self):
        return "alpha" in inspect.getfullargspec(self.__f)[4]

    def requests_beta(self):
        return "beta" in inspect.getfullargspec(self.__f)[4]

    def requests_model_c(self):
        return "c" in inspect.getfullargspec(self.__f)[0]

    def requests_cache(self):
        return "cache" in inspect.getfullargspec(self.__f)[4]


def convert_to_recipe(
    f: Optional[Callable] = None,
):
    if f is None:
        return lambda f: __convert_to_recipe_impl(f)
    return __convert_to_recipe_impl(f)


def __convert_to_recipe_impl(
    f: Callable,
):
    merge_method = MergeMethod(f)

    c_param = f"c: {SymbolicRecipeNode.__name__}, " if merge_method.requests_model_c() else ""
    alpha_param = f"alpha: {ModelParameter.__name__} = 0.0, " if merge_method.requests_alpha() else ""
    beta_param = f"beta: {ModelParameter.__name__} = 0.0, " if merge_method.requests_beta() else ""
    cache_param = "cache: Optional[Dict[str, torch.Tensor]] = None, " if merge_method.requests_cache() else ""

    c_arg = f"c={path_to_node.__name__}(c)," if merge_method.requests_model_c() else ""
    alpha_arg = f"alpha=alpha," if merge_method.requests_alpha() else ""
    beta_arg = f"beta=beta," if merge_method.requests_beta() else ""
    cache_arg = f"cache=cache," if merge_method.requests_cache() else ""

    fn_locals = {}
    fn_globals = globals() | {
        "merge_method": merge_method,
        "dtype": torch.dtype,
    }
    exec(textwrap.dedent(f"""
        def {f.__name__}(
            a: {SymbolicRecipeNode.__name__}, b: {SymbolicRecipeNode.__name__}, {c_param}*,
            {alpha_param}
            {beta_param}
            {cache_param}
            device: {Optional.__name__}[{str.__name__}] = None,
            dtype: {Optional.__name__}[{torch.dtype.__name__}] = None,
        ):
            return {SymbolicRecipeNode.__name__}(
                merge_method=merge_method,
                a={path_to_node.__name__}(a),
                b={path_to_node.__name__}(b),
                {c_arg}
                {alpha_arg}
                {beta_arg}
                {cache_arg}
                device=device,
                dtype=dtype,
            )
    """), fn_globals, fn_locals)
    res = fn_locals[f.__name__]
    res.__wrapped__ = f
    return res


def path_to_node(a: RecipeNodeOrModel) -> RecipeNode:
    if isinstance(a, (str, pathlib.Path)):
        return ModelRecipeNode(a)
    return a
