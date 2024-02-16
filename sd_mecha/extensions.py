import enum
import inspect
import torch
from typing import Optional, Callable, Dict, Tuple, TypeVar, Generic, get_type_hints, get_origin, Union, get_args

T = TypeVar("T")


class LiftFlag(Generic[T]):
    def __init__(self):
        raise TypeError


class MergeSpace(enum.Flag):
    MODEL = enum.auto()
    DELTA = enum.auto()


class MergeMethod:
    def __init__(self, f: Callable):
        self.__f = f
        self.__name = f.__name__

    def __call__(self, inputs, hyper_parameters, device, dtype, cache):
        kwargs = self._get_kwargs(inputs, hyper_parameters, device, dtype, cache)

        # pix2pix and inpainting models
        # todo: verify whether we want to slice, merge and then concat this key instead of ignore it
        if (a_size := kwargs["a"].size()) != (b_size := kwargs["b"].size()):
            if a_size[1] > b_size[1]:
                return kwargs["a"]
            else:
                return kwargs["b"]

        return self.__f(**kwargs)

    def _get_kwargs(
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
                merge_space_param, key = self._extract_liftflag(annotation, param)

                if key in resolved_input_spaces:
                    # occurrence of already seen type var
                    if merge_space_arg != resolved_input_spaces[key]:
                        raise TypeError(f"parameter '{param}' of method {self.__name} expects {resolved_input_spaces[key]} but got {merge_space_arg}")
                elif merge_space_arg & merge_space_param:
                    resolved_input_spaces[key] = merge_space_arg
                else:
                    raise TypeError(f"parameter '{param}' expects {merge_space_param} but got {merge_space_arg}")

        merge_space_param, key = self._extract_liftflag(type_hints.get("return"), None)
        if key in resolved_input_spaces:
            return resolved_input_spaces[key]
        else:
            return merge_space_param

    def _extract_liftflag(self, annotation, param) -> Tuple[MergeSpace, object]:
        if get_origin(annotation) is Union:
            for arg in get_args(annotation):
                if get_origin(arg) is LiftFlag:
                    return get_args(arg)[0], param
                elif isinstance(arg, TypeVar):
                    return get_args(arg.__bound__)[0], arg

    def requests_alpha(self):
        return "alpha" in inspect.getfullargspec(self.__f)[0]

    def requests_beta(self):
        return "beta" in inspect.getfullargspec(self.__f)[0]

    def requests_model_c(self):
        return "c" in inspect.getfullargspec(self.__f)[0]

    def requests_cache(self):
        return "cache" in inspect.getfullargspec(self.__f)[0]


class MergeMethodRepository:
    def __init__(self):
        self.__methods = {}

    def register(
        self,
        f: Optional[Callable] = None, *,
        name: Optional[str] = None,
    ) -> Callable:
        if f is None:
            return lambda f: self.__register_impl(f, name=name)
        return self.__register_impl(f, name=name)

    def __register_impl(
        self,
        f: Callable, *,
        name: Optional[str],
    ):
        if name is None:
            name = f.__name__

        self.__methods[name] = MergeMethod(f)
        return f

    def get(self, name: str) -> Tuple[MergeMethod, MergeSpace]:
        return self.__methods[name]


merge_methods = MergeMethodRepository()
