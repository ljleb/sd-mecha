import enum
import inspect
import torch
from typing import Optional, Callable, Set, Dict, Tuple


class MergeSpace(enum.Enum):
    MODEL = enum.auto()
    DELTA = enum.auto()


class MergeMethod:
    def __init__(self, f: Callable):
        self.__f = f

    def __call__(self, inputs, hyper_parameters, device, work_device, dtype, work_dtype, cache):
        kwargs = self._get_kwargs(inputs, hyper_parameters, work_device, work_dtype, cache)

        # pix2pix and inpainting models
        # todo: verify whether we want to slice, merge and then concat this key instead of ignore it
        if (a_size := kwargs["a"].size()) != (b_size := kwargs["b"].size()):
            if a_size[1] > b_size[1]:
                return kwargs["a"]
            else:
                return kwargs["b"]

        if dtype is None:
            to_args = device,
        else:
            to_args = device, dtype
        return self.__f(**kwargs).to(*to_args)

    def _get_kwargs(
        self,
        inputs: Dict[str, torch.Tensor],
        hyper_parameters: Dict[str, float],
        work_device: str,
        work_dtype: Optional[torch.dtype],
        cache: Optional[Dict],
    ) -> Dict:
        if self.requests_model_c() and "c" not in inputs:
            raise ValueError

        if work_dtype is None:
            to_args = work_device,
        else:
            to_args = work_device, work_dtype

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
        merge_spaces: MergeSpace | Set[MergeSpace] = MergeSpace.MODEL,
    ) -> Callable:
        if f is None:
            return lambda f: self.__register_impl(f, name=name, merge_spaces=merge_spaces)
        return self.__register_impl(f, name=name, merge_spaces=merge_spaces)

    def __register_impl(
        self,
        f: Callable, *,
        name: Optional[str],
        merge_spaces: MergeSpace | Set[MergeSpace],
    ):
        if name is None:
            name = f.__name__
        if not isinstance(merge_spaces, set):
            merge_spaces = {merge_spaces}

        self.__methods[name] = (MergeMethod(f), merge_spaces)
        return f

    def get(self, name: str) -> Tuple[MergeMethod, MergeSpace]:
        return self.__methods[name]


merge_methods = MergeMethodRepository()
