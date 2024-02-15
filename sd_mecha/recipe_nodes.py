import abc
import pathlib

from tensordict import TensorDict
import torch
from typing import Optional, List
from sd_mecha.sd_meh.extensions import MergeSpace
from sd_mecha.sd_meh.streaming import InSafetensorDict


class RecipeNode(abc.ABC):
    @abc.abstractmethod
    def visit(self, key: str, scheduler) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def depth(self) -> int:
        pass

    @abc.abstractmethod
    def get_arbitrary_input_model(self, scheduler) -> InSafetensorDict:
        pass

    @property
    @abc.abstractmethod
    def merge_space(self) -> MergeSpace:
        pass


class LeafRecipeNode(RecipeNode):
    def __init__(
        self, state_dict: str | pathlib.Path | TensorDict,
    ):
        self.__state_dict = state_dict

    def visit(self, key: str, scheduler) -> torch.Tensor:
        self.__state_dict = scheduler.load_state_dict(self.__state_dict)
        return self.__state_dict[key]

    def depth(self) -> int:
        return 1

    def get_arbitrary_input_model(self, scheduler) -> InSafetensorDict:
        self.__state_dict = scheduler.load_state_dict(self.__state_dict)
        return self.__state_dict

    @property
    def merge_space(self) -> MergeSpace:
        return MergeSpace.MODEL


class SymbolicRecipeNode(RecipeNode):
    def __init__(
        self,
        merge_method,
        a: RecipeNode,
        b: RecipeNode,
        alpha: float,
        c: RecipeNode = None,
        beta: Optional[float] = None,
        rebasin_iters: int = 0,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = "cpu",
        work_device: Optional[str] = None,
        work_dtype: Optional[torch.dtype] = None,
    ):
        self.__merge_method = merge_method
        self.__a = a
        self.__b = b
        self.__c = c
        self.__alpha = alpha
        self.__beta = beta
        self.__rebasin_iters = rebasin_iters
        self.__device = device
        self.__dtype = dtype
        self.__work_device = work_device
        self.__work_dtype = work_dtype

    def visit(self, key: str, scheduler):
        return scheduler.symbolic_merge(
            key,
            self.__merge_method,
            self._models_dict(*visit_deeper_first([self.__a, self.__b, self.__c], key, scheduler)),
            self.__alpha, self.__beta,
            self.__device, self.__dtype,
            self.__work_device, self.__work_dtype,
        )

    @staticmethod
    def _models_dict(a, b, c=None) -> dict:
        models = {
            "a": a,
            "b": b,
        }
        if c is not None:
            models["c"] = c
        return models

    def depth(self) -> int:
        return max(
            model.depth()
            for model in (self.__a, self.__b, self.__c)
            if model is not None
        ) + 1

    def get_arbitrary_input_model(self, scheduler) -> InSafetensorDict:
        return self.__a.get_arbitrary_input_model(scheduler)

    @property
    def merge_space(self) -> MergeSpace:
        assert self.__a.merge_space == self.__b.merge_space == self.__c.merge_space
        return MergeSpace.MODEL


class ClipRecipeNode(RecipeNode):
    def __init__(self, model, a, b):
        self.__model = model
        self.__a = a
        self.__b = b

    def visit(self, key: str, scheduler):
        return scheduler.clip_weights(*visit_deeper_first([self.__model, self.__a, self.__b], key, scheduler))

    def depth(self) -> int:
        return max(
            model.depth()
            for model in (self.__model, self.__a, self.__b)
            if model is not None
        ) + 1


class CombineRecipeNode(RecipeNode):
    def __init__(self, a: RecipeNode, b: RecipeNode):
        self.__a = a
        self.__b = b

    def visit(self, key: str, scheduler):
        dicts = visit_deeper_first([self.__a, self.__b], key, scheduler)
        for k, v in dicts[0].items():
            dicts[1][k] = v

        return dicts[1]

    def depth(self) -> int:
        return max(model.depth() for model in (self.__a, self.__b)) + 1


def visit_deeper_first(nodes: List[RecipeNode], key: str, scheduler) -> list:
    merged: List[Optional[torch.Tensor]] = [None] * len(nodes)

    def depth_of_value(index) -> int:
        if nodes[index] is None:
            return 0
        return nodes[index].depth()

    for index in sorted(range(len(nodes)), key=depth_of_value, reverse=True):
        if nodes[index] is None:
            continue
        merged[index] = nodes[index].visit(key, scheduler)

    return merged
