import abc
import pathlib

from tensordict import TensorDict
import torch
from typing import Optional


class MergeNode(abc.ABC):
    @abc.abstractmethod
    def visit(self, scheduler):
        pass

    @abc.abstractmethod
    def depth(self) -> int:
        pass


class LeafMergeNode(MergeNode):
    def __init__(self, state_dict: str | pathlib.Path | TensorDict, device=None):
        self.__state_dict = state_dict
        self.__device = device

    def visit(self, scheduler):
        return scheduler.load_state_dict(self.__state_dict, self.__device)

    def depth(self) -> int:
        return 1


class SymbolicMergeNode(MergeNode):
    def __init__(
        self,
        merge_method,
        a,
        b,
        alpha: float,
        c=None,
        beta: Optional[float] = None,
        rebasin_iters: Optional[int] = None,
        threads: int = 1,
        device: str = "cpu",
        work_device: Optional[str] = None,
        work_dtype: Optional[torch.dtype] = None,
        weights_clip: bool = False,
    ):
        self.__merge_method = merge_method
        self.__a = a
        self.__b = b
        self.__c = c
        self.__alpha = alpha
        self.__beta = beta
        self.__rebasin_iters = rebasin_iters
        self.__threads = threads
        self.__device = device
        self.__work_device = work_device
        self.__work_dtype = work_dtype
        self.__weights_clip = weights_clip

    def visit(self, scheduler):
        return scheduler.symbolic_merge(
            self.__merge_method,
            *visit_deeper_first([self.__a, self.__b, self.__c], scheduler),
            self.__alpha, self.__beta,
            self.__rebasin_iters,
            self.__device, self.__work_device, self.__work_dtype,
            self.__threads,
            self.__weights_clip,
        )

    def depth(self) -> int:
        return max(
            model.depth()
            for model in (self.__a, self.__b, self.__c)
            if model is not None
        ) + 1


class ClipMergeNode(MergeNode):
    def __init__(self, model, a, b):
        self.__model = model
        self.__a = a
        self.__b = b

    def visit(self, scheduler):
        return scheduler.clip_weights(*visit_deeper_first([self.__model, self.__a, self.__b], scheduler))

    def depth(self) -> int:
        return max(
            model.depth()
            for model in (self.__model, self.__a, self.__b)
            if model is not None
        ) + 1


def visit_deeper_first(nodes: list, scheduler) -> list:
    models = [None]*len(nodes)

    def depth_of_value(index) -> int:
        if nodes[index] is None:
            return 0
        return nodes[index].depth()

    for index in sorted(range(len(nodes)), key=depth_of_value, reverse=True):
        if nodes[index] is None:
            continue

        models[index] = nodes[index].visit(scheduler)

    return models
