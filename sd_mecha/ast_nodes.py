import abc
import pathlib
from typing import Optional

import sd_meh.merge


class MergeNode(abc.ABC):
    @abc.abstractmethod
    def visit(self, scheduler):
        pass


class LeafMergeNode(MergeNode):
    def __init__(self, state_dict, device=None):
        self.__state_dict = state_dict
        self.__device = device

    def visit(self, scheduler):
        return scheduler.load_state_dict(self.__state_dict, self.__device)


class SymbolicMergeNode(MergeNode):
    def __init__(
        self,
        merge_method,
        a,
        b,
        alpha,
        c=None,
        beta=None,
        rebasin_iters=None,
        precision: int = 16,
        prune: bool = False,
        threads: int = 1,
        device: str = "cpu",
        work_device: Optional[str] = None,
    ):
        self.__merge_method = merge_method
        self.__a = a
        self.__b = b
        self.__c = c
        self.__alpha = alpha
        self.__beta = beta
        self.__rebasin_iters = rebasin_iters
        self.__precision = precision
        self.__prune = prune
        self.__threads = threads
        self.__device = device
        self.__work_device = work_device

    def visit(self, scheduler):
        return scheduler.symbolic_merge(
            self.__merge_method,
            self.__a.visit(scheduler),
            self.__b.visit(scheduler),
            self.__c.visit(scheduler) if self.__c is not None else None,
            self.__alpha, self.__beta,
            self.__rebasin_iters,
            self.__device, self.__work_device,
            self.__prune,
            self.__threads,
        )


class ClipMergeNode(MergeNode):
    def __init__(self, model, a, b, device=None):
        self.__model = model
        self.__a = a
        self.__b = b
        self.__device = device

    def visit(self, scheduler):
        return scheduler.clip_weights(self.__model, self.__a, self.__b, self.__device)
