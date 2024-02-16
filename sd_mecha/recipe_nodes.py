import abc
import enum
import pathlib
import torch
from typing import Optional, List, Mapping
from sd_mecha.streaming import InModelSafetensorsDict, InLoraSafetensorsDict
from sd_mecha.weight import get_weight, validate_model_parameter


class MergeSpace(enum.Flag):
    MODEL = enum.auto()
    DELTA = enum.auto()


class RecipeNode(abc.ABC):
    @abc.abstractmethod
    def visit(self, key: str, scheduler) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def depth(self) -> int:
        pass

    @abc.abstractmethod
    def get_input_dicts(self, scheduler) -> List[Mapping[str, torch.Tensor]]:
        pass

    @property
    @abc.abstractmethod
    def merge_space(self) -> MergeSpace:
        pass


class ModelRecipeNode(RecipeNode):
    def __init__(
        self,
        state_dict: str | pathlib.Path | InModelSafetensorsDict,
    ):
        self.__state_dict = state_dict

    def visit(self, key: str, scheduler) -> torch.Tensor:
        self.__state_dict = scheduler.load_model(self.__state_dict)
        return self.__state_dict[key]

    def depth(self) -> int:
        return 1

    def get_input_dicts(self, scheduler) -> List[Mapping[str, torch.Tensor]]:
        self.__state_dict = scheduler.load_model(self.__state_dict)
        return [self.__state_dict]

    @property
    def merge_space(self) -> MergeSpace:
        return MergeSpace.MODEL


class LoraRecipeNode(RecipeNode):
    def __init__(
        self,
        state_dict: str | pathlib.Path | InLoraSafetensorsDict,
    ):
        self.__state_dict = state_dict

    def visit(self, key: str, scheduler) -> torch.Tensor:
        self.__state_dict = scheduler.load_lora(self.__state_dict)
        return self.__state_dict[key]

    def depth(self) -> int:
        return 1

    def get_input_dicts(self, scheduler) -> List[Mapping[str, torch.Tensor]]:
        self.__state_dict = scheduler.load_lora(self.__state_dict)
        return [self.__state_dict]

    @property
    def merge_space(self) -> MergeSpace:
        return MergeSpace.DELTA


class SymbolicRecipeNode(RecipeNode):
    def __init__(
        self,
        merge_method,
        a: RecipeNode,
        b: RecipeNode,
        c: RecipeNode = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.__merge_method = merge_method
        self.__a = a
        self.__b = b
        self.__c = c
        self.__alpha = validate_model_parameter(alpha) if alpha is not None else None
        self.__beta = validate_model_parameter(beta) if beta is not None else None
        self.__device = device
        self.__dtype = dtype
        self.__merge_space = self.__merge_method.get_return_merge_space(
            self.__a.merge_space,
            self.__b.merge_space,
            self.__c.merge_space if self.__c is not None else None,
        )

    def visit(self, key: str, scheduler):
        return scheduler.symbolic_merge(
            key,
            self.__merge_method,
            self._models_dict(*visit_deeper_first([self.__a, self.__b, self.__c], key, scheduler)),
            get_weight(self.__alpha, key) if self.__alpha is not None else None,
            get_weight(self.__beta, key) if self.__beta is not None else None,
            self.__device, self.__dtype,
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

    def get_input_dicts(self, scheduler) -> List[Mapping[str, torch.Tensor]]:
        return [
            input_dict
            for node in (self.__a, self.__b, self.__c)
            if node is not None
            for input_dict in node.get_input_dicts(scheduler)
        ]

    @property
    def merge_space(self) -> MergeSpace:
        return self.__merge_space


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
