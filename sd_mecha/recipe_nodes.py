import abc
import enum
import pathlib
import torch
from typing import Optional, List, Mapping, Tuple
from sd_mecha.streaming import InModelSafetensorsDict, InLoraSafetensorsDict
from sd_mecha.weight import get_weight, validate_hyper, Hyper


class MergeSpace(enum.Flag):
    MODEL = enum.auto()
    DELTA = enum.auto()


class RecipeNode(abc.ABC):
    @abc.abstractmethod
    def visit(self, key: str, scheduler) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def serialize(self, instructions: List[str]) -> int:
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

    def serialize(self, instructions: List[str]) -> int:
        line = f'model "{self.__state_dict}"'
        try:
            return instructions.index(line)
        except ValueError:
            res = len(instructions)
            instructions.append(line)
            return res

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

    def serialize(self, instructions: List[str]) -> int:
        line = f'model "{self.__state_dict}"'
        try:
            return instructions.index(line)
        except ValueError:
            res = len(instructions)
            instructions.append(line)
            return res

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
        *models: RecipeNode,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        **hypers: Hyper
    ):
        self.__merge_method = merge_method
        self.__models = models
        self.__hypers = {k: validate_hyper(v) for k, v in hypers.items()}
        self.__device = device
        self.__dtype = dtype
        self.__merge_space = self.__merge_method.get_return_merge_space([
            model.merge_space
            for model in self.__models
        ])

    def visit(self, key: str, scheduler):
        return scheduler.symbolic_merge(
            self.__merge_method,
            visit_deeper_first(self.__models, key, scheduler),
            {k: get_weight(v, key) for k, v in self.__hypers.items()},
            self.__device, self.__dtype,
        )

    def serialize(self, instructions: List[str]) -> int:
        models = []
        for model in self.__models:
            models.append(f"&{model.serialize(instructions)}")

        hypers = []
        for hyper_k, hyper_v in self.__hypers.items():
            if isinstance(hyper_v, dict):
                line = "dict " + " ".join(f"{k}={v}" for k, v in hyper_v.items())
                try:
                    hyper_v = instructions.index(line)
                except ValueError:
                    hyper_v = f"&{len(instructions)}"
                    instructions.append(line)
            hypers.append(f"{hyper_k}={hyper_v}")

        models = " ".join(models)
        hypers = " ".join(hypers)

        line = f'call "{self.__merge_method.get_name()}" {models} {hypers}'
        try:
            return instructions.index(line)
        except ValueError:
            res = len(instructions)
            instructions.append(line)
            return res

    def depth(self) -> int:
        return max(
            model.depth()
            for model in self.__models
        ) + 1

    def get_input_dicts(self, scheduler) -> List[Mapping[str, torch.Tensor]]:
        return [
            input_dict
            for model in self.__models
            for input_dict in model.get_input_dicts(scheduler)
        ]

    @property
    def merge_space(self) -> MergeSpace:
        return self.__merge_space


def visit_deeper_first(nodes: Tuple[RecipeNode, ...], key: str, scheduler) -> list:
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
