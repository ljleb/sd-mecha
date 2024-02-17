import abc
import enum
import pathlib
import torch
from typing import Optional
from sd_mecha.streaming import InModelSafetensorsDict, InLoraSafetensorsDict
from sd_mecha.hypers import validate_hyper, Hyper


class MergeSpace(enum.Flag):
    MODEL = enum.auto()
    DELTA = enum.auto()


class RecipeNode(abc.ABC):
    @abc.abstractmethod
    def accept(self, visitor, *args, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def merge_space(self) -> MergeSpace:
        pass


class LeafRecipeNode(RecipeNode, abc.ABC):
    def __init__(
        self,
        state_dict,
        dict_class,
    ):
        if isinstance(state_dict, dict_class):
            self.path = state_dict.file_path
            self.state_dict = state_dict
        else:
            self.path = state_dict
            self.state_dict = None


class ModelRecipeNode(LeafRecipeNode):
    def __init__(
        self,
        state_dict: str | pathlib.Path | InModelSafetensorsDict,
    ):
        super().__init__(state_dict, InModelSafetensorsDict)

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_model(self, *args, **kwargs)

    @property
    def merge_space(self) -> MergeSpace:
        return MergeSpace.MODEL


class LoraRecipeNode(LeafRecipeNode):
    def __init__(
        self,
        state_dict: str | pathlib.Path | InLoraSafetensorsDict,
    ):
        super().__init__(state_dict, InLoraSafetensorsDict)

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_lora(self, *args, **kwargs)

    @property
    def merge_space(self) -> MergeSpace:
        return MergeSpace.DELTA


class MergeRecipeNode(RecipeNode):
    def __init__(
        self,
        merge_method,
        *models: RecipeNode,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        **hypers: Hyper
    ):
        self.merge_method = merge_method
        self.models = models
        self.hypers = {k: validate_hyper(v) for k, v in hypers.items()}
        self.device = device
        self.dtype = dtype
        self.__merge_space = self.merge_method.get_return_merge_space([
            model.merge_space
            for model in self.models
        ])

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_merge(self, *args, **kwargs)

    @property
    def merge_space(self) -> MergeSpace:
        return self.__merge_space


class DepthRecipeVisitor:
    def visit_model(self, _node: ModelRecipeNode):
        return 1

    def visit_lora(self, _node: LoraRecipeNode):
        return 1

    def visit_merge(self, node: MergeRecipeNode):
        return max(
            model.accept(self)
            for model in node.models
        ) + 1