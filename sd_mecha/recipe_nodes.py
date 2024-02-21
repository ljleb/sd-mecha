import abc
import dataclasses
import enum
import pathlib
import torch
from typing import Optional, Dict, Any
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


class ParameterRecipeNode(RecipeNode):
    def __init__(self, name: str, merge_space: MergeSpace = MergeSpace.MODEL):
        self.name = name
        self.__merge_space = merge_space

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_parameter(self, *args, **kwargs)

    @property
    def merge_space(self) -> MergeSpace:
        return self.__merge_space


class MergeRecipeNode(RecipeNode):
    def __init__(
        self,
        merge_method,
        *models: RecipeNode,
        hypers: Dict[str, Hyper],
        volatile_hypers: Dict[str, Any],
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.merge_method = merge_method
        self.models = models
        for hyper_v in hypers.values():
            validate_hyper(hyper_v)
        self.hypers = hypers
        self.volatile_hypers = volatile_hypers
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

    def visit_parameter(self, _node: LoraRecipeNode):
        return 0

    def visit_merge(self, node: MergeRecipeNode):
        return max(
            model.accept(self)
            for model in node.models
        ) + 1


class ModelsCountVisitor:
    def __init__(self):
        self.__seen_nodes = []

    def visit_model(self, node: ModelRecipeNode):
        seen = node in self.__seen_nodes
        self.__seen_nodes.append(node)
        return not seen

    def visit_lora(self, node: LoraRecipeNode):
        seen = node in self.__seen_nodes
        self.__seen_nodes.append(node)
        return not seen

    def visit_parameter(self, _node: ParameterRecipeNode):
        return 0

    def visit_merge(self, node: MergeRecipeNode):
        return sum(
            model.accept(self)
            for model in node.models
        )


class ParameterResolverVisitor:
    def __init__(self, arguments: Dict[str, RecipeNode]):
        self.__arguments = arguments

    def visit_model(self, node: ModelRecipeNode) -> RecipeNode:
        return node

    def visit_lora(self, node: LoraRecipeNode) -> RecipeNode:
        return node

    def visit_parameter(self, node: ParameterRecipeNode) -> RecipeNode:
        return self.__arguments.get(node.name, node)

    def visit_merge(self, node: MergeRecipeNode) -> RecipeNode:
        return MergeRecipeNode(
            node.merge_method,
            *(node.accept(self) for node in node.models),
            hypers=node.hypers,
            volatile_hypers=node.volatile_hypers,
            device=node.device,
            dtype=node.dtype,
        )
