import abc
import pathlib
import torch
from collections import OrderedDict
from typing import Optional, Dict, Any, Mapping
from torch import Tensor
from sd_mecha import extensions
from sd_mecha.extensions.model_config import ModelConfig
from sd_mecha.hypers import Hyper
from sd_mecha.streaming import TensorMetadata, MemoryDict


class RecipeNode(abc.ABC):
    @abc.abstractmethod
    def accept(self, visitor, *args, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def merge_space(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def model_config(self) -> Optional[ModelConfig]:
        pass

    @abc.abstractmethod
    def compute_keys(self) -> OrderedDict[str, TensorMetadata]:
        pass

    @abc.abstractmethod
    def __contains__(self, item):
        pass


class ModelRecipeNode(RecipeNode):
    def __init__(
        self,
        state_dict: str | pathlib.Path | Mapping[str, Tensor],
        model_config: Optional[str | ModelConfig] = None,
    ):
        if isinstance(state_dict, Mapping):
            self.path = "<memory>"
            self.state_dict = MemoryDict(state_dict)
        else:
            self.path = state_dict
            self.state_dict = None
        self.__model_config = model_config

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_model(self, *args, **kwargs)

    @property
    def merge_space(self) -> str:
        return "weight"

    @property
    def model_config(self) -> Optional[ModelConfig]:
        if isinstance(self.__model_config, str):
            return extensions.model_config.resolve(self.__model_config)
        return self.__model_config

    @model_config.setter
    def model_config(self, value: Optional[ModelConfig]):
        self.__model_config = value

    def compute_keys(self) -> OrderedDict[str, TensorMetadata]:
        all_keys = self.model_config.compute_keys()
        return OrderedDict(
            (k, v)
            for k, v in self.state_dict.metadata()
            if k in all_keys
        )

    def __contains__(self, item):
        if isinstance(item, ModelRecipeNode):
            return self.path == item.path
        else:
            return False


class MergeRecipeNode(RecipeNode):
    def __init__(
        self,
        merge_method,
        *inputs: RecipeNode,
        hypers: Dict[str, Hyper],
        volatile_hypers: Dict[str, Any],
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.merge_method = merge_method
        self.inputs = inputs
        self.hypers = hypers
        self.volatile_hypers = volatile_hypers
        self.device = device
        self.dtype = dtype

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_merge(self, *args, **kwargs)

    @property
    def merge_space(self) -> str:
        return self.merge_method.get_return_merge_space([
            input.merge_space for input in self.inputs
        ])

    @property
    def model_config(self) -> Optional[ModelConfig]:
        return self.merge_method.get_return_config([
            input.model_config for input in self.inputs
        ])

    def compute_keys(self) -> OrderedDict[str, TensorMetadata]:
        res = OrderedDict()
        for model in self.inputs:
            for k, v in model.compute_keys().items():
                if k not in res:
                    res[k] = v
                elif res[k].shape != v.shape:
                    raise RuntimeError(f"key {k} has ambiguous shape. candidates are {res[k].shape} and {v.shape}")
        return OrderedDict(self.model_config.compute_keys())

    def __contains__(self, item):
        if isinstance(item, MergeRecipeNode):
            return self is item or any(
                item in model
                for model in self.inputs
            )
        else:
            return False


class RecipeVisitor(abc.ABC):
    @abc.abstractmethod
    def visit_model(self, node: ModelRecipeNode):
        pass

    @abc.abstractmethod
    def visit_merge(self, node: MergeRecipeNode):
        pass


class DepthRecipeVisitor(RecipeVisitor):
    def visit_model(self, _node: ModelRecipeNode):
        return 1

    def visit_merge(self, node: MergeRecipeNode):
        return max(
            model.accept(self)
            for model in node.inputs
        ) + 1


class ModelsCountVisitor(RecipeVisitor):
    def __init__(self):
        self.__seen_nodes = []

    def visit_model(self, node: ModelRecipeNode):
        seen = node in self.__seen_nodes
        self.__seen_nodes.append(node)
        return int(not seen)

    def visit_merge(self, node: MergeRecipeNode):
        return sum(
            model.accept(self)
            for model in node.inputs
        )


class ParameterResolverVisitor(RecipeVisitor):
    def __init__(self, arguments: Dict[str, RecipeNode]):
        self.__arguments = arguments

    def visit_model(self, node: ModelRecipeNode) -> RecipeNode:
        return node

    def visit_merge(self, node: MergeRecipeNode) -> RecipeNode:
        return MergeRecipeNode(
            node.merge_method,
            *(node.accept(self) for node in node.inputs),
            hypers=node.hypers,
            volatile_hypers=node.volatile_hypers,
            device=node.device,
            dtype=node.dtype,
        )
