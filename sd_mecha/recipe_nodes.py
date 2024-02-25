import abc
import pathlib
import torch
from typing import Optional, Dict, Any
from sd_mecha import extensions
from sd_mecha.extensions.model_version import ModelVersion
from sd_mecha.hypers import validate_hyper, Hyper
from sd_mecha.merge_space import MergeSpace


class RecipeNode(abc.ABC):
    model_version: ModelVersion

    @abc.abstractmethod
    def accept(self, visitor, *args, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def merge_space(self) -> MergeSpace:
        pass

    @abc.abstractmethod
    def __contains__(self, item):
        pass


class ModelRecipeNode(RecipeNode):
    def __init__(
        self,
        state_dict_path: str | pathlib.Path,
        model_version: str = "sd1",
        model_type: str = "base",
    ):
        self.path = state_dict_path
        self.state_dict = None
        self.model_type = extensions.model_type.resolve(model_type)
        self.model_version = extensions.model_version.resolve(model_version)

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_model(self, *args, **kwargs)

    @property
    def merge_space(self) -> MergeSpace:
        return self.model_type.merge_space

    def __contains__(self, item):
        if isinstance(item, ModelRecipeNode):
            return self.path == item.path
        else:
            return False


class ParameterRecipeNode(RecipeNode):
    def __init__(self, name: str, merge_space: MergeSpace = MergeSpace.BASE):
        self.name = name
        self.__merge_space = merge_space

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_parameter(self, *args, **kwargs)

    @property
    def merge_space(self) -> MergeSpace:
        return self.__merge_space

    def __contains__(self, item):
        if isinstance(item, ParameterRecipeNode):
            return self.name == item.name and self.merge_space == item.merge_space
        else:
            return False


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
        self.model_version = self.models[0].model_version
        for hyper_v in hypers.values():
            validate_hyper(hyper_v, self.model_version)
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

    def __contains__(self, item):
        if isinstance(item, MergeRecipeNode):
            return self is item or any(
                item in model
                for model in self.models
            )
        else:
            return False


class RecipeVisitor(abc.ABC):
    @abc.abstractmethod
    def visit_model(self, node: ModelRecipeNode):
        pass

    @abc.abstractmethod
    def visit_parameter(self, node: ParameterRecipeNode):
        pass

    @abc.abstractmethod
    def visit_merge(self, node: MergeRecipeNode):
        pass


class DepthRecipeVisitor(RecipeVisitor):
    def visit_model(self, _node: ModelRecipeNode):
        return 1

    def visit_parameter(self, _node: ParameterRecipeNode):
        return 0

    def visit_merge(self, node: MergeRecipeNode):
        return max(
            model.accept(self)
            for model in node.models
        ) + 1


class ModelsCountVisitor(RecipeVisitor):
    def __init__(self):
        self.__seen_nodes = []

    def visit_model(self, node: ModelRecipeNode):
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


class ParameterResolverVisitor(RecipeVisitor):
    def __init__(self, arguments: Dict[str, RecipeNode]):
        self.__arguments = arguments

    def visit_model(self, node: ModelRecipeNode) -> RecipeNode:
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
