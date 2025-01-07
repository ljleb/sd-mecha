import abc
import itertools
import pathlib
from typing import Optional, Dict, Mapping, Tuple
from torch import Tensor
from . import extensions
from .extensions.model_config import ModelConfig
from .streaming import MemoryDict


class RecipeNode(abc.ABC):
    @abc.abstractmethod
    def accept(self, visitor, *args, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def merge_space(self) -> extensions.merge_space.MergeSpace:
        pass

    @property
    @abc.abstractmethod
    def model_config(self) -> Optional[ModelConfig]:
        pass

    @abc.abstractmethod
    def __contains__(self, item):
        pass

    def __add__(self, other):
        other = extensions.merge_method.value_to_node(other)
        base, delta = self, other
        if other.merge_space == "weight":
            base, delta = other, self
        return extensions.merge_method.resolve("add_difference")(base, delta)

    def __radd__(self, other):
        other = extensions.merge_method.value_to_node(other)
        return other + self

    def __sub__(self, other):
        other = extensions.merge_method.value_to_node(other)
        return extensions.merge_method.resolve("subtract")(self, other)

    def __rsub__(self, other):
        other = extensions.merge_method.value_to_node(other)
        return other - self


class LiteralRecipeNode(RecipeNode):
    def __init__(
        self,
        value: str | int | float | Dict[str, str | int | float],
        model_config: Optional[str | ModelConfig] = None,
    ):
        self.value = value
        self.__model_config = model_config

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_literal(self, *args, **kwargs)

    @property
    def merge_space(self) -> extensions.merge_space.MergeSpace:
        return extensions.merge_space.resolve("param")

    @property
    def model_config(self) -> Optional[ModelConfig]:
        if isinstance(self.__model_config, str):
            return extensions.model_config.resolve(self.__model_config)
        return self.__model_config

    @model_config.setter
    def model_config(self, model_config: Optional[ModelConfig]):
        self.__model_config = model_config

    def __contains__(self, item):
        if isinstance(item, LiteralRecipeNode):
            return self.value == item.value
        else:
            return False


class ModelRecipeNode(RecipeNode):
    def __init__(
        self,
        state_dict: str | pathlib.Path | Mapping[str, Tensor],
        model_config: Optional[str | ModelConfig] = None,
        merge_space: str = "weight",
    ):
        if isinstance(state_dict, Mapping):
            self.path = "<memory>"
            self.state_dict = MemoryDict(state_dict)
        else:
            self.path = state_dict
            self.state_dict = None
        self.__model_config = model_config
        self.__merge_space = merge_space

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_model(self, *args, **kwargs)

    @property
    def merge_space(self) -> extensions.merge_space.MergeSpace:
        return extensions.merge_space.resolve(self.__merge_space)

    @property
    def model_config(self) -> Optional[ModelConfig]:
        if isinstance(self.__model_config, str):
            return extensions.model_config.resolve(self.__model_config)
        return self.__model_config

    @model_config.setter
    def model_config(self, value: Optional[ModelConfig]):
        self.__model_config = value

    def __contains__(self, item):
        if isinstance(item, ModelRecipeNode):
            return self.path == item.path
        else:
            return False


class MergeRecipeNode(RecipeNode):
    def __init__(
        self,
        merge_method,
        args: Tuple[RecipeNode, ...],
        kwargs: Dict[str, RecipeNode],
    ):
        self.merge_method = merge_method
        self.args = args
        self.kwargs = kwargs

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_merge(self, *args, **kwargs)

    @property
    def merge_space(self) -> extensions.merge_space.MergeSpace:
        return self.merge_method.get_return_merge_space(
            [v.merge_space for v in self.args],
            {k: v.merge_space for k, v in self.kwargs.items()},
        )

    @property
    def model_config(self) -> Optional[ModelConfig]:
        return self.merge_method.get_return_config(
            [v.model_config for v in self.args],
            {k: v.model_config for k, v in self.kwargs.items()},
        )

    def __contains__(self, item):
        return self is item or any(
            item in v
            for v in itertools.chain(self.args, self.kwargs.values())
            if isinstance(v, RecipeNode)
        )


class RecipeVisitor(abc.ABC):
    @abc.abstractmethod
    def visit_literal(self, node: LiteralRecipeNode):
        pass

    @abc.abstractmethod
    def visit_model(self, node: ModelRecipeNode):
        pass

    @abc.abstractmethod
    def visit_merge(self, node: MergeRecipeNode):
        pass


class ModelDepthRecipeVisitor(RecipeVisitor):
    def visit_literal(self, node: LiteralRecipeNode):
        return 0

    def visit_model(self, _node: ModelRecipeNode):
        return 1

    def visit_merge(self, node: MergeRecipeNode):
        return max(
            child.accept(self)
            for children in (node.args, node.kwargs.values())
            for child in children
        ) + 1


class ModelsCountVisitor(RecipeVisitor):
    def __init__(self):
        self.__seen_nodes = []

    def visit_literal(self, node: LiteralRecipeNode):
        return 0

    def visit_model(self, node: ModelRecipeNode):
        seen = node in self.__seen_nodes
        self.__seen_nodes.append(node)
        return int(not seen)

    def visit_merge(self, node: MergeRecipeNode):
        return sum(
            child.accept(self)
            for children in (node.args, node.kwargs.values())
            for child in children
        )
