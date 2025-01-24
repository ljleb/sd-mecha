import abc
import itertools
import pathlib
from typing import Optional, Dict, Tuple
import torch
from . import extensions
from .extensions.model_config import ModelConfig


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

    def set_cache(self, _cache: dict = ...):
        return self

    def __add__(self, other: "RecipeNodeOrValue") -> "RecipeNode":
        other = extensions.merge_method.value_to_node(other)
        base, delta = self, other
        if other.merge_space == "weight":
            base, delta = other, self
        return extensions.merge_method.resolve("add_difference")(base, delta)

    def __radd__(self, other: "RecipeNodeOrValue") -> "RecipeNode":
        other = extensions.merge_method.value_to_node(other)
        return other + self

    def __sub__(self, other: "RecipeNodeOrValue") -> "RecipeNode":
        other = extensions.merge_method.value_to_node(other)
        return extensions.merge_method.resolve("subtract")(self, other)

    def __rsub__(self, other: "RecipeNodeOrValue") -> "RecipeNode":
        other = extensions.merge_method.value_to_node(other)
        return other - self

    def __or__(self, other: "RecipeNodeOrValue") -> "RecipeNode":
        other = extensions.merge_method.value_to_node(other)
        return extensions.merge_method.resolve("fallback")(self, other)

    def __ror__(self, other: "RecipeNodeOrValue") -> "RecipeNode":
        other = extensions.merge_method.value_to_node(other)
        return other | self

    def to(self, *, device: Optional[str | torch.device] = None, dtype: Optional[torch.dtype] = None):
        if device is not None:
            device = str(device)
        if dtype is not None:
            from sd_mecha.merge_methods import cast_dtype_map_reversed
            dtype = cast_dtype_map_reversed[dtype]
        return extensions.merge_method.resolve("cast").create_recipe(self, device, dtype)


NonDictLiteralValue = str | int | float | bool
LiteralValue = NonDictLiteralValue | dict
RecipeNodeOrValue = RecipeNode | LiteralValue | pathlib.Path


class LiteralRecipeNode(RecipeNode):
    def __init__(
        self,
        value: LiteralValue,
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
        path: pathlib.Path,
        model_config: Optional[str | ModelConfig] = None,
        merge_space: str = "weight",
    ):
        if not isinstance(path, pathlib.Path):
            raise TypeError(f"The type of 'state_dict' must be Path, not {type(path).__name__}")

        self.path = path
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
        self.cache = None
        self.__validate_args()

    def __validate_args(self):
        if self.merge_method.get_return_merge_space(
            [arg.merge_space for arg in self.args],
            {k: v.merge_space for k, v in self.kwargs.items()}
        ) is None:
            raise RuntimeError(f"Could not infer merge space from arguments for method {self.merge_method.identifier}")

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

    def set_cache(self, cache: dict = ...):
        if cache is Ellipsis:
            cache = {}

        self.cache = cache
        return self


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

    def visit_literal(self, node: LiteralRecipeNode) -> int:
        return 0

    def visit_model(self, node: ModelRecipeNode) -> int:
        seen = node in self.__seen_nodes
        self.__seen_nodes.append(node)
        return int(not seen)

    def visit_merge(self, node: MergeRecipeNode) -> int:
        return sum(
            child.accept(self)
            for children in (node.args, node.kwargs.values())
            for child in children
        )
