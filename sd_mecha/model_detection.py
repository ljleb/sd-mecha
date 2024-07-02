import abc
import dataclasses
import functools
import pathlib
import torch
from typing import Iterable, Dict, List, Mapping, Callable, Tuple, Optional
from sd_mecha.extensions.model_arch import ModelArch
from sd_mecha.hypers import get_hyper
from sd_mecha.recipe_nodes import RecipeNode, ModelRecipeNode, ParameterRecipeNode, MergeRecipeNode, DepthRecipeVisitor, RecipeVisitor
from sd_mecha.streaming import InSafetensorsDict


@dataclasses.dataclass
class ModelConfig:
    __minimal_dummy_header: Dict[str, Dict[str, str | List[int]]]
    __input_paths: List[str | pathlib.Path]
    __model_arch: ModelArch

    def __post_init__(self):
        self.__keys_to_forward = set(
            k for k in self.__minimal_dummy_header
            if k in self.__model_arch.keys_to_forward
        )
        self.__keys_to_merge = set(
            k for k in self.__minimal_dummy_header
            if k not in self.__keys_to_forward
            and k in self.__model_arch.keys_to_merge
        )
        self.__minimal_dummy_header = {
            k: v
            for k, v in self.__minimal_dummy_header.items()
            if k in self.__keys_to_forward or k in self.__keys_to_merge
        }

    def keys(self) -> Iterable[str]:
        return self.get_minimal_dummy_header().keys()

    def get_keys_to_merge(self):
        return self.__keys_to_merge

    def get_shape(self, key: str) -> List[int]:
        return self.get_minimal_dummy_header()[key]["shape"]

    def get_minimal_dummy_header(self) -> Dict[str, Dict[str, str | List[int]]]:
        return self.__minimal_dummy_header

    def get_input_paths(self) -> List[str | pathlib.Path]:
        return self.__input_paths

    def intersect(self, other):
        if self.__model_arch is not other.__model_arch:
            self_paths = ', '.join(str(p) for p in self.__input_paths)
            other_paths = ', '.join(str(p) for p in other.__input_paths)
            raise ValueError(
                "Found incompatible model architectures: "
                f"{len(self.__input_paths)} {self.__model_arch} models ({self_paths}) and "
                f"{len(other.__input_paths)} {other.__model_arch} models ({other_paths})"
            )

        return ModelConfig(
            self.__minimal_dummy_header | other.__minimal_dummy_header,
            self.__input_paths + other.__input_paths,
            self.__model_arch,
        )

    def get_key_merger(
        self,
        key: str,
        recipe: RecipeNode,
        fallback_model: Mapping[str, torch.Tensor],
        device: str,
        dtype: torch.dtype,
    ) -> Callable[[], torch.Tensor]:
        key_merger = KeyMerger(
            recipe,
            fallback_model,
            device,
            dtype,
        )
        if key in self.__keys_to_forward:
            return functools.partial(key_merger.forward_and_save, key)
        else:
            return functools.partial(key_merger.merge_and_save, key)


class DetermineConfigVisitor(RecipeVisitor):
    def visit_model(self, node: ModelRecipeNode) -> ModelConfig:
        state_dict_path = getattr(node.state_dict, "file_path", "<memory>")
        return ModelConfig(
            node.model_type.convert_header(state_dict_path, node.state_dict, node.model_arch),
            [state_dict_path],
            node.model_arch,
        )

    def visit_parameter(self, node: ParameterRecipeNode) -> ModelConfig:
        raise TypeError("Recipe parameters do not have configuration")

    def visit_merge(self, node: MergeRecipeNode) -> ModelConfig:
        configs = [
            model.accept(self)
            for model in node.models
        ]
        if configs:
            return functools.reduce(ModelConfig.intersect, configs)
        raise ValueError("No input models")


@dataclasses.dataclass
class KeyMerger:
    recipe: RecipeNode
    fallback_model: Mapping[str, torch.Tensor]
    default_device: str
    default_dtype: torch.dtype

    def merge_and_save(
        self,
        key: str,
    ) -> torch.Tensor:
        key_merger = KeyMergeVisitor(
            key,
            self.default_device,
            self.default_dtype,
            self.__get_passthrough_tensor,
        )
        return self.recipe.accept(key_merger)

    def forward_and_save(
        self,
        key: str,
    ) -> torch.Tensor:
        return self.__get_passthrough_tensor(key)

    def __get_passthrough_tensor(self, key: str):
        if self.fallback_model is not None and key in self.fallback_model:
            return self.fallback_model[key]

        key_merger = KeyPassthroughVisitor(
            key,
            self.default_device,
            self.default_dtype,
        )
        return self.recipe.accept(key_merger)


@dataclasses.dataclass
class KeyVisitor(RecipeVisitor, abc.ABC):
    _key: str
    _default_device: str
    _default_dtype: torch.dtype

    def visit_model(self, node: ModelRecipeNode) -> torch.Tensor:
        state_dict_path = getattr(node.state_dict, "file_path", "<memory>")
        return node.model_type.get_tensor(state_dict_path, node.state_dict, self._key)

    def visit_parameter(self, node: ParameterRecipeNode) -> torch.Tensor:
        raise NotImplementedError(f"Interactive arguments are not yet implemented. Recipe is abstract: parameter '{node.name}' has no value.")

    @abc.abstractmethod
    def visit_merge(self, node: MergeRecipeNode) -> torch.Tensor:
        pass


@dataclasses.dataclass
class KeyMergeVisitor(KeyVisitor):
    _passthrough_callback: Callable[[str], torch.Tensor]

    def visit_merge(self, node: MergeRecipeNode) -> torch.Tensor:
        try:
            return node.merge_method(
                self.__visit_deeper_first(node.models),
                {k: get_hyper(v, self._key, node.model_arch) for k, v in node.hypers.items()} | node.volatile_hypers,
                self._key,
                node.device if node.device is not None else self._default_device,
                node.dtype if node.dtype is not None else self._default_dtype,
            )
        except KeyError as e:
            return self._passthrough_callback(self._key)

    def __visit_deeper_first(self, nodes: Tuple[RecipeNode, ...]) -> list:
        merged: List[Optional[torch.Tensor]] = [None] * len(nodes)

        def depth_of_value(index) -> int:
            if nodes[index] is None:
                return 0
            return nodes[index].accept(DepthRecipeVisitor())

        for index in sorted(range(len(nodes)), key=depth_of_value, reverse=True):
            if nodes[index] is None:
                continue
            merged[index] = nodes[index].accept(self)

        return merged


@dataclasses.dataclass
class KeyPassthroughVisitor(KeyVisitor):
    def visit_merge(self, node: MergeRecipeNode) -> torch.Tensor:
        for model in node.models:
            try:
                return model.accept(self)
            except KeyError:
                continue

        raise KeyError(f"No model has key '{self._key}'")
