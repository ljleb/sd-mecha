import abc
import dataclasses
import functools
import json
import pathlib
import torch
from typing import Iterable, Dict, List, Mapping, Callable, Tuple, Optional, Set
from sd_mecha.hypers import get_hyper
from sd_mecha.recipe_nodes import RecipeNode, ModelRecipeNode, LoraRecipeNode, ParameterRecipeNode, MergeRecipeNode, DepthRecipeVisitor, RecipeVisitor


@dataclasses.dataclass
class ModelConfig:
    __minimal_dummy_header: Dict[str, Dict[str, str | List[int]]]
    __input_paths: Set[str | pathlib.Path]
    __keys_to_forward: Set[str] = dataclasses.field(init=False, default_factory=set)
    __keys_to_merge: Set[str] = dataclasses.field(init=False, default_factory=set)

    def __post_init__(self):
        self.__keys_to_forward = set(
            k for k in self.__minimal_dummy_header
            if self.__is_passthrough_key(k)
        )
        self.__keys_to_merge = set(
            k for k in self.__minimal_dummy_header
            if k not in self.__keys_to_forward
            and self.__is_merge_key(k)
        )
        self.__minimal_dummy_header = {
            k: v
            for k, v in self.__minimal_dummy_header.items()
            if k in self.__keys_to_forward or k in self.__keys_to_merge
        }

    def keys(self) -> Iterable[str]:
        return self.get_minimal_dummy_header().keys()

    def get_shape(self, key: str) -> List[int]:
        return self.get_minimal_dummy_header()[key]["shape"]

    def get_minimal_dummy_header(self) -> Dict[str, Dict[str, str | List[int]]]:
        return self.__minimal_dummy_header

    def get_input_paths(self) -> Set[str | pathlib.Path]:
        return self.__input_paths

    def is_sdxl(self) -> bool:
        return any(
            k.startswith("conditioner.")
            for k in self.get_minimal_dummy_header()
        )

    def intersect(self, other):
        return ModelConfig(
            self.__minimal_dummy_header | other.__minimal_dummy_header,
            self.__keys_to_merge | other.__keys_to_merge,
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

    def __is_passthrough_key(self, key: str):
        if key == "__metadata__":
            return False

        is_vae = key.startswith("first_stage_model.")
        is_position_ids = key == "cond_stage_model.transformer.text_model.embeddings.position_ids"

        # sdxl only
        is_label_embed = key.startswith("model.diffusion_model.label_emb.")
        is_position_ids = is_position_ids or key == "conditioner.embedders.0.transformer.text_model.embeddings.position_ids"

        return is_vae or is_position_ids or is_label_embed or self.get_shape(key) == [1000]

    def __is_merge_key(self, key: str):
        is_unet = key.startswith("model.diffusion_model.")
        is_text_encoder = key.startswith("cond_stage_model.")

        # sdxl only
        is_text_encoder = is_text_encoder or key.startswith("conditioner.embedders.")

        return is_unet or is_text_encoder


class DetermineConfigVisitor(RecipeVisitor):
    def visit_model(self, node: ModelRecipeNode) -> Optional[ModelConfig]:
        return ModelConfig(node.state_dict.header, {node.path})

    def visit_lora(self, node: LoraRecipeNode) -> Optional[ModelConfig]:
        return ModelConfig(node.state_dict.header, {node.path})

    def visit_parameter(self, node: ParameterRecipeNode) -> Optional[ModelConfig]:
        raise TypeError("Recipe parameters do not have configuration")

    def visit_merge(self, node: MergeRecipeNode) -> Optional[ModelConfig]:
        configs = [
            model.accept(self)
            for model in node.models
        ]
        if configs:
            return functools.reduce(ModelConfig.intersect, configs)
        raise ValueError("No input models")


@dataclasses.dataclass
class GatherCombinedHeaderVisitor(RecipeVisitor):
    def visit_model(self, node: ModelRecipeNode) -> Dict[str, dict]:
        return node.state_dict.header.items()

    def visit_lora(self, node: LoraRecipeNode):
        return node.state_dict.header.items()

    def visit_parameter(self, _node: ParameterRecipeNode):
        return {}

    def visit_merge(self, node: MergeRecipeNode):
        return {
            k: v
            for model in node.models
            for k, v in model.accept(self)
        }


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
        )
        try:
            merged = self.recipe.accept(key_merger)
        except KeyError as e:
            merged = self.__get_passthrough_tensor(key)
        return merged

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
        return node.state_dict[self._key]

    def visit_lora(self, node: LoraRecipeNode) -> torch.Tensor:
        lora_key = SD15_LORA_KEY_MAP.get(self._key)
        if lora_key is None:
            raise KeyError(f"No lora key mapping found for target key: {self._key}")

        up_weight = node.state_dict[f"{lora_key}.lora_up.weight"].to(torch.float64)
        down_weight = node.state_dict[f"{lora_key}.lora_down.weight"].to(torch.float64)
        alpha = node.state_dict[f"{lora_key}.alpha"].to(torch.float64)
        dim = down_weight.size()[0]

        if len(down_weight.size()) == 2:  # linear
            res = up_weight @ down_weight
        elif down_weight.size()[2:4] == (1, 1):  # conv2d 1x1
            res = (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:  # conv2d 3x3
            res = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
        return res * (alpha / dim)

    def visit_parameter(self, node: ParameterRecipeNode) -> torch.Tensor:
        raise NotImplementedError(f"Interactive arguments are not yet implemented: parameter '{node.name}' has no value.")

    @abc.abstractmethod
    def visit_merge(self, node: MergeRecipeNode) -> torch.Tensor:
        pass


with open(pathlib.Path(__file__).parent / "lora" / "sd15_keys.json", 'r') as f:
    SD15_LORA_KEY_MAP = json.load(f)


@dataclasses.dataclass
class KeyMergeVisitor(KeyVisitor):
    def visit_merge(self, node: MergeRecipeNode) -> torch.Tensor:
        return node.merge_method(
            self.__visit_deeper_first(node.models),
            {k: get_hyper(v, self._key) for k, v in node.hypers.items()} | node.volatile_hypers,
            self._key,
            node.device if node.device is not None else self._default_device,
            node.dtype if node.dtype is not None else self._default_dtype,
        )

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


def validate_compatibility(configs: List[ModelConfig]):
    are_sdxl = [
        config.is_sdxl()
        for config in configs
    ]
    if all(are_sdxl) or not any(are_sdxl):
        return

    sdxl_count = are_sdxl.count(True)
    sd15_count = are_sdxl.count(False)
    if sdxl_count < sd15_count:
        bad_count, good_count = sdxl_count, sd15_count
        bad_version, good_version = "SDXL", "SD1.5"
        bad_models = ', '.join(
            input_path
            for is_sdxl, config in zip(are_sdxl, configs)
            for input_path in config.get_input_paths()
            if is_sdxl
        )
    else:
        bad_count, good_count = sd15_count, sdxl_count
        bad_version, good_version = "SD1.5", "SDXL"
        bad_models = ', '.join(
            input_path
            for is_sdxl, config in zip(are_sdxl, configs)
            for input_path in config.get_input_paths()
            if not is_sdxl
        )
    raise ValueError(f"Input models are not all the same version. Found {good_count} {good_version} vs {bad_count} {bad_version} models ({bad_models})")
