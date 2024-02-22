import dataclasses
import json
import logging
import pathlib
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from sd_mecha.streaming import InSafetensorsDict, OutSafetensorsDict
from sd_mecha import extensions, recipe_nodes, streaming, recipe_serializer
from sd_mecha.hypers import get_hyper
from tqdm import tqdm
from typing import Optional, Tuple, List, Mapping, MutableMapping, Dict


class RecipeMerger:
    def __init__(
        self, *,
        models_dir: Optional[pathlib.Path | str] = None,
        default_device: str = "cpu",
        default_dtype: Optional[torch.dtype] = torch.float64,
    ):
        self.__base_dir = models_dir if models_dir is not None else models_dir
        if isinstance(self.__base_dir, str):
            self.__base_dir = pathlib.Path(self.__base_dir)
        if self.__base_dir is not None:
            self.__base_dir = self.__base_dir.absolute()

        self.__default_device = default_device
        self.__default_dtype = default_dtype

    def merge_and_save(
        self, recipe: extensions.RecipeNodeOrPath, *,
        output: MutableMapping[str, torch.Tensor] | pathlib.Path | str = "merge",
        fallback_model: Optional[Mapping[str, torch.Tensor] | pathlib.Path | str] = None,
        save_dtype: Optional[torch.dtype] = torch.float16,
        threads: Optional[int] = None,
        total_buffer_size: int = 2**28,
    ):
        recipe = extensions.path_to_node(recipe)
        if isinstance(fallback_model, (str, pathlib.Path)):
            fallback_model = extensions.path_to_node(fallback_model)
        fallback_in_recipe = fallback_model in recipe
        extensions.clear_model_paths_cache()

        total_files_open = (
            recipe.accept(recipe_nodes.ModelsCountVisitor()) +
            int(isinstance(output, (str, pathlib.Path))) +
            int(isinstance(fallback_model, recipe_nodes.RecipeNode) and not fallback_in_recipe)
        )
        buffer_size_per_file = total_buffer_size // total_files_open
        if threads is None:
            threads = total_files_open

        load_input_dicts_visitor = LoadInputDictsVisitor(
            self.__base_dir,
            buffer_size_per_file,
        )
        recipe.accept(load_input_dicts_visitor)
        if isinstance(fallback_model, recipe_nodes.RecipeNode):
            fallback_model.accept(load_input_dicts_visitor)
            fallback_model = fallback_model.accept(GatherInputDictsVisitor())[0]

        input_dicts = recipe.accept(GatherInputDictsVisitor())
        validate_same_sd_version(input_dicts)
        merged_header = recipe.accept(GatherCombinedHeaderVisitor())

        key_decisions = {}
        for key in merged_header:
            if is_passthrough_key(key, merged_header[key]):
                key_decisions[key] = lambda: key_merger.forward_and_save
            elif is_merge_key(key):
                key_decisions[key] = lambda: key_merger.merge_and_save

        merged_header = {
            k: v
            for k, v in merged_header.items()
            if k in key_decisions
        }

        output = self.__normalize_output_to_dict(
            output,
            merged_header,
            recipe_serializer.serialize(recipe),
            buffer_size_per_file // threads,
        )

        key_merger = KeyMerger(
            output,
            fallback_model,
            input_dicts,
            recipe,
            self.__default_device,
            self.__default_dtype,
            save_dtype or self.__default_dtype,
        )

        progress = tqdm(total=len(key_decisions), desc="Merging recipe")
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for key, f in key_decisions.items():
                futures.append(executor.submit(self.__wrap_progress(f(), progress), key, merged_header[key]))

            for future in as_completed(futures):
                future.result()

        progress.close()
        if hasattr(output, "close"):
            output.close()

    def __normalize_output_to_dict(
        self,
        output: MutableMapping[str, torch.Tensor] | pathlib.Path | str,
        merged_header: Dict[str, dict],
        serialized_recipe: str,
        buffer_size_per_thread: int,
    ):
        if isinstance(output, (str, pathlib.Path)):
            if not isinstance(output, pathlib.Path):
                output = pathlib.Path(output)
            if not output.is_absolute():
                output = self.__base_dir / output
            if not output.suffix:
                output = output.with_suffix(".safetensors")
            logging.info(f"Saving to {output}")

            output = OutSafetensorsDict(
                output,
                merged_header,
                serialized_recipe,
                buffer_size_per_thread,
            )
        return output

    def __wrap_progress(self, f, progress):
        def track_progress(key, key_header, *args, **kwargs):
            progress.set_postfix({"key": key, "shape": key_header.get("shape")})
            res = f(key, key_header, *args, **kwargs)
            progress.update()
            return res
        return track_progress


@dataclasses.dataclass
class KeyMerger:
    output: MutableMapping[str, torch.Tensor]
    fallback_model: Mapping[str, torch.Tensor]
    input_dicts: List[Mapping[str, torch.Tensor]]
    recipe: recipe_nodes.RecipeNode
    default_device: str
    default_dtype: torch.dtype
    save_dtype: torch.dtype

    def merge_and_save(
        self,
        key: str,
        key_header: dict,
    ):
        try:
            key_merger = KeyMergeVisitor(
                key,
                self.default_device,
                self.default_dtype,
            )
            merged = self.recipe.accept(key_merger)
        except KeyError as e:
            merged = self.__get_passthrough_tensor(key)
        self.output[key] = merged.to(self.save_dtype)

    def forward_and_save(
        self,
        key: str,
        key_header: dict,
    ):
        self.output[key] = self.__get_passthrough_tensor(key).to(self.save_dtype)

    def __get_passthrough_tensor(self, key: str):
        if self.fallback_model is not None and key in self.fallback_model:
            return self.fallback_model[key]

        for input_dict in self.input_dicts:
            try:
                return input_dict[key]
            except KeyError:
                continue


@dataclasses.dataclass
class KeyMergeVisitor:
    __key: str
    __default_device: str
    __default_dtype: torch.dtype

    def visit_model(self, node: recipe_nodes.ModelRecipeNode) -> torch.Tensor:
        return node.state_dict[self.__key]

    def visit_lora(self, node: recipe_nodes.LoraRecipeNode) -> torch.Tensor:
        lora_key = SD15_LORA_KEY_MAP.get(self.__key)
        if lora_key is None:
            raise KeyError(f"No lora key mapping found for target key: {self.__key}")

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

    def visit_parameter(self, node: recipe_nodes.ParameterRecipeNode) -> torch.Tensor:
        raise NotImplementedError(f"Interactive arguments are not yet implemented: parameter '{node.name}' has no value.")

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode) -> torch.Tensor:
        return node.merge_method(
            self.__visit_deeper_first(node.models),
            {k: get_hyper(v, self.__key) for k, v in node.hypers.items()} | node.volatile_hypers,
            self.__key,
            node.device if node.device is not None else self.__default_device,
            node.dtype if node.dtype is not None else self.__default_dtype,
        )

    def __visit_deeper_first(self, nodes: Tuple[recipe_nodes.RecipeNode, ...]) -> list:
        merged: List[Optional[torch.Tensor]] = [None] * len(nodes)

        def depth_of_value(index) -> int:
            if nodes[index] is None:
                return 0
            return nodes[index].accept(recipe_nodes.DepthRecipeVisitor())

        for index in sorted(range(len(nodes)), key=depth_of_value, reverse=True):
            if nodes[index] is None:
                continue
            merged[index] = nodes[index].accept(self)

        return merged


with open(pathlib.Path(__file__).parent / "lora" / "sd15_keys.json", 'r') as f:
    SD15_LORA_KEY_MAP = json.load(f)


class GatherInputDictsVisitor:
    def visit_model(self, node: recipe_nodes.ModelRecipeNode) -> List[Mapping[str, torch.Tensor]]:
        return [node.state_dict]

    def visit_lora(self, node: recipe_nodes.LoraRecipeNode) -> List[Mapping[str, torch.Tensor]]:
        return [node.state_dict]

    def visit_parameter(self, _node: recipe_nodes.ParameterRecipeNode) -> List[Mapping[str, torch.Tensor]]:
        return []

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode) -> List[Mapping[str, torch.Tensor]]:
        res = []
        for model in node.models:
            for input_dict in model.accept(self):
                if input_dict not in res:
                    res.append(input_dict)
        return res


@dataclasses.dataclass
class LoadInputDictsVisitor:
    __base_dir: pathlib.Path
    __buffer_size_per_dict: int

    def visit_model(self, node: recipe_nodes.ModelRecipeNode):
        node.state_dict = self.__load_dict(node)

    def visit_lora(self, node: recipe_nodes.LoraRecipeNode):
        node.state_dict = self.__load_dict(node)

    def visit_parameter(self, _node: recipe_nodes.ParameterRecipeNode):
        return

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode):
        for model in node.models:
            model.accept(self)

    def __load_dict(
        self,
        node: recipe_nodes.LeafRecipeNode,
    ) -> InSafetensorsDict:
        if node.state_dict is not None:
            return node.state_dict

        path = node.path
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        if not path.is_absolute():
            path = self.__base_dir / path
        if not path.suffix:
            path = path.with_suffix(".safetensors")
        return InSafetensorsDict(path, self.__buffer_size_per_dict)


@dataclasses.dataclass
class GatherCombinedHeaderVisitor:
    def visit_model(self, node: recipe_nodes.ModelRecipeNode) -> Dict[str, dict]:
        return node.state_dict.header.items()

    def visit_lora(self, node: recipe_nodes.LoraRecipeNode):
        return node.state_dict.header.items()

    def visit_parameter(self, _node: recipe_nodes.ParameterRecipeNode):
        return {}

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode):
        return {
            k: v
            for model in node.models
            for k, v in model.accept(self)
        }


def is_passthrough_key(key: str, header: dict):
    is_vae = key.startswith("first_stage_model.")
    is_position_ids = key == "cond_stage_model.transformer.text_model.embeddings.position_ids"

    # sdxl only
    is_label_embed = key.startswith("model.diffusion_model.label_emb.")
    is_position_ids = is_position_ids or key == "conditioner.embedders.0.transformer.text_model.embeddings.position_ids"

    return is_vae or is_position_ids or is_label_embed or (key != "__metadata__" and header["shape"] == [1000])


def is_merge_key(key: str):
    is_unet = key.startswith("model.diffusion_model.")
    is_text_encoder = key.startswith("cond_stage_model.")

    # sdxl only
    is_text_encoder = is_text_encoder or key.startswith("conditioner.embedders.")

    return is_unet or is_text_encoder


def validate_same_sd_version(input_dicts: List[streaming.InSafetensorsDict]):
    are_sdxl = [
        input_dict.is_sdxl
        for input_dict in input_dicts
    ]
    if all(are_sdxl) or not any(are_sdxl):
        return

    sdxl_count = are_sdxl.count(True)
    sd15_count = are_sdxl.count(False)
    if sdxl_count < sd15_count:
        bad_count, good_count = sdxl_count, sd15_count
        bad_version, good_version = "SDXL", "SD1.5"
        bad_models = ', '.join(
            input_dict.input_path
            for is_sdxl, input_dict in zip(are_sdxl, input_dicts)
            if is_sdxl
        )
    else:
        bad_count, good_count = sd15_count, sdxl_count
        bad_version, good_version = "SD1.5", "SDXL"
        bad_models = ', '.join(
            input_dict.input_path
            for is_sdxl, input_dict in zip(are_sdxl, input_dicts)
            if not is_sdxl
        )
    raise ValueError(f"Input models are not all the same version. Found {good_count} {good_version} vs {bad_count} {bad_version} models ({bad_models})")
