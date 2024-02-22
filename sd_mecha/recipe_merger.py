import dataclasses
import logging
import pathlib
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from sd_mecha.streaming import InLoraSafetensorsDict, InModelSafetensorsDict, OutSafetensorsDict
from sd_mecha import extensions, recipe_nodes, streaming, recipe_serializer
from sd_mecha.hypers import get_hyper
from tqdm import tqdm
from typing import Optional, Tuple, List, Mapping, MutableMapping, Dict
from sd_mecha.user_error import UserError


class RecipeMerger:
    def __init__(
        self, *,
        models_dir: Optional[pathlib.Path | str] = None,
        default_device: str = "cpu",
        default_dtype: Optional[torch.dtype] = torch.float32,
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
        save_dtype: Optional[torch.dtype] = torch.float16,
        threads: Optional[int] = None,
        total_buffer_size: int = 2**28,
        passthrough_model: Optional[Mapping[str, torch.Tensor] | pathlib.Path | str] = None,
    ):
        recipe = extensions.path_to_node(recipe)
        if save_dtype is None:
            save_dtype = self.__default_dtype

        total_files_open = (
            recipe.accept(recipe_nodes.ModelsCountVisitor()) +
            int(isinstance(output, (str, pathlib.Path))) +
            int(isinstance(passthrough_model, (str, pathlib.Path)))
        )
        buffer_size_per_file = total_buffer_size // total_files_open
        if threads is None:
            threads = total_files_open

        load_input_dicts_visitor = LoadInputDictsVisitor(
            self.__base_dir,
            buffer_size_per_file,
        )
        recipe.accept(load_input_dicts_visitor)
        if isinstance(passthrough_model, (str, pathlib.Path)):
            passthrough_model = extensions.path_to_node(passthrough_model)
            passthrough_model.accept(load_input_dicts_visitor)
            passthrough_model = passthrough_model.accept(GatherInputDictsVisitor())[0]
        extensions.clear_model_paths_cache()

        input_dicts = recipe.accept(GatherInputDictsVisitor())
        validate_same_sd_version(input_dicts)
        merged_header = {
            k: {k: v for k, v in h.items() if k != "data_offsets"}
            for input_dict in input_dicts
            for k, h in input_dict.header.items()
            if k != "__metadata__"
        }

        output = self.__normalize_output_to_dict(
            output,
            merged_header,
            recipe_serializer.serialize(recipe),
            buffer_size_per_file // threads,
        )

        def _get_passthrough_tensor(key: str):
            if passthrough_model is not None and key in passthrough_model:
                return passthrough_model[key]

            for input_dict in input_dicts:
                try:
                    return input_dict[key]
                except KeyError:
                    continue

        progress = tqdm(total=len(merged_header.keys()), desc="Merging recipe")

        def _merge_and_save(key: str):
            progress.set_postfix({"key": key, "shape": merged_header[key].get("shape")})
            try:
                key_merger = KeyMergeVisitor(
                    key,
                    self.__default_device,
                    self.__default_dtype,
                )
                merged = recipe.accept(key_merger)
            except KeyError as e:
                merged = _get_passthrough_tensor(key)
            output[key] = merged.to(save_dtype)
            progress.update()

        def _forward_and_save(key: str):
            progress.set_postfix({"key": key, "shape": merged_header[key].get("shape")})
            output[key] = _get_passthrough_tensor(key).to(save_dtype)
            progress.update()

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for key in merged_header:
                if is_passthrough_key(key, merged_header[key]):
                    futures.append(executor.submit(_forward_and_save, key))
                elif is_merge_key(key):
                    futures.append(executor.submit(_merge_and_save, key))
                else:
                    progress.total -= 1
                    progress.refresh()

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


@dataclasses.dataclass
class KeyMergeVisitor:
    __key: str
    __default_device: str
    __default_dtype: torch.dtype

    def visit_model(self, node: recipe_nodes.ModelRecipeNode) -> torch.Tensor:
        return node.state_dict[self.__key]

    def visit_lora(self, node: recipe_nodes.LoraRecipeNode) -> torch.Tensor:
        return node.state_dict[self.__key]

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
        node.state_dict = self.__load_dict(node, InModelSafetensorsDict)

    def visit_lora(self, node: recipe_nodes.LoraRecipeNode):
        node.state_dict = self.__load_dict(node, InLoraSafetensorsDict)

    def visit_parameter(self, _node: recipe_nodes.ParameterRecipeNode):
        return

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode):
        for model in node.models:
            model.accept(self)

    def __load_dict(
        self,
        node: recipe_nodes.LeafRecipeNode,
        dict_class: type,
    ) -> Mapping[str, torch.Tensor]:
        if node.state_dict is not None:
            return node.state_dict

        path = node.path
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        if not path.is_absolute():
            path = self.__base_dir / path
        if not path.suffix:
            path = path.with_suffix(".safetensors")
        return dict_class(path, self.__buffer_size_per_dict)


def is_passthrough_key(key: str, header: dict):
    is_metadata = key == "__metadata__"
    is_vae = key.startswith("first_stage_model.")
    is_position_ids = key == "cond_stage_model.transformer.text_model.embeddings.position_ids"

    # sdxl only
    is_label_embed = key.startswith("model.diffusion_model.label_emb.")
    is_position_ids = is_position_ids or key == "conditioner.embedders.0.transformer.text_model.embeddings.position_ids"

    return is_metadata or is_vae or is_position_ids or is_label_embed or header["shape"] == [1000]


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
    raise UserError(f"Input models are not all the same version. Found {good_count} {good_version} vs {bad_count} {bad_version} models ({bad_models})")
