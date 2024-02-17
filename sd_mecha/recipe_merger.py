import logging
import pathlib
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from sd_mecha.streaming import InLoraSafetensorsDict, InModelSafetensorsDict, OutSafetensorsDict
from sd_mecha import extensions, recipe_nodes
from sd_mecha.hypers import get_hyper
from tqdm import tqdm
from typing import Optional, Tuple, List, Mapping


class RecipeMerger:
    def __init__(
        self, *,
        base_dir: Optional[pathlib.Path | str] = None,
        default_device: str = "cpu",
        default_dtype: Optional[torch.dtype] = torch.float32,
    ):
        self.__base_dir = base_dir if base_dir is not None else base_dir
        if isinstance(self.__base_dir, str):
            self.__base_dir = pathlib.Path(self.__base_dir)
        if self.__base_dir is not None:
            self.__base_dir = self.__base_dir.absolute()

        self.__default_device = default_device
        self.__default_dtype = default_dtype

    def merge_and_save(
        self, recipe, *,
        output_path: Optional[pathlib.Path | str] = None,
        save_dtype: Optional[torch.dtype] = torch.float16,
        threads: int = 1,
    ):
        extensions.clear_model_paths_cache()
        if save_dtype is None:
            save_dtype = self.__default_dtype
        if not isinstance(output_path, pathlib.Path):
            output_path = pathlib.Path(output_path)
        if not output_path.is_absolute():
            output_path = self.__base_dir / output_path
        if not output_path.suffix:
            output_path = output_path.with_suffix(".safetensors")
        logging.info(f"Saving to {output_path}")

        input_dicts_visitor = GatherInputDictsVisitor(self.__base_dir)
        input_dicts = recipe.accept(input_dicts_visitor)
        merged_header = {
            k: {k: v for k, v in h.items() if k != "data_offsets"}
            for input_dict in input_dicts
            for k, h in input_dict.header.items()
            if k != "__metadata__"
        }

        def _get_any_tensor(key: str):
            for input_dict in input_dicts:
                try:
                    return input_dict[key]
                except KeyError:
                    continue

        output = OutSafetensorsDict(output_path, merged_header)
        progress = tqdm(total=len(merged_header.keys()), desc="Merging recipe")

        def _merge_and_save(key: str):
            progress.set_postfix({"key": key, "shape": merged_header[key].get("shape")})
            try:
                key_merger = KeyMergeVisitor(
                    key,
                    self.__base_dir,
                    self.__default_device,
                    self.__default_dtype,
                )
                merged = recipe.accept(key_merger)
            except KeyError:
                merged = _get_any_tensor(key)
            output[key] = merged.to(save_dtype)
            progress.update()

        def _forward_and_save(key: str):
            progress.set_postfix({"key": key, "shape": merged_header[key].get("shape")})
            output[key] = _get_any_tensor(key).to(save_dtype)
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

        output.finalize()


class KeyMergeVisitor:
    def __init__(
        self, key: str,
        base_dir: pathlib.Path,
        default_device: str,
        default_dtype: torch.dtype,
    ):
        self.__key = key
        self.__base_dir = base_dir
        self.__default_device = default_device
        self.__default_dtype = default_dtype

    def visit_model(self, node: recipe_nodes.ModelRecipeNode) -> torch.Tensor:
        node.state_dict = load_dict(node, InModelSafetensorsDict, self.__base_dir)
        return node.state_dict[self.__key]

    def visit_lora(self, node: recipe_nodes.LoraRecipeNode) -> torch.Tensor:
        node.state_dict = load_dict(node, InLoraSafetensorsDict, self.__base_dir)
        return node.state_dict[self.__key]

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode) -> torch.Tensor:
        return node.merge_method(
            self.__visit_deeper_first(node.models),
            {k: get_hyper(v, self.__key) for k, v in node.hypers.items()},
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
    def __init__(self, base_dir: pathlib.Path):
        self.__base_dir = base_dir

    def visit_model(self, node: recipe_nodes.ModelRecipeNode) -> List[Mapping[str, torch.Tensor]]:
        node.state_dict = load_dict(node, InModelSafetensorsDict, self.__base_dir)
        return [node.state_dict]

    def visit_lora(self, node: recipe_nodes.LoraRecipeNode) -> List[Mapping[str, torch.Tensor]]:
        node.state_dict = load_dict(node, InLoraSafetensorsDict, self.__base_dir)
        return [node.state_dict]

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode) -> List[Mapping[str, torch.Tensor]]:
        return [
            input_dict
            for model in node.models
            for input_dict in model.accept(self)
        ]


def load_dict(
    node: recipe_nodes.ModelRecipeNode | recipe_nodes.LoraRecipeNode,
    dict_class: type,
    base_dir: pathlib.Path,
) -> Mapping[str, torch.Tensor]:
    if node.state_dict is not None:
        return node.state_dict

    path = node.path
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)
    if not path.is_absolute():
        path = base_dir / path
    if not path.suffix:
        path = path.with_suffix(".safetensors")
    return dict_class(path)


def is_passthrough_key(key: str, header: dict):
    is_metadata = key == "__metadata__"
    is_vae = key.startswith("first_stage_model.")
    is_time_embed = key.startswith("model.diffusion_model.time_embed.")
    is_position_ids = key == "cond_stage_model.transformer.text_model.embeddings.position_ids"
    return is_metadata or is_vae or is_time_embed or is_position_ids or header["shape"] == [1000]


def is_merge_key(key: str):
    is_unet = key.startswith("model.diffusion_model.")
    is_text_encoder = key.startswith("cond_stage_model.")
    return is_unet or is_text_encoder