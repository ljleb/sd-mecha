import logging
import pathlib
import torch
from typing import Optional, Dict
from sd_mecha.recipe_merger import RecipeMerger
from sd_mecha import recipe_nodes, merge_methods
from sd_mecha.extensions import RecipeNodeOrPath, path_to_node
from sd_mecha.recipe_nodes import MergeSpace
from sd_mecha.hypers import (
    Hyper,
    sd15_unet_blocks, sd15_unet_classes, sd15_txt_blocks, sd15_txt_classes,
    sdxl_unet_blocks, sdxl_unet_classes, sdxl_txt_blocks, sdxl_txt_classes, sdxl_txt_g14_blocks, sdxl_txt_g14_classes,
)
from sd_mecha.recipe_serializer import serialize, deserialize, deserialize_path


def merge_and_save(
    recipe: recipe_nodes.RecipeNode,
    models_dir: pathlib.Path,
    output_path: pathlib.Path,
):
    merger = RecipeMerger(models_dir=models_dir)
    merger.merge_and_save(recipe, output=output_path)


def serialize_and_save(
    recipe: recipe_nodes.RecipeNode,
    output_path: pathlib.Path | str,
):
    serialized = serialize(recipe)

    if not isinstance(output_path, pathlib.Path):
        output_path = pathlib.Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".mecha")
    output_path = output_path.absolute()

    logging.info(f"Saving recipe to {output_path}")
    with open(output_path, "w") as f:
        f.write(serialized)


weighted_sum = merge_methods.weighted_sum
slerp = merge_methods.slerp


def add_difference(
    a: RecipeNodeOrPath, b: RecipeNodeOrPath, c: Optional[RecipeNodeOrPath] = None, *,
    alpha: Hyper = 1.0,
    clip_to_ab: Optional[bool] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> recipe_nodes.RecipeNode:
    a: recipe_nodes.RecipeNode = extensions.path_to_node(a)
    b: recipe_nodes.RecipeNode = extensions.path_to_node(b)
    original_b = b

    if c is not None:
        c = extensions.path_to_node(c)
        b = subtract(
            b, c,
            device=device,
            dtype=dtype,
        )

    res = merge_methods.add_difference(
        a, b,
        alpha=alpha,
        device=device,
        dtype=dtype,
    )

    if a.merge_space == original_b.merge_space:
        b = original_b

    if clip_to_ab is None:
        clip_to_ab = a.merge_space == b.merge_space

    if clip_to_ab:
        if a.merge_space != b.merge_space:
            raise TypeError(f"Merge space of A {a.merge_space} and B {b.merge_space} must be the same to clip the merge.")
        res = clip(
            res, a, b,
            device=device,
            dtype=dtype,
        )

    return res


subtract = merge_methods.subtract
perpendicular_component = merge_methods.perpendicular_component


def add_perpendicular(
    a: RecipeNodeOrPath, b: RecipeNodeOrPath, c: RecipeNodeOrPath, *,
    alpha: Hyper = 1.0,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> recipe_nodes.RecipeNode:
    a = path_to_node(a)
    b = path_to_node(b)
    c = path_to_node(c)

    a_diff = subtract(
        a, c,
        device=device,
        dtype=dtype,
    )
    b_diff = subtract(
        b, c,
        device=device,
        dtype=dtype,
    )

    perp_diff = merge_methods.perpendicular_component(
        a_diff, b_diff,
        device=device,
        dtype=dtype,
    )

    return merge_methods.add_difference(
        a, perp_diff,
        alpha=alpha,
        device=device,
        dtype=dtype,
    )


geometric_sum = merge_methods.geometric_sum
copy_difference = train_difference = merge_methods.copy_difference
similarity_add_difference = merge_methods.similarity_add_difference
normalized_similarity_sum = cosine_add_a = merge_methods.normalized_similarity_sum
similarity_sum = cosine_add_b = merge_methods.similarity_sum
ties_sum = merge_methods.ties_sum


def add_difference_ties(
    base: RecipeNodeOrPath,
    *models: RecipeNodeOrPath,
    alpha: float,
    k: float = 0.2,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> recipe_nodes.RecipeNode:
    base = path_to_node(base)
    models = tuple(path_to_node(model) for model in models)
    models = tuple(
        subtract(model, base)
        if model.merge_space is MergeSpace.MODEL else
        model
        for model in models
    )
    res = ties_sum(
        *models,
        alpha=alpha,
        k=k,
        device=device,
        dtype=dtype,
    )
    return add_difference(
        base, res,
        alpha=1.0,
        device=device,
        dtype=dtype,
    )


def copy_region(
    a: RecipeNodeOrPath, b: RecipeNodeOrPath, c: Optional[RecipeNodeOrPath], *,
    width: Hyper = 0.5,
    offset: Hyper = 0.0,
    top_k: bool = False,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> recipe_nodes.RecipeNode:
    a = path_to_node(a)
    b = path_to_node(b)

    if c is not None:
        c = path_to_node(c)

        a = subtract(
            a, c,
            device=device,
            dtype=dtype,
        )
        b = subtract(
            b, c,
            device=device,
            dtype=dtype,
        )

    copy_method = [merge_methods.copy_region, merge_methods.copy_top_k][int(top_k)]
    res = copy_method(
        a, b,
        alpha=width,
        beta=offset,
        device=device,
        dtype=dtype,
    )

    if c is not None:
        res = merge_methods.add_difference(
            c, res,
            alpha=1.0,
            device=device,
            dtype=dtype,
        )

    return res


tensor_sum = copy_region
distribution_crossover = merge_methods.distribution_crossover
crossover = merge_methods.crossover


def rotate(
    a: RecipeNodeOrPath, b: RecipeNodeOrPath, c: Optional[RecipeNodeOrPath] = None, *,
    alpha: Hyper = 1.0,
    beta: Hyper = 0.0,
    device: Optional[str] = "cuda",
    dtype: Optional[torch.dtype] = torch.float64,
    cache: Optional[Dict[str, torch.Tensor]] = None,
) -> recipe_nodes.RecipeNode:
    a = path_to_node(a)
    b = path_to_node(b)

    if c is not None:
        c = path_to_node(c)

        a = subtract(
            a, c,
            device=device,
            dtype=dtype,
        )
        b = subtract(
            b, c,
            device=device,
            dtype=dtype,
        )

    res = merge_methods.rotate(
        a, b,
        alpha=alpha,
        beta=beta,
        cache=cache,
        device=device,
        dtype=dtype,
    )

    if c is not None:
        res = merge_methods.add_difference(
            c, res,
            alpha=1.0,
            device=device,
            dtype=dtype,
        )

    return res


clip = merge_methods.clip


def model(state_dict: str | pathlib.Path):
    return recipe_nodes.ModelRecipeNode(state_dict)


def lora(state_dict: str | pathlib.Path):
    return recipe_nodes.LoraRecipeNode(state_dict)


def parameter(name: str):
    return recipe_nodes.ParameterRecipeNode(name)


def set_log_level(level: str = "INFO"):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)
