import logging
import pathlib
import torch
from typing import Optional
from sd_mecha.merge_scheduler import MergeScheduler
from sd_mecha import recipe_nodes, merge_methods
from sd_mecha.extensions import RecipeNodeOrPath, path_to_node
from sd_mecha.recipe_nodes import MergeSpace
from sd_mecha.weight import Hyper, unet15_blocks, unet15_classes, txt15_blocks, txt15_classes
from sd_mecha.recipe_serializer import serialize, deserialize


def merge_and_save(
    merge_tree: recipe_nodes.RecipeNode,
    base_dir: pathlib.Path,
    output_path: pathlib.Path,
):
    scheduler = MergeScheduler(base_dir=base_dir)
    scheduler.merge_and_save(merge_tree, output_path=output_path)


weighted_sum = merge_methods.weighted_sum
slerp = merge_methods.slerp


def add_difference(
    a: RecipeNodeOrPath, b: RecipeNodeOrPath, c: Optional[RecipeNodeOrPath] = None, *,
    alpha: Hyper = 0.5,
    clip_to_ab: Optional[bool] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> recipe_nodes.RecipeNode:
    a = extensions.path_to_node(a)
    b = extensions.path_to_node(b)
    original_b = b

    if c is not None:
        c = extensions.path_to_node(c)
        b = subtract(
            b, c,
            device=device,
            dtype=dtype,
        )

    res = merge_methods.add_difference(
        a=a,
        b=b,
        alpha=alpha,
        device=device,
        dtype=dtype,
    )

    if clip_to_ab is None:
        ab_same_space = a.merge_space == b.merge_space
        abo_same_space = a.merge_space == original_b.merge_space
        clip_to_ab = ab_same_space or abo_same_space
        if abo_same_space:
            b = original_b

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
        a=a_diff,
        b=b_diff,
        device=device,
        dtype=dtype,
    )

    return merge_methods.add_difference(
        a=a,
        b=perp_diff,
        alpha=alpha,
        device=device,
        dtype=dtype,
    )


multiply_difference = merge_methods.multiply_difference
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
        base,
        res,
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

    copy_method = [merge_methods.copy_top_k, merge_methods.copy_region][int(top_k)]
    res = copy_method(
        a=a,
        b=b,
        alpha=width,
        beta=offset,
        device=device,
        dtype=dtype,
    )

    if c is not None:
        res = merge_methods.add_difference(
            a=c,
            b=res,
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
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = torch.float64,
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
        a=a,
        b=b,
        alpha=alpha,
        beta=beta,
        device=device,
        dtype=dtype,
    )

    if c is not None:
        res = merge_methods.add_difference(
            a=c,
            b=res,
            alpha=1.0,
            device=device,
            dtype=dtype,
        )

    return res


clip = merge_methods.clip


def model(
    state_dict: str | pathlib.Path,
):
    return recipe_nodes.ModelRecipeNode(
        state_dict=state_dict,
    )


def lora(
    state_dict: str | pathlib.Path,
):
    return recipe_nodes.LoraRecipeNode(
        state_dict=state_dict,
    )


def set_log_level(level: str = "INFO"):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)
