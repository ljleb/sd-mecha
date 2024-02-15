import logging
import pathlib
import torch
from typing import Optional
from sd_mecha.merge_scheduler import MergeScheduler
from sd_mecha import recipe_nodes
from sd_mecha.sd_meh import merge_methods, extensions, streaming

RecipeNodeOrModel = recipe_nodes.RecipeNode | str | pathlib.Path | streaming.InSafetensorDict


def merge_and_save(
    merge_tree: recipe_nodes.RecipeNode,
    base_dir,
    output_path,
):
    scheduler = MergeScheduler(base_dir=base_dir)
    scheduler.merge_and_save(merge_tree, output_path=output_path)


def weighted_sum(
    a: RecipeNodeOrModel, b: RecipeNodeOrModel, *,
    alpha: float = 0.5,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    work_device: Optional[str] = None,
    work_dtype: Optional[torch.dtype] = None,
) -> recipe_nodes.RecipeNode:
    if not isinstance(a, recipe_nodes.RecipeNode):
        a = recipe_nodes.LeafRecipeNode(a)
    if not isinstance(b, recipe_nodes.RecipeNode):
        b = recipe_nodes.LeafRecipeNode(b)

    if a.merge_space != b.merge_space:
        raise ValueError(f"both models must be in the same merge space. Got: {a.merge_space} != {b.merge_space}")

    return recipe_nodes.SymbolicRecipeNode(
        merge_method=extensions.merge_methods.get("weighted_sum")[0],
        a=a,
        b=b,
        alpha=alpha,
        rebasin_iters=0,
        device=device,
        dtype=dtype,
        work_device=work_device,
        work_dtype=work_dtype,
    )


def add_difference(
    a: RecipeNodeOrModel, b: RecipeNodeOrModel, base: RecipeNodeOrModel, *,
    alpha: float = 0.5,
    threads: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    work_device: Optional[str] = None,
    work_dtype: Optional[torch.dtype] = None,
    clip_weights_to_ab: bool = False,
) -> recipe_nodes.RecipeNode:
    if not isinstance(a, recipe_nodes.RecipeNode):
        a = recipe_nodes.LeafRecipeNode(a)
    if not isinstance(b, recipe_nodes.RecipeNode):
        b = recipe_nodes.LeafRecipeNode(b)
    if not isinstance(base, recipe_nodes.RecipeNode):
        base = recipe_nodes.LeafRecipeNode(base)

    return recipe_nodes.SymbolicRecipeNode(
        merge_method=extensions.merge_methods.get("add_difference"),
        a=a,
        b=b,
        alpha=alpha,
        threads=threads,
        device=device,
        dtype=dtype,
        work_device=work_device,
        work_dtype=work_dtype,
        weights_clip=clip_weights_to_ab,
    )


def tensor_sum(
    a: RecipeNodeOrModel, b: RecipeNodeOrModel, *,
    width: float = 0.5,
    offset: float = 0.0,
    threads: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    work_device: Optional[str] = None,
    work_dtype: Optional[torch.dtype] = None,
) -> recipe_nodes.RecipeNode:
    if not isinstance(a, recipe_nodes.RecipeNode):
        a = recipe_nodes.LeafRecipeNode(a)
    if not isinstance(b, recipe_nodes.RecipeNode):
        b = recipe_nodes.LeafRecipeNode(b)

    return recipe_nodes.SymbolicRecipeNode(
        merge_method=extensions.merge_methods.get("tensor_sum"),
        a=a,
        b=b,
        alpha=width,
        beta=offset,
        threads=threads,
        device=device,
        dtype=dtype,
        work_device=work_device,
        work_dtype=work_dtype,
    )


def add_perpendicular(
    a: RecipeNodeOrModel, b: RecipeNodeOrModel, base: RecipeNodeOrModel, *,
    alpha: float = 1.0,
    rebasin_iters: Optional[int] = None,
    threads: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    work_device: Optional[str] = None,
    work_dtype: Optional[torch.dtype] = None,
    clip_weights_to_ab: bool = False,
) -> recipe_nodes.RecipeNode:
    if not isinstance(a, recipe_nodes.RecipeNode):
        a = recipe_nodes.LeafRecipeNode(a)
    if not isinstance(b, recipe_nodes.RecipeNode):
        b = recipe_nodes.LeafRecipeNode(b)
    if not isinstance(base, recipe_nodes.RecipeNode):
        base = recipe_nodes.LeafRecipeNode(base)

    return recipe_nodes.SymbolicRecipeNode(
        merge_method=extensions.merge_methods.get("add_perpendicular"),
        a=a,
        b=b,
        alpha=alpha,
        rebasin_iters=rebasin_iters,
        threads=threads,
        device=device,
        dtype=dtype,
        work_device=work_device,
        work_dtype=work_dtype,
        weights_clip=clip_weights_to_ab,
    )


def rotate(
    a: RecipeNodeOrModel, b: RecipeNodeOrModel, *,
    alpha: float = 1.0,
    beta: float = 0.0,
    threads: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    work_device: Optional[str] = None,
    work_dtype: Optional[torch.dtype] = None,
    clip_weights_to_ab: bool = False,
) -> recipe_nodes.RecipeNode:
    if not isinstance(a, recipe_nodes.RecipeNode):
        a = recipe_nodes.LeafRecipeNode(a)
    if not isinstance(b, recipe_nodes.RecipeNode):
        b = recipe_nodes.LeafRecipeNode(b)

    return recipe_nodes.SymbolicRecipeNode(
        merge_method=extensions.merge_methods.get("rotate"),
        a=a,
        b=b,
        alpha=alpha,
        beta=beta,
        threads=threads,
        device=device,
        dtype=dtype,
        work_device=work_device,
        work_dtype=work_dtype,
        weights_clip=clip_weights_to_ab,
    )


def clip(model: RecipeNodeOrModel, a: RecipeNodeOrModel, b: RecipeNodeOrModel) -> recipe_nodes.RecipeNode:
    if not isinstance(a, recipe_nodes.RecipeNode):
        a = recipe_nodes.LeafRecipeNode(a)
    if not isinstance(b, recipe_nodes.RecipeNode):
        b = recipe_nodes.LeafRecipeNode(b)

    return recipe_nodes.ClipRecipeNode(model, a, b)


def model(
    state_dict: str | pathlib.Path | streaming.InSafetensorDict,
):
    return recipe_nodes.LeafRecipeNode(
        state_dict=state_dict,
    )


def set_log_level(level: str = "INFO"):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)
