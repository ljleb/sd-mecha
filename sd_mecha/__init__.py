import logging
import pathlib
import torch
from typing import Optional
from sd_mecha.merge_scheduler import MergeScheduler
from sd_mecha import recipe_nodes
from sd_mecha.sd_meh import merge_methods, extensions, streaming
from sd_mecha.sd_meh.extensions import MergeSpace


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

    return recipe_nodes.SymbolicRecipeNode(
        merge_method=extensions.merge_methods.get("weighted_sum"),
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
    a: RecipeNodeOrModel, b: RecipeNodeOrModel, c: RecipeNodeOrModel, *,
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
    if not isinstance(c, recipe_nodes.RecipeNode):
        c = recipe_nodes.LeafRecipeNode(c)

    # outputs .to(work_device, work_dtype) because we are still inside the method
    b_diff = subtract(
        b, c,
        device=work_device if work_device is not None else device,
        work_device=work_device if work_device is not None else device,
        dtype=work_dtype if work_dtype is not None else dtype,
        work_dtype=work_dtype if work_dtype is not None else dtype,
    )

    return recipe_nodes.SymbolicRecipeNode(
        merge_method=extensions.merge_methods.get("add"),
        a=a,
        b=b_diff,
        alpha=alpha,
        device=device,
        dtype=dtype,
        work_device=work_device,
        work_dtype=work_dtype,
    )


def subtract(
    a: RecipeNodeOrModel, b: RecipeNodeOrModel, *,
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
        merge_method=extensions.merge_methods.get("subtract"),
        a=a,
        b=b,
        device=device,
        dtype=dtype,
        work_device=work_device,
        work_dtype=work_dtype,
    )


def tensor_sum(
    a: RecipeNodeOrModel, b: RecipeNodeOrModel, *,
    width: float = 0.5,
    offset: float = 0.0,
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
        device=device,
        dtype=dtype,
        work_device=work_device,
        work_dtype=work_dtype,
    )


def add_perpendicular(
    a: RecipeNodeOrModel, b: RecipeNodeOrModel, c: RecipeNodeOrModel, *,
    alpha: float = 1.0,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    work_device: Optional[str] = None,
    work_dtype: Optional[torch.dtype] = None,
) -> recipe_nodes.RecipeNode:
    if not isinstance(a, recipe_nodes.RecipeNode):
        a = recipe_nodes.LeafRecipeNode(a)
    if not isinstance(b, recipe_nodes.RecipeNode):
        b = recipe_nodes.LeafRecipeNode(b)
    if not isinstance(c, recipe_nodes.RecipeNode):
        c = recipe_nodes.LeafRecipeNode(c)

    # outputs .to(work_device, work_dtype) because we are still inside the method
    a_diff = subtract(
        a, c,
        device=work_device if work_device is not None else device,
        work_device=work_device if work_device is not None else device,
        dtype=work_dtype if work_dtype is not None else dtype,
        work_dtype=work_dtype if work_dtype is not None else dtype,
    )

    # outputs .to(work_device, work_dtype) because we are still inside the method
    b_diff = subtract(
        b, c,
        device=work_device if work_device is not None else device,
        work_device=work_device if work_device is not None else device,
        dtype=work_dtype if work_dtype is not None else dtype,
        work_dtype=work_dtype if work_dtype is not None else dtype,
    )

    # outputs .to(work_device, work_dtype) because we are still inside the method
    perp_diff = recipe_nodes.SymbolicRecipeNode(
        merge_method=extensions.merge_methods.get("add_perpendicular"),
        a=a_diff,
        b=b_diff,
        alpha=alpha,
        device=work_device if work_device is not None else device,
        work_device=work_device if work_device is not None else device,
        dtype=work_dtype if work_dtype is not None else dtype,
        work_dtype=work_dtype if work_dtype is not None else dtype,
    )

    return recipe_nodes.SymbolicRecipeNode(
        merge_method=extensions.merge_methods.get("add"),
        a=c,
        b=perp_diff,
        alpha=alpha,
        device=device,
        dtype=dtype,
        work_device=work_device,
        work_dtype=work_dtype,
    )


def rotate(
    a: RecipeNodeOrModel, b: RecipeNodeOrModel, *,
    alpha: float = 1.0,
    beta: float = 0.0,
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
        merge_method=extensions.merge_methods.get("rotate"),
        a=a,
        b=b,
        alpha=alpha,
        beta=beta,
        device=device,
        dtype=dtype,
        work_device=work_device,
        work_dtype=work_dtype,
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
