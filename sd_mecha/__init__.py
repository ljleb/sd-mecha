import logging
import pathlib
import torch
from typing import Optional
from sd_mecha.merge_scheduler import MergeScheduler
from sd_mecha import recipe_nodes, merge_methods, extensions, streaming
from sd_mecha.extensions import MergeSpace
from sd_mecha.weight import ModelParameter, unet15_blocks, unet15_classes, txt15_blocks, txt15_classes


RecipeNodeOrModel = recipe_nodes.RecipeNode | str | pathlib.Path


def merge_and_save(
    merge_tree: recipe_nodes.RecipeNode,
    base_dir: pathlib.Path,
    output_path: pathlib.Path,
):
    scheduler = MergeScheduler(base_dir=base_dir)
    scheduler.merge_and_save(merge_tree, output_path=output_path)


def weighted_sum(
    a: RecipeNodeOrModel, b: RecipeNodeOrModel, *,
    alpha: ModelParameter = 0.5,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> recipe_nodes.RecipeNode:
    a = to_node(a)
    b = to_node(b)

    return recipe_nodes.SymbolicRecipeNode(
        merge_method=extensions.merge_methods.get("weighted_sum"),
        a=a,
        b=b,
        alpha=alpha,
        device=device,
        dtype=dtype,
    )


def add_difference(
    a: RecipeNodeOrModel, b: RecipeNodeOrModel, c: Optional[RecipeNodeOrModel] = None, *,
    alpha: ModelParameter = 0.5,
    clip_to_ab: Optional[bool] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> recipe_nodes.RecipeNode:
    a = to_node(a)
    b = to_node(b)
    original_b = b

    if c is not None:
        c = to_node(c)
        b = subtract(
            b, c,
            device=device,
            dtype=dtype,
        )

    res = recipe_nodes.SymbolicRecipeNode(
        merge_method=extensions.merge_methods.get("add_difference"),
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
            raise TypeError(f"Merge space of A {a.merge_space} and B {b.merge_space} must be the same to clip the merge")
        res = clip(
            res, a, b,
            device=device,
            dtype=dtype,
        )

    return res


def subtract(
    a: RecipeNodeOrModel, b: RecipeNodeOrModel, *,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> recipe_nodes.RecipeNode:
    if not isinstance(a, recipe_nodes.RecipeNode):
        a = recipe_nodes.ModelRecipeNode(a)
    if not isinstance(b, recipe_nodes.RecipeNode):
        b = recipe_nodes.ModelRecipeNode(b)

    return recipe_nodes.SymbolicRecipeNode(
        merge_method=extensions.merge_methods.get("subtract"),
        a=a,
        b=b,
        device=device,
        dtype=dtype,
    )


def tensor_sum(
    a: RecipeNodeOrModel, b: RecipeNodeOrModel, c: Optional[RecipeNodeOrModel], *,
    width: ModelParameter = 0.5,
    offset: ModelParameter = 0.0,
    top_k: bool = False,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> recipe_nodes.RecipeNode:
    if not isinstance(a, recipe_nodes.RecipeNode):
        a = recipe_nodes.ModelRecipeNode(a)
    if not isinstance(b, recipe_nodes.RecipeNode):
        b = recipe_nodes.ModelRecipeNode(b)
    if c is not None:
        if not isinstance(c, recipe_nodes.RecipeNode):
            c = recipe_nodes.ModelRecipeNode(c)

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

    method_name = "top_k_tensor_sum" if top_k else "tensor_sum"
    res = recipe_nodes.SymbolicRecipeNode(
        merge_method=extensions.merge_methods.get(method_name),
        a=a,
        b=b,
        alpha=width,
        beta=offset,
        device=device,
        dtype=dtype,
    )

    if c is not None:
        res = recipe_nodes.SymbolicRecipeNode(
            merge_method=extensions.merge_methods.get("add_difference"),
            a=c,
            b=res,
            alpha=width,
            beta=offset,
            device=device,
            dtype=dtype,
        )

    return res


def add_perpendicular(
    a: RecipeNodeOrModel, b: RecipeNodeOrModel, c: RecipeNodeOrModel, *,
    alpha: ModelParameter = 1.0,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> recipe_nodes.RecipeNode:
    if not isinstance(a, recipe_nodes.RecipeNode):
        a = recipe_nodes.ModelRecipeNode(a)
    if not isinstance(b, recipe_nodes.RecipeNode):
        b = recipe_nodes.ModelRecipeNode(b)
    if not isinstance(c, recipe_nodes.RecipeNode):
        c = recipe_nodes.ModelRecipeNode(c)

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

    perp_diff = recipe_nodes.SymbolicRecipeNode(
        merge_method=extensions.merge_methods.get("perpendicular_component"),
        a=a_diff,
        b=b_diff,
        device=device,
        dtype=dtype,
    )

    return recipe_nodes.SymbolicRecipeNode(
        merge_method=extensions.merge_methods.get("add_difference"),
        a=a,
        b=perp_diff,
        alpha=alpha,
        device=device,
        dtype=dtype,
    )


def rotate(
    a: RecipeNodeOrModel, b: RecipeNodeOrModel, c: Optional[RecipeNodeOrModel] = None, *,
    alpha: ModelParameter = 1.0,
    beta: ModelParameter = 0.0,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = torch.float64,
) -> recipe_nodes.RecipeNode:
    if not isinstance(a, recipe_nodes.RecipeNode):
        a = recipe_nodes.ModelRecipeNode(a)
    if not isinstance(b, recipe_nodes.RecipeNode):
        b = recipe_nodes.ModelRecipeNode(b)
    if c is not None:
        if not isinstance(c, recipe_nodes.RecipeNode):
            c = recipe_nodes.ModelRecipeNode(c)

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

    res = recipe_nodes.SymbolicRecipeNode(
        merge_method=extensions.merge_methods.get("rotate"),
        a=a,
        b=b,
        alpha=alpha,
        beta=beta,
        device=device,
        dtype=dtype,
    )

    if c is not None:
        res = recipe_nodes.SymbolicRecipeNode(
            merge_method=extensions.merge_methods.get("add_difference"),
            a=c,
            b=res,
            alpha=1.0,
            device=device,
            dtype=dtype,
        )

    return res


def clip(
    model: RecipeNodeOrModel, a: RecipeNodeOrModel, b: RecipeNodeOrModel, *,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
) -> recipe_nodes.RecipeNode:
    if not isinstance(model, recipe_nodes.RecipeNode):
        model = recipe_nodes.ModelRecipeNode(a)
    if not isinstance(a, recipe_nodes.RecipeNode):
        a = recipe_nodes.ModelRecipeNode(a)
    if not isinstance(b, recipe_nodes.RecipeNode):
        b = recipe_nodes.ModelRecipeNode(b)

    return recipe_nodes.SymbolicRecipeNode(
        merge_method=extensions.merge_methods.get("clip"),
        a=model,
        b=a,
        c=b,
        device=device,
        dtype=dtype,
    )


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


def to_node(a: RecipeNodeOrModel) -> recipe_nodes.RecipeNode:
    if isinstance(a, (str, pathlib.Path)):
        return recipe_nodes.ModelRecipeNode(a)
    return a


def set_log_level(level: str = "INFO"):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)
