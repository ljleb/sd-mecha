import torch
from typing import Optional, Dict
from torch import Tensor
from .extensions.merge_methods import value_to_node, RecipeNodeOrValue, Parameter
from . import recipe_nodes
from sd_mecha.extensions.builtin import merge_methods
from sd_mecha.extensions.builtin.merge_methods import (
    subtract,
    ties_sum,
    ties_sum_extended,
    clamp,
    model_stock,
)


def add_difference(
    a: RecipeNodeOrValue, b: RecipeNodeOrValue, c: Optional[RecipeNodeOrValue] = None, *,
    alpha: Parameter(Tensor) = 1.0,
    clamp_to_ab: Parameter(bool) = False,
) -> recipe_nodes.RecipeNode:
    a = value_to_node(a)
    b = value_to_node(b)
    original_b = b

    if c is not None:
        c = value_to_node(c)
        b = subtract(
            b, c,
        )

    res = merge_methods.add_difference(
        a, b,
        alpha=alpha,
    )

    if a.merge_space == original_b.merge_space:
        b = original_b

    if clamp_to_ab:
        if a.merge_space != b.merge_space:
            raise TypeError(f"Merge space of A {a.merge_space} and B {b.merge_space} must be the same to clamp the merge.")
        res = clamp(
            res, a, b,
        )

    return res | a


def add_perpendicular(
    a: RecipeNodeOrValue, b: RecipeNodeOrValue, c: RecipeNodeOrValue, *,
    alpha: Parameter(Tensor) = 1.0,
) -> recipe_nodes.RecipeNode:
    a = value_to_node(a)
    b = value_to_node(b)
    c = value_to_node(c)

    a_diff = subtract(
        a, c,
    )
    b_diff = subtract(
        b, c,
    )

    perp_diff = merge_methods.perpendicular_component(
        a_diff, b_diff,
    )

    return merge_methods.add_difference(
        a, perp_diff,
        alpha=alpha,
    )


# implementation of: https://arxiv.org/abs/2306.01708
# Special mode "TIES-SOUP" has been implemented by setting `vote_sgn` > 0.0
# Special mode "TIES-STOCK" has been implemented by setting `apply_stock` > 0.0
def add_difference_ties(
    base: RecipeNodeOrValue,
    *models: RecipeNodeOrValue,
    alpha: Parameter(Tensor) = 1.0,
    k: Parameter(float) = 1.0,
) -> recipe_nodes.RecipeNode:
    base = value_to_node(base)
    models = tuple(value_to_node(model) for model in models)

    models = tuple(
        subtract(model, base)
        if model.merge_space == "weight" else
        model
        for model in models
    )

    res = ties_sum(
        *models,
        k=k,
    )

    return add_difference(
        base, res,
        alpha=alpha,
    )


def add_difference_ties_extended(
    base: RecipeNodeOrValue,
    *models: RecipeNodeOrValue,
    alpha: Parameter(Tensor) = 1.0,
    k: Parameter(float) = 0.2,
    vote_sgn: Parameter(bool) = False,
    apply_stock: Parameter(bool) = False,
    cos_eps: Parameter(float) = 1e-6,
    apply_median: Parameter(bool) = False,
    eps: Parameter(float) = 1e-6,
    maxiter: Parameter(int) = 100,
    ftol: Parameter(float) = 1e-20,
) -> recipe_nodes.RecipeNode:
    base = value_to_node(base)
    models = tuple(value_to_node(model) for model in models)

    models = tuple(
        subtract(model, base)
        if model.merge_space == "weight" else
        model
        for model in models
    )

    res = ties_sum_extended(
        *models,
        k=k,
        vote_sgn=vote_sgn,
        apply_stock=apply_stock,
        cos_eps=cos_eps,
        apply_median=apply_median,
        eps=eps,
        maxiter=maxiter,
        ftol=ftol,
    )

    return add_difference(
        base, res,
        alpha=alpha,
    )


def copy_region(
    a: RecipeNodeOrValue, b: RecipeNodeOrValue, c: Optional[RecipeNodeOrValue] = None, *,
    width: Parameter(float) = 1.0,
    offset: Parameter(float) = 0.0,
    top_k: Parameter(bool) = False,
) -> recipe_nodes.RecipeNode:
    a = value_to_node(a)
    b = value_to_node(b)

    if c is not None:
        c = value_to_node(c)

        a = subtract(
            a, c,
        )
        b = subtract(
            b, c,
        )

    copy_method = [merge_methods.tensor_sum, merge_methods.top_k_tensor_sum][int(top_k)]
    res = copy_method(
        a, b,
        width=width,
        offset=offset,
    )

    if c is not None:
        res = merge_methods.add_difference(
            c, res,
            alpha=1.0,
        )

    return res


tensor_sum = copy_region


def rotate(
    a: RecipeNodeOrValue, b: RecipeNodeOrValue, c: Optional[RecipeNodeOrValue] = None, *,
    alignment: Parameter(float) = 1.0,
    alpha: Parameter(float) = 0.0,
    cache: Optional[Dict[str, torch.Tensor]] = None,
) -> recipe_nodes.RecipeNode:
    a = value_to_node(a)
    b = value_to_node(b)

    if c is not None:
        c = value_to_node(c)

        a = subtract(
            a, c,
        )
        b = subtract(
            b, c,
        )

    res = merge_methods.rotate(
        a, b,
        alignment=alignment,
        alpha=alpha,
    ).set_cache(cache)

    if c is not None:
        res = merge_methods.add_difference(
            c, res,
            alpha=1.0,
        )

    return res


def dropout(
    a: RecipeNodeOrValue,
    *models: RecipeNodeOrValue,
    probability: Parameter(float) = 0.9,
    alpha: Parameter(Tensor) = 0.5,
    overlap: Parameter(float) = 1.0,
    overlap_emphasis: Parameter(float) = 0.0,
    seed: Parameter(int) = None,
) -> recipe_nodes.RecipeNode:
    deltas = [
        subtract(model, a)
        for model in models
    ]
    ba_delta = merge_methods.dropout(*deltas, probability=probability, overlap=overlap, overlap_emphasis=overlap_emphasis, seed=seed)
    return merge_methods.add_difference(a, ba_delta, alpha=alpha)


ties_sum_with_dropout = merge_methods.ties_sum_with_dropout


# implementation of: https://arxiv.org/abs/2311.03099
# Notice that this is "TIES Merging w/ DARE", which is "Prune > Merge (TIES) > Rescale"
# See https://slgero.medium.com/merge-large-language-models-29897aeb1d1a for details
# mode "TIES-SOUP" has been implemented by setting `vote_sgn` > 0.0
def ties_with_dare(
    base: RecipeNodeOrValue,
    *models: RecipeNodeOrValue,
    della_eps: Parameter(float) = 0.0,
    probability: Parameter(float) = 0.9,
    rescale: Parameter(bool) = True,
    alpha: Parameter(Tensor) = 1.0,
    seed: Parameter(int) = None,
    k: Parameter(float) = 1.0,
    vote_sgn: Parameter(bool) = False,
    apply_stock: Parameter(bool) = False,
    cos_eps: Parameter(float) = 1e-6,
    apply_median: Parameter(bool) = False,
    eps: Parameter(float) = 1e-6,
    maxiter: Parameter(int) = 100,
    ftol: Parameter(float) = 1e-20,
) -> recipe_nodes.RecipeNode:
    base = value_to_node(base)
    models = tuple(value_to_node(model) for model in models)
    deltas = tuple(
        subtract(model, base)
        if model.merge_space == "weight" else
        model
        for model in models
    )

    res = ties_sum_with_dropout(
        *deltas,
        della_eps=della_eps,
        probability=probability,
        rescale=rescale,
        k=k,
        vote_sgn=vote_sgn,
        seed=seed,
        apply_stock=apply_stock,
        cos_eps=cos_eps,
        apply_median=apply_median,
        eps=eps,
        maxiter=maxiter,
        ftol=ftol,
    )

    return merge_methods.add_difference(base, res, alpha=alpha)


# Following mergekit's implementation of Model Stock (which official implementation doesn't exist)
# https://github.com/arcee-ai/mergekit/blob/main/mergekit/merge_methods/model_stock.py
def n_model_stock(
    base: RecipeNodeOrValue,
    *models: RecipeNodeOrValue,
    cos_eps: Parameter(float) = 1e-6,
) -> recipe_nodes.RecipeNode:
    base = value_to_node(base)
    models = tuple(value_to_node(model) for model in models)
    deltas = tuple(
        subtract(model, base)
        if model.merge_space == "weight" else
        model
        for model in models
    )

    res = model_stock(
        *deltas,
        cos_eps=cos_eps,
    )

    return merge_methods.add_difference(base, res, alpha=1.0)
