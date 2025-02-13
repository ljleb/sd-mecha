import torch
from typing import Optional, Dict
from .extensions.merge_methods import value_to_node, RecipeNodeOrValue
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
    alpha: float = 1.0,
    clamp_to_ab: bool = False,
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

    return res


def add_perpendicular(
    a: RecipeNodeOrValue, b: RecipeNodeOrValue, c: RecipeNodeOrValue, *,
    alpha: float = 1.0,
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


# latex notes in reference to original implementation: https://arxiv.org/abs/2306.01708
# - `base`: $$ \theta_{init} $$
# - `*models`: $$ \{\theta_{init}\}_{t=1}^n $$
# - `models` after `subtract`: $$ \tau_t $$
# - `alpha`: $$ \lambda $$
# - `k`: $$ k $$ ( From $$ \% $$ to $$ 1 $$ )
# - `res`: $$ \lambda * \tau_m $$
# - `return`: $$ \theta_m $$
# Special mode "TIES-SOUP" has been implemented by setting `vote_sgn` > 0.0
# Special mode "TIES-STOCK" has been implemented by setting `apply_stock` > 0.0
def add_difference_ties(
    base: RecipeNodeOrValue,
    *models: RecipeNodeOrValue,
    alpha: float,
    k: float = 0.2,
) -> recipe_nodes.RecipeNode:
    # $$ \{\theta_{init}\}_{t=1}^n $$
    base = value_to_node(base)
    models = tuple(value_to_node(model) for model in models)

    # Create task vectors.
    # $$ \tau_t $$
    models = tuple(
        subtract(model, base)
        if model.merge_space == "weight" else
        model
        for model in models
    )

    # step 1 + step 2 + step 3
    res = ties_sum(
        *models,
        k=k,
    )

    # Obtain merged checkpoint

    # $$ \theta_{init} + \lambda * \tau_m $$
    return add_difference(
        base, res,
        alpha=alpha,
    )


def add_difference_ties_extended(
    base: RecipeNodeOrValue,
    *models: RecipeNodeOrValue,
    alpha: float = 1.0,
    k: float = 0.2,
    vote_sgn: float = 0.0,
    apply_stock: float = 0.0,
    cos_eps: float = 1e-6,
    apply_median: float = 0.0,
    eps: float = 1e-6,
    maxiter: float = 100,
    ftol: float =1e-20,
) -> recipe_nodes.RecipeNode:
    # $$ \{\theta_{init}\}_{t=1}^n $$
    base = value_to_node(base)
    models = tuple(value_to_node(model) for model in models)

    # Create task vectors.
    # $$ \tau_t $$
    models = tuple(
        subtract(model, base)
        if model.merge_space == "weight" else
        model
        for model in models
    )

    # step 1 + step 2 + step 3
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

    # Obtain merged checkpoint

    # $$ \theta_{init} + \lambda * \tau_m $$
    return add_difference(
        base, res,
        alpha=alpha,
    )


def copy_region(
    a: RecipeNodeOrValue, b: RecipeNodeOrValue, c: Optional[RecipeNodeOrValue] = None, *,
    width: float = 1.0,
    offset: float = 0.0,
    top_k: bool = False,
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
    alignment: float = 1.0,
    alpha: float = 0.0,
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
    probability: float = 0.9,
    alpha: float = 0.5,
    overlap: float = 1.0,
    overlap_emphasis: float = 0.0,
    seed: Optional[float] = None,
) -> recipe_nodes.RecipeNode:
    deltas = [
        subtract(model, a)
        for model in models
    ]
    ba_delta = merge_methods.dropout(*deltas, probability=probability, overlap=overlap, overlap_skew=overlap_emphasis, seed=seed)
    return merge_methods.add_difference(a, ba_delta, alpha=alpha)


ties_sum_with_dropout = merge_methods.ties_sum_with_dropout


# latex notes in reference to original implementation: https://arxiv.org/abs/2311.03099
# Notice that this is "TIES Merging w/ DARE", which is "Prune > Merge (TIES) > Rescale"
# See https://slgero.medium.com/merge-large-language-models-29897aeb1d1a for details
# - `base`: $$ \theta_{PRE} $$
# - `*models`: $$ \theta_{SFT}^{t_k} $$
# - `deltas`: $$ \delta^t = \theta_{SFT}^{t} - \theta_{PRE} \in \mathbb{R}^d $$
# - `probability`: $$ p $$
# - `res`: $$ \hat{\delta}^t = \tilde{\delta}^t / (1-p) $$
# - `alpha`: $$ \lambda $$
# - `k`: $$ k $$ ( From $$ \% $$ to $$ 1 $$ ) in TIES paper
# - `return`: $$ \theta_M = \theta_{PRE} + \lambda \cdot \Sigma_{k=1}^{K} \tilde{\delta}^{t_k} $$
# Special mode "TIES-SOUP" has been implemented by setting `vote_sgn` > 0.0
def ties_with_dare(
    base: RecipeNodeOrValue,
    *models: RecipeNodeOrValue,
    probability: float = 0.9,
    rescale: float = 1.0,
    alpha: float = 0.5,
    seed: float = -1,
    k: float = 0.2,
    vote_sgn: float = 0.0,
    apply_stock: float = 0.0,
    cos_eps: float = 1e-6,
    apply_median: float = 0.0,
    eps: float = 1e-6,
    maxiter: float = 100,
    ftol: float = 1e-20,
) -> recipe_nodes.RecipeNode:
    # $$ \delta^t = \theta_{SFT}^{t} - \theta_{PRE} \in \mathbb{R}^d $$
    base = value_to_node(base)
    models = tuple(value_to_node(model) for model in models)
    deltas = tuple(
        subtract(model, base)
        if model.merge_space == "weight" else
        model
        for model in models
    )

    # $$ \tilde{\delta}^{t_k} $$
    res = ties_sum_with_dropout(
        *deltas,
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

    # $$ \theta_M = \theta_{PRE} + \lambda \cdot \Sigma_{k=1}^{K} \tilde{\delta}^{t_k} $$
    return merge_methods.add_difference(base, res, alpha=alpha)


# Following mergekit's implementation of Model Stock (which official implementation doesn't exist)
# https://github.com/arcee-ai/mergekit/blob/main/mergekit/merge_methods/model_stock.py
def n_model_stock(
    base: RecipeNodeOrValue,
    *models: RecipeNodeOrValue,
    cos_eps: float = 1e-6,
) -> recipe_nodes.RecipeNode:
    base = value_to_node(base)
    models = tuple(value_to_node(model) for model in models)
    deltas = tuple(
        subtract(model, base)
        if model.merge_space == "weight" else
        model
        for model in models
    )

    # This is hacky: Both w_avg and w_h will be calculated there.
    # Notice that t and cos_theta is vector instead of single value.
    # Conceptually it could compatable with TIES, but algorithm should be rewritten.
    res = model_stock(
        *deltas,
        cos_eps=cos_eps,
    )

    return merge_methods.add_difference(base, res, alpha=1.0)
