import logging
import pathlib
import torch
from typing import Optional, Dict
from sd_mecha.extensions.merge_space import MergeSpace
from sd_mecha.extensions.model_config import ModelConfig
from sd_mecha.recipe_merger import RecipeMerger
from sd_mecha import recipe_nodes, merge_methods, extensions
from sd_mecha.extensions.merge_method import convert_to_recipe, value_to_node, RecipeNodeOrValue, Parameter, Return, StateDict
from sd_mecha.recipe_serializer import serialize, deserialize, deserialize_path
from sd_mecha.merge_methods import (
    weighted_sum,
    slerp,
    n_average,
    geometric_median,
    subtract,
    perpendicular_component,
    geometric_sum,
    train_difference_mask,
    add_opposite_mask,
    add_strict_opposite_mask,
    add_cosine_a,
    add_cosine_b,
    ties_sum,
    ties_sum_extended,
    crossover,
    clamp,
    model_stock,
    fallback,
)


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


def add_difference(
    a: RecipeNodeOrValue, b: RecipeNodeOrValue, c: Optional[RecipeNodeOrValue] = None, *,
    alpha: float = 1.0,
    clamp_to_ab: Optional[bool] = None,
) -> recipe_nodes.RecipeNode:
    a = extensions.merge_method.value_to_node(a)
    b = extensions.merge_method.value_to_node(b)
    original_b = b

    if c is not None:
        c = extensions.merge_method.value_to_node(c)
        b = subtract(
            b, c,
        )

    res = merge_methods.add_difference(
        a, b,
        alpha=alpha,
    )

    if a.merge_space == original_b.merge_space:
        b = original_b

    if clamp_to_ab is None:
        clamp_to_ab = a.merge_space == b.merge_space

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
        cache=cache,
    )

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
def model_stock_n_models(
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


def model(path: str | pathlib.Path, model_config: Optional[str] = None):
    if isinstance(path, str):
        path = pathlib.Path(path)
    return recipe_nodes.ModelRecipeNode(path, model_config)


def set_log_level(level: str = "INFO"):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)
