import logging
import pathlib
import torch
from torch import Tensor

import sd_mecha.builtin_model_archs
import sd_mecha.builtin_model_types
from typing import Optional, Dict, Mapping
from sd_mecha.recipe_merger import RecipeMerger
from sd_mecha import recipe_nodes, merge_methods, extensions
from sd_mecha.extensions.merge_method import RecipeNodeOrPath, path_to_node
from sd_mecha.recipe_nodes import MergeSpace
from sd_mecha.hypers import Hyper, blocks, default
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
n_average = merge_methods.n_average


def add_difference(
    a: RecipeNodeOrPath, b: RecipeNodeOrPath, c: Optional[RecipeNodeOrPath] = None, *,
    alpha: Hyper = 1.0,
    clamp_to_ab: Optional[bool] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> recipe_nodes.RecipeNode:
    a: recipe_nodes.RecipeNode = extensions.merge_method.path_to_node(a)
    b: recipe_nodes.RecipeNode = extensions.merge_method.path_to_node(b)
    original_b = b

    if c is not None:
        c = extensions.merge_method.path_to_node(c)
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

    if clamp_to_ab is None:
        clamp_to_ab = a.merge_space == b.merge_space

    if clamp_to_ab:
        if a.merge_space != b.merge_space:
            raise TypeError(f"Merge space of A {a.merge_space} and B {b.merge_space} must be the same to clamp the merge.")
        res = clamp(
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
train_difference = merge_methods.train_difference
cosine_add_a = merge_methods.add_cosine_a
cosine_add_b = merge_methods.add_cosine_b
ties_sum = merge_methods.ties_sum
ties_sum_extended = merge_methods.ties_sum_extended


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
    base: RecipeNodeOrPath,
    *models: RecipeNodeOrPath,
    alpha: Hyper,
    k: Hyper = 0.2,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> recipe_nodes.RecipeNode:
    # $$ \{\theta_{init}\}_{t=1}^n $$
    base = path_to_node(base)
    models = tuple(path_to_node(model) for model in models)

    # Create task vectors.
    # $$ \tau_t $$
    models = tuple(
        subtract(model, base)
        if model.merge_space is MergeSpace.BASE else
        model
        for model in models
    )

    # step 1 + step 2 + step 3
    res = ties_sum(
        *models,
        k=k,
        device=device,
        dtype=dtype,
    )

    # Obtain merged checkpoint

    # $$ \theta_{init} + \lambda * \tau_m $$
    return add_difference(
        base, res,
        alpha=alpha,
        device=device,
        dtype=dtype,
    )


def add_difference_ties_extended(
    base: RecipeNodeOrPath,
    *models: RecipeNodeOrPath,
    alpha: Hyper,
    k: Hyper = 0.2,
    vote_sgn: Hyper = 0.0,
    apply_stock: Hyper = 0.0,
    cos_eps: Hyper = 1e-6,
    apply_median: Hyper = 0.0,
    eps: Hyper = 1e-6,
    maxiter: Hyper = 100,
    ftol: Hyper =1e-20,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> recipe_nodes.RecipeNode:
    # $$ \{\theta_{init}\}_{t=1}^n $$
    base = path_to_node(base)
    models = tuple(path_to_node(model) for model in models)

    # Create task vectors.
    # $$ \tau_t $$
    models = tuple(
        subtract(model, base)
        if model.merge_space is MergeSpace.BASE else
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
        device=device,
        dtype=dtype,
    )

    # Obtain merged checkpoint

    # $$ \theta_{init} + \lambda * \tau_m $$
    return add_difference(
        base, res,
        alpha=alpha,
        device=device,
        dtype=dtype,
    )


def copy_region(
    a: RecipeNodeOrPath, b: RecipeNodeOrPath, c: Optional[RecipeNodeOrPath] = None, *,
    width: Hyper = 1.0,
    offset: Hyper = 0.0,
    top_k: bool = False,
    device: Optional[str] = "cuda",
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

    copy_method = [merge_methods.tensor_sum, merge_methods.top_k_tensor_sum][int(top_k)]
    res = copy_method(
        a, b,
        width=width,
        offset=offset,
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
    alignment: Hyper = 1.0,
    alpha: Hyper = 0.0,
    device: Optional[str] = "cuda",
    dtype: Optional[torch.dtype] = None,
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
        alignment=alignment,
        alpha=alpha,
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


clamp = merge_methods.clamp


def dropout(
    a: RecipeNodeOrPath,
    *models: RecipeNodeOrPath,
    probability: Hyper = 0.9,
    alpha: Hyper = 0.5,
    overlap: Hyper = 1.0,
    overlap_emphasis: Hyper = 0.0,
    seed: Optional[Hyper] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> recipe_nodes.RecipeNode:
    deltas = [
        subtract(model, a)
        for model in models
    ]
    ba_delta = merge_methods.dropout(*deltas, probability=probability, overlap=overlap, overlap_skew=overlap_emphasis, seed=seed, device=device, dtype=dtype)
    return sd_mecha.add_difference(a, ba_delta, alpha=alpha, device=device, dtype=dtype)


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
    base: RecipeNodeOrPath,
    *models: RecipeNodeOrPath,
    probability: Hyper = 0.9,
    rescale: Hyper = 1.0,
    alpha: Hyper = 0.5,
    seed: Optional[Hyper] = None,
    k: Hyper = 0.2,
    vote_sgn: Hyper = 0.0,
    apply_stock: Hyper = 0.0,
    cos_eps: Hyper = 1e-6,
    apply_median: Hyper = 0.0,
    eps: Hyper = 1e-6,    
    maxiter: Hyper = 100, 
    ftol: Hyper = 1e-20,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> recipe_nodes.RecipeNode:
    # $$ \delta^t = \theta_{SFT}^{t} - \theta_{PRE} \in \mathbb{R}^d $$
    base = path_to_node(base)
    models = tuple(path_to_node(model) for model in models)
    deltas = tuple(
        subtract(model, base)
        if model.merge_space is MergeSpace.BASE else
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
        device=device, 
        dtype=dtype
    )

    # $$ \theta_M = \theta_{PRE} + \lambda \cdot \Sigma_{k=1}^{K} \tilde{\delta}^{t_k} $$
    return sd_mecha.add_difference(base, res, alpha=alpha, device=device, dtype=dtype)


model_stock_for_tensor = merge_methods.model_stock_for_tensor


# Following mergekit's implementation of Model Stock (which official implementation doesn't exist)
# https://github.com/arcee-ai/mergekit/blob/main/mergekit/merge_methods/model_stock.py
def model_stock_n_models(
    base: RecipeNodeOrPath,
    *models: RecipeNodeOrPath,    
    cos_eps: Hyper = 1e-6,    
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> recipe_nodes.RecipeNode:

    base = path_to_node(base)
    models = tuple(path_to_node(model) for model in models)
    deltas = tuple(
        subtract(model, base)
        if model.merge_space is MergeSpace.BASE else
        model
        for model in models
    )

    # This is hacky: Both w_avg and w_h will be calculated there.    
    # Notice that t and cos_theta is vector instead of single value.
    # Conceptually it could compatable with TIES, but algorithm should be rewritten.
    res = model_stock_for_tensor(
        *deltas,
        cos_eps=cos_eps,
        device=device, 
        dtype=dtype
    )

    return sd_mecha.add_difference(base, res, alpha=1.0, device=device, dtype=dtype)


geometric_median = merge_methods.geometric_median


def model(state_dict: str | pathlib.Path | Mapping[str, Tensor], model_arch: str = "sd1", model_type: str = "base"):
    return recipe_nodes.ModelRecipeNode(state_dict, model_arch, model_type)


def lora(state_dict: str | pathlib.Path | Mapping[str, Tensor], model_arch: str = "sd1"):
    return recipe_nodes.ModelRecipeNode(state_dict, model_arch, "lora")


def parameter(name: str, merge_space: MergeSpace, model_arch: Optional[str] = None):
    return recipe_nodes.ParameterRecipeNode(name, merge_space, model_arch)


def set_log_level(level: str = "INFO"):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)
