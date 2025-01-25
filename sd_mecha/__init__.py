from .recipe_merger import RecipeMerger
from .recipe_serializer import serialize, deserialize, deserialize_path
from .streaming import StateDictKeyError
from .extensions.merge_method import recipe, value_to_node, RecipeNodeOrValue, Parameter, Return, StateDict
from .conversion import convert
from .merge_methods import (
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
from .merge_method_wrappers import (
    add_difference,
    add_perpendicular,
    add_difference_ties,
    add_difference_ties_extended,
    copy_region,
    tensor_sum,
    rotate,
    dropout,
    ties_with_dare,
    n_model_stock,
)
from .helpers import model, literal, set_log_level, serialize_and_save
from . import recipe_nodes, merge_methods, extensions


def _load_builtin_extensions():
    import sd_mecha.extensions.builtin.model_configs
    import sd_mecha.extensions.builtin.lycoris


_load_builtin_extensions()
