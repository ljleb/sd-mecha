from .recipe_merging import merge_and_save, open_input_dicts, infer_model_configs
from .recipe_serializer import serialize, deserialize, deserialize_path
from .streaming import StateDictKeyError
from .extensions.merge_methods import merge_method, value_to_node, RecipeNodeOrValue, Parameter, Return, StateDict
from .conversion import convert
from sd_mecha.extensions.builtin.merge_methods import (
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
    pick_component,
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
from .helpers import model, literal, serialize_and_save, Defaults, set_log_level
from . import recipe_nodes, extensions
import torch
# greedy load linalg module, see https://github.com/pytorch/pytorch/issues/90613
torch.linalg.inv(torch.ones((1, 1), device="cuda" if torch.cuda.is_available() else "cpu"))


def _load_builtin_extensions():
    import sd_mecha.extensions.builtin.model_configs
    import sd_mecha.extensions.builtin.lycoris


_load_builtin_extensions()
