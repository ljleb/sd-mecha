def _load_builtin_extensions():
    import sd_mecha.extensions.builtin.model_configs
    import sd_mecha.extensions.builtin.lycoris


_load_builtin_extensions()


from .merging import merge, open_input_dicts, infer_model_configs
from .serialization import serialize, deserialize, deserialize_path
from .streaming import StateDictKeyError
from .extensions.merge_methods import merge_method, value_to_node, RecipeNodeOrValue, Parameter, Return, StateDict
from .conversion import convert
from sd_mecha.extensions.builtin.merge_methods import (
    clamp,
    add_cosine_a,
    add_cosine_b,
    crossover,
    exchange_ema,
    weighted_sum,
    n_average,
    slerp,
    subtract,
    perpendicular_component,
    train_difference_mask,
    add_opposite_mask,
    add_strict_opposite_mask,
    geometric_sum,
    multiply_quotient,
    fallback,
    cast,
    get_dtype,
    get_device,
    pick_component,
    omit_component,
    cast_dtype_map,
    cast_dtype_map_reversed,
    stack,
    top_k_tensor_sum,
    truncate_rank,
    ties_sum_with_dropout,
    ties_sum,
    ties_sum_extended,
    model_stock,
    geometric_median,
    add_difference,
    add_perpendicular,
    add_difference_ties,
    add_difference_ties_extended,
    copy_region,
    tensor_sum,
    rotate,
    dropout,
    add_ties_with_dare,
    n_model_stock,
)
from .helpers import model, literal, Defaults, set_log_level, skip_key
from . import recipe_nodes, extensions
