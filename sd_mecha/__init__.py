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
    ties_sum_with_dropout,
    crossover,
    clamp,
    model_stock,
    fallback,
    cast,
    get_dtype,
    get_device,
    pick_component,
    omit_component,
    exchange_ema,
    stack,
    sdxl_sgm_split_rebasin,
    sdxl_sgm_split_randperm,
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
from .helpers import model, literal, Defaults, set_log_level
from . import recipe_nodes, extensions
