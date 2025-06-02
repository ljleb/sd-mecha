import sd_mecha
import torch
from torch import Tensor
from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return


# sets loglevel to INFO. some operations will report extra detail through stdout/stderr
sd_mecha.set_log_level()


# define a custom merge method
# `@merge_method` converts the decorated function to work with the merge method API
@merge_method
def custom_sum(
    # Each positional argument is a single tensor from one of the input models.
    # Merge methods are called once for each key that all input models have in common.
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    alpha: Parameter(Tensor) = 0.5,  # params with a default value are automatically in "param" merge space
    *,
    beta: Parameter(Tensor, merge_space="param"),
    # extra info or metadata is passed in **kwargs if it is present
    # this includes the name of the key currently being merged, a cache mechanism, and a few more things
    # add **kwargs to receive this extra information:`
    **kwargs,
) -> Return(Tensor):
    weighted_sum = (1-alpha)*a + alpha*b

    # just for the sake of the example, let's add noise to the sum
    return (1 - beta) * weighted_sum + beta * torch.randn_like(weighted_sum)


# plan our custom weighted sum
recipe = custom_sum(
    # this merge uses the "sd1-ldm" model config.
    # the config is automatically inferred, there is no need to pass its identifier here
    sd_mecha.model("ghostmix_v20Bakedvae.safetensors"),
    sd_mecha.model("dreamshaper_332BakedVaeClipFix.safetensors"),
    # merge params, literal values (int, float, str, bool, None) are broadcasted to the entire state dict
    alpha=0.6,
    beta=0.1,
)

# perform the entire merge plan and save to output path
sd_mecha.merge(recipe, output="custom_merge.safetensors")
