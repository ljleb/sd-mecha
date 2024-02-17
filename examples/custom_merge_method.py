import sd_mecha
import torch
from sd_mecha.extensions import convert_to_recipe, LiftFlag, MergeSpace
sd_mecha.set_log_level()


# define a custom merge method
# `@convert_to_recipe` converts the annotated function to work with the merge recipe API
@convert_to_recipe
def custom_sum(
    # All positional arguments are model layers that share the same model key
    # We use a trick to include the merge space within the type system: `Tensor | LiftFlag[MergeSpace...]`
    # We can tell `@convert_to_recipe` that a specific set of models must share the same merge space using a TypeVar:
    #    ```
    #    SharedSpace = TypeVar("SharedSpace", bound=LiftFlag[MergeSpace.MODEL | MergeSpace.DELTA])
    #    ...
    #    def my_method(
    #        a: torch.Tensor | SharedSpace,
    #        b: torch.Tensor | SharedSpace,
    #        **kwargs,
    #    ) -> torch.Tensor | SharedSpace
    #    ```
    # For more examples, see /sd_mecha/merge_methods.py
    a: torch.Tensor | LiftFlag[MergeSpace.MODEL],
    b: torch.Tensor | LiftFlag[MergeSpace.MODEL],
    *,
    # hyperparameters go here
    alpha: float = 0.5,  # default arguments are honored
    beta: float,
    # `@convert_to_recipe` introduces additional kwargs `device=` and `dtype=`
    # so we must put `**kwargs` to satisfy the type system
    **kwargs,
) -> torch.Tensor | LiftFlag[MergeSpace.MODEL]:

    # to use an existing `@convert_to_recipe` merge method inside another one (i.e. this one),
    #  we use the `__wrapped__` attribute that returns the original unwrapped function
    weighted_sum = sd_mecha.weighted_sum.__wrapped__(a, b, alpha=alpha)

    # in this example we add noise to the sum
    return (1 - beta) * weighted_sum + beta * torch.randn_like(weighted_sum)


# plan our custom weighted sum
recipe = custom_sum(
    # put models as positional arguments
    "ghostmix_v20Bakedvae",
    "dreamshaper_332BakedVaeClipFix",
    # put hyperparameters as keyword only arguments
    alpha=0.5,
    beta=0.1,
    # these parameters are added by `@convert_to_recipe`:
    device="cuda",        # we can change the device just before calling the function
    dtype=torch.float64,  # and the dtype too
)

scheduler = sd_mecha.RecipeMerger(
    base_dir=r"E:\sd\models\Stable-diffusion",
)

# perform the entire merge plan and save to output path
scheduler.merge_and_save(recipe, output_path="basic_merge")
