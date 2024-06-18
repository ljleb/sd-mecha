import sd_mecha
import torch
from sd_mecha.extensions.merge_method import convert_to_recipe, LiftFlag, MergeSpace
from sd_mecha import Hyper
sd_mecha.set_log_level()


# define a custom merge method
# `@convert_to_recipe` converts the annotated function to work with the merge recipe API
@convert_to_recipe
def custom_sum(
    # Each positional argument is either a weight or bias
    # Merge methods are called once for each key that all input models have in common
    # We use a type system trick to specify the expected merge space of each model: `Tensor | LiftFlag[MergeSpace...]`
    # We can represent complex constraints where multiple models must be in the exact same merge space using a TypeVar:
    #    ```
    #    SameSpace = TypeVar("SharedSpace", bound=LiftFlag[MergeSpace.BASE | MergeSpace.DELTA])
    #    ...
    #    def my_method(
    #        a: torch.Tensor | SameSpace,
    #        b: torch.Tensor | SameSpace,
    #        **kwargs,
    #    ) -> torch.Tensor | SameSpace
    #    ```
    # In this code, `a` and `b` must be in the same space, either in BASE space or DELTA space.
    # The return merge space is exactly the merge space that satisfies `a` and `b` at the same time.
    # For more examples, see /sd_mecha/merge_methods.py
    a: torch.Tensor | LiftFlag[MergeSpace.BASE],
    b: torch.Tensor | LiftFlag[MergeSpace.BASE],
    *,
    # hyperparameters go here
    # `Hyper` is an union type of `float`, `int` and `dict` (the dict case is for a different weight per block MBW), which is what the caller of the method excpects.
    # in practice the method only ever receives numeric types (int or float), so no need to worry about the dict case
    alpha: Hyper = 0.5,  # default arguments are honored
    beta: Hyper,
    # `@convert_to_recipe` introduces additional kwargs: `device=` and `dtype=`
    # We must put `**kwargs` to satisfy the type system:
    **kwargs,
) -> torch.Tensor | LiftFlag[MergeSpace.BASE]:

    # to call an existing `@convert_to_recipe` merge method inside another one (i.e. this one),
    #  we use the `__wrapped__` attribute that returns the original unwrapped function
    weighted_sum = sd_mecha.weighted_sum.__wrapped__(a, b, alpha=alpha)

    # in this example we add noise to the sum
    return (1 - beta) * weighted_sum + beta * torch.randn_like(weighted_sum)


# plan our custom weighted sum
recipe = custom_sum(
    # put models as positional arguments
    "ghostmix_v20Bakedvae",
    "dreamshaper_332BakedVaeClipFix",
    # put hyperparameters as keyword arguments
    alpha=0.5,
    beta=0.1,
    # these parameters are added by `@convert_to_recipe`:
    device="cuda",        # we can change the device just before calling the function
    dtype=torch.float64,  # same for dtype
)

merger = sd_mecha.RecipeMerger(
    models_dir=r"E:\sd\models\Stable-diffusion",
)

# perform the entire merge plan and save to output path
merger.merge_and_save(recipe, output="basic_merge")
