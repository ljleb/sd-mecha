# User-Defined Merge Methods

While there are a multiple merge methods and conversion functions built into sd-mecha, sometimes there is a need for something other than what the builtins can offer.
This is often the case when experimenting with new methods or when a new foundational model is released and there comes a need to perform state dict operations on it.
To take advantage of the low memory usage sd-mecha offers in these cases, a public interface is defined through which it is possible to define custom merge methods.

## Custom Merge Method

Here is an example merge method that blends parameters with noise. Let's start with the code:

```python
# 1.
from torch import Tensor, randn, Generator
from sd_mecha import model, merge_and_save, merge_method, Parameter, Return


# 2.
@merge_method
def noise_sum(
    a: Parameter(Tensor),
    alpha: Parameter(Tensor) = 0.5,
    seed: Parameter(int) = None,
) -> Return(Tensor):
    gen = Generator()
    gen.manual_seed(seed)
    noise = randn(a.shape, generator=gen)

    return (1-alpha)*a + alpha*noise


# 3.
any_model = model("path/to/any_model.safetensors")
recipe = noise_sum(
    any_model,
    alpha=0.01,
    seed=42,
)
merge_and_save(recipe, "path/to/model_out.safetensors")
```

In order, we have:

1. imports
2. definition of our custom merge method `custom_sum`
3. usage of the merge method

Let's focus on the merge method definition.

In plain english, in this case, `noise_sum` is a merge method that accepts 3 parameters: `a`, `alpha` and `seed`.
When it is eventually called, `a` and `alpha` will be instances of `torch.Tensor`, and `seed` will be an `int`.
The method is expected to return an instance of `torch.Tensor`.

Applying the `@merge_method` decorator to `custom_sum` tells sd-mecha that the function is a merge method and should be converted to a recipe node constructor.
The decorator does this by replacing the function in place with an instance of `sd_mecha.extensions.merge_methods.MergeMethod`.
When called, the `MergeMethod` instantiates a `sd_mecha.recipe_nodes.MergeRecipeNode` (which holds a reference to the original `MergeMethod` object) instead of actually calling the function.
As shown in 3., we can pass this new recipe node object to `merge_and_save` or `serialize_and_save`. This will finally call the real function on the right inputs as many times as needed to complete the planned task.

In general, certain rules must be followed to define a custom merge method:

- The type of each function parameter needs to be `sd_mecha.Parameter(type, ...)`. This type is used to specify additional metadata for a recipe node parameter.
    In particular, additional keyword arguments can be passed to `Parameter(...)` if a parameter needs to:
    - receive a specific model config (`model_config=...`) or
    - constrain its merge space (`merge_space=...`)
    This can be useful in different contexts and for different reasons, all covered below.
- The return type needs to be `sd_mecha.Return(type, ...)`. This type is used similarly to `Parameter(...)`.
    However, it differs slightly in what it can receive as arguments:
    - the type cannot be `sd_mecha.StateDict[...]`
    - the merge space has to be either a single merge space or an instance of `sd_mecha.MergeSpaceSymbol`
    Otherwise it is used in the same way as `sd_mecha.Parameter(...)` by `@merge_method` to instantiate the method object
- Optionally, `**kwargs` can be added to the method. If present, the merge method will receive these additional parameters when called:
    - `key`: the name of the target key currently being processed.
    This is the name of the key from the model config for which the merge method is returning values.
    - `cache`: either a dict or `None`. When it is a dict, the reference will be the same for different calls to `merge_and_save` on the same recipe.
    This can sometimes be used to save intermediate results that can be reused even if some of the parameters change.
    Sometimes, reusing intermediate results can significantly accelerate a merge method when testing multiple different values for some parameters.

The types that can be used as input to a merge method are:

- `torch.Tensor`
- `float`
- `int`
- `bool`
- `str`
- an instance of `typing.TypeVar`
- `sd_mecha.StateDict[T]` where `T` is any of the above types (excluding `StateDict` itself)

When a parameter is of type `sd_mecha.StateDict[T]`, instead of receiving a value from an input state dict, it will be an instance of a subclass of `sd_mecha.StateDict`.
While it looks like we are materializing the entire input dict for each output key to be merged, in fact `sd_mecha.StateDict` is a dict-like interface that loads tensors from disk or other sources on demand.
When the input config is the same as the output config, we can simply use `state_dict[kwargs["key"]]` to materialize the expected key from the input.
When the input config is not the same as the output config, we can list all keys from the associated input config using `state_dict.keys()`.

## Custom Blocks Config

Block merging basics are covered in [Typical Use Cases > Blocks Merging (MBW)](../1-typical-use-cases#blocks-merging-mbw).

To define a custom block config, two things need to be defined:

1. A model config to hold the block keys. This is required to enable conversion and serialization
2. A conversion function that maps each block to a key

This is an example config that splits SDXL in two blocks (a specific key vs the rest of the model):

```python
from sd_mecha.extensions import model_configs
model_configs.register(model_configs.from_yaml("""identifier: sdxl-my_blocks
components:
  blocks:
    the_key_we_want: {shape: [], dtype: float32}
    rest: {shape: [], dtype: float32}
"""))


from typing import TypeVar
T = TypeVar("T")


from sd_mecha import merge_method, Parameter, Return, StateDict

@merge_method(is_conversion=True)
def convert_blocks_to_sdxl(
    blocks_dict: Parameter(StateDict[T], "sdxl-my_blocks"),
    **kwargs,
) -> Return(T, "sdxl-sgm"):
    sdxl_key = kwargs["key"]
    if sdxl_key == "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.attn2.to_v.weight":
        return blocks_dict["the_key_we_want"]
    else:
        return blocks_dict["rest"]


# we can then use the conversion like this:
from sd_mecha import model, convert, weighted_sum
a = model("path/to/model_a.safetensors")
b = model("path/to/model_b.safetensors")

blocks = {
    "the_key_we_want": 1.0,
    "rest": 0.0,
}
blocks = convert(blocks, a)  # <- the conversion method is called here

recipe = weighted_sum(a, b, alpha=blocks)
```

Using `is_conversion=True` tells sd-mecha to register the merge method as a conversion method.
This makes the merge method a candidate transition between two model configs (in this case "sdxl-my_blocks" -> "sdxl-sgm") when looking for a conversion path using `sd_mecha.convert()`

As an alternative, we can still create the recipe node by calling the conversion method directly:

```python
# ...
blocks = convert_blocks_to_sdxl({
    "the_key_we_want": 1.0,
    "rest": 0.0,
})
```

Usually, `sd_mecha.convert` is preferred because it simplifies finding the right conversion method implicitly.
