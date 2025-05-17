# User-Defined Merge Methods

While there are a multiple merge methods and conversion functions built into sd-mecha, sometimes there is a need for something other than what the builtins can offer.
This is often the case when experimenting with new methods or when a new foundational model is released and there comes a need to convert it to different formats.
To take advantage of the low memory usage of sd-mecha in these cases, the library exposes an interface through which it is possible to extend its capabilities.

## Custom Merge Methods

We can create new merge methods without modifying the core library.

To illustrate this, here is an example merge method `noise_sum` that blends parameters with noise:

```python
# 1. definition of our custom merge method `noise_sum`
from sd_mecha import merge_method, Parameter, Return
from torch import Tensor, randn, Generator

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


# 2. usage of the merge method
from sd_mecha import model, merge

any_model = model("path/to/any_model.safetensors")
recipe = noise_sum(any_model, alpha=0.01, seed=42)
merge(recipe, output="path/to/model_out.safetensors")
```

Let's focus on the merge method definition.

In plain english, `noise_sum` is a merge method that accepts 3 parameters: `a`, `alpha` and `seed`.
`alpha` has a default value of `0.5` (which gets implicitly converted to a tensor if used) and `seed` has a default value of `None`.
When the function is eventually called, `a` and `alpha` will be instances of `torch.Tensor`, and `seed` will be an `Optional[int]`.
The method is expected to return an instance of `torch.Tensor`.

Applying the `@merge_method` decorator to `def noise_sum` tells sd-mecha that the function is a merge method and should be converted to a recipe node constructor.
The decorator does this by replacing the function with an instance of `sd_mecha.extensions.merge_methods.MergeMethod`, which defines a `__call__` method.
When called, this `MergeMethod` object instantiates a `sd_mecha.recipe_nodes.MergeRecipeNode` (which holds a reference to the original `MergeMethod` object) instead of actually calling `noise_sum`.
As shown in 2. above, we can pass recipe node objects created by this merge method to `sd_mecha.merge` or other functions like `sd_mecha.serialize`.
This will in turn call the original undecorated function on the right inputs as many times as needed to complete the planned task.

In general, certain rules must be followed to define a custom merge method:

- The type of each function parameter needs to be `sd_mecha.Parameter(interface, merge_space=..., model_config=...)`. This type is used to specify additional metadata for a merge method parameter.
    In particular, additional keyword arguments can be passed to `Parameter(...)` if a parameter is to be required to:

    - receive a specific model config (`model_config=...`)
    - constrain its merge space to a set of merge spaces or an instance of `sd_mecha.MergeSpaceSymbol` (`merge_space=...`)

    This can be useful for different reasons, all covered below.

- The return type needs to be `sd_mecha.Return(interface, ...)`. This type is used similarly to `Parameter(...)`.
    However, it differs slightly in its arguments:

    - `interface` cannot be `sd_mecha.StateDict[...]`
    - the merge space has to be either a single merge space or an instance of `sd_mecha.MergeSpaceSymbol`
    - no changes for `model_config=...`, however it is worth mentioning that it is still a valid argument

- Optionally, `**kwargs` can be added to the arguments list. If present, the merge method will receive these additional parameters when called:

    - `key`: the name of the target key currently being processed.
    This is the name of the key from the target model config for which the merge method is expected to return a value.
    - `cache`: either a dict or `None`. When it is a dict, the reference will be the same for different calls to `merge` for each key to be processed. This is also true when the recipe graph is reused.
    This is useful when there is an opportunity to save intermediate results that can be reused across calls to the merge method with different parameters or target keys.
    Sometimes, reusing intermediate results can significantly accelerate a merge method.

The types that can be used as input to a merge method are:

- `torch.Tensor`
- `float`
- `int`
- `bool`
- `str`
- an instance of `typing.TypeVar` optionally bound by one or more of the above types
- `sd_mecha.StateDict[T]` where `T` is any of the above types

When a parameter is of type `sd_mecha.StateDict[T]`, instead of receiving a tensor from an input state dict, it will be an instance of a subclass of `sd_mecha.StateDict`.
`StateDict` is a dict-like interface that will load tensors from disk or other sources on demand.
When the config of a state dict parameter is the same as the target config, we can simply use `state_dict[kwargs["key"]]` to materialize the target key from the state dict.
When the config of a state dict parameter is not the same as the target config, we can list all keys in the state dict using `state_dict.keys()` and then chose which one to load to derive a value for the target key.

## Custom Blocks Config

Block merging basics are covered in [Typical Use Cases > Blocks Merging (MBW)](1-typical-use-cases.md#blocks-merging-mbw).

To define a custom block config, two things need to be defined:

1. The name of each block, in a model config. This is required to enable conversion and serialization
2. A conversion function that maps each block to a key in an existing model config

This is an example that splits SDXL in two "blocks" (a specific key vs the rest of the model):

```python
# 1. define our custom blocks config
from sd_mecha.extensions import model_configs

my_blocks_config = model_configs.from_yaml("""
identifier: sdxl-my_blocks
components:
  blocks:
    the_key_we_want: {shape: [], dtype: float32}
    rest: {shape: [], dtype: float32}
""")

model_configs.register(my_blocks_config)


# 2. define a conversion merge method from our blocks config to sdxl-sgm
from sd_mecha import merge_method, Parameter, Return, StateDict
from typing import TypeVar

T = TypeVar("T")

@merge_method(is_conversion=True)
def convert_my_blocks_to_sdxl(
    blocks_dict: Parameter(StateDict[T], model_config=my_blocks_config),
    **kwargs,
) -> Return(T, "sdxl-sgm"):
    sdxl_key = kwargs["key"]
    if sdxl_key == "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.attn2.to_v.weight":
        return blocks_dict["the_key_we_want"]
    else:
        return blocks_dict["rest"]


# 3. use the conversion method
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

Passing `is_conversion=True` to `@merge_method` tells sd-mecha to register the merge method as a conversion candidate.
This makes the merge method a possible transition between two model configs (in this case `sdxl-my_blocks` -> `sdxl-sgm`) when looking for a conversion path using `sd_mecha.convert()`.

As an alternative, we can still instantiate a recipe node by calling the conversion method directly:

```python
# ...
blocks = convert_my_blocks_to_sdxl({
    "the_key_we_want": 1.0,
    "rest": 0.0,
})
```

While `sd_mecha.convert()` is usually preferred for simplicity, calling the conversion method directly removes the need to use `model_configs.register()`.
It is still necessary to define the config to be able to define the conversion method, however.
Without registering the config, since the model config registry will not be aware of the custom config,
it will not be possible to infer the conversion path automatically using `sd_mecha.convert()`.
As a result, we would then need to explicitly specify the unregistered config to `sd_mecha.literal()` when we want to use it:

```python
# ...
blocks = convert_my_blocks_to_sdxl(sd_mecha.literal({
    "the_key_we_want": 1.0,
    "rest": 0.0,
}, my_blocks_config))


# example usage
recipe = sd_mecha.weighted_sum(a, b, alpha=blocks)
```
