# User-Defined Merge Methods

While there are multiple merge methods and conversion functions built into sd-mecha, sometimes there is a need for something other than what the builtins can offer.
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
    gen = None
    if seed is not None:
        gen = Generator()
        gen.manual_seed(seed)
    noise = randn(a.shape, generator=gen, dtype=a.dtype).to(a.device)

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
`merge()` calls the original function for the planned keys.

In general, certain rules must be followed to define a custom merge method:

- The type of each function parameter needs to be `sd_mecha.Parameter(interface, merge_space=..., model_config=...)`. This type is used to specify additional metadata for a merge method parameter.
    In particular, additional keyword arguments can be passed to `Parameter(...)` if a parameter is to be required to:

    - receive a specific model config (`model_config=...`)
    - constrain its merge space to a set of merge spaces or an instance of `sd_mecha.extensions.merge_spaces.MergeSpaceSymbol` (`merge_space=...`)

    This can be useful for different reasons, all covered below.

- The return type needs to be `sd_mecha.Return(interface, ...)`. This type is used similarly to `Parameter(...)`.
    However, it differs slightly in its arguments:

    - Use `sd_mecha.StateDict[...]` to declare a method that returns a mapping for a group of output keys. Declare the group in `map_keys()`.
    - the merge space has to be either a single merge space or an instance of `sd_mecha.extensions.merge_spaces.MergeSpaceSymbol`
    - `model_config=` accepts a model config object or a registered ID string. For example, use `Return(Tensor, model_config="sdxl-sgm")`. The second positional argument selects the merge space.

- Optionally, `**kwargs` can be added to the arguments list. If present, the merge method will receive these additional parameters when called:

    - `key`: the name of the target key currently being processed.
    This is the name of the key from the target model config for which the merge method is expected to return a value.
    - `key_relation`: the selected input keys, output keys, and key map metadata.
    - `cache`: the cache object passed to `merge()` for this node, or `None`.
    - `output_reused`: whether sd-mecha plans to reuse this output.

The types that can be used as input to a merge method are:

- `torch.Tensor`
- `float`
- `int`
- `bool`
- `typing.Literal` for a fixed set of strings, such as `Literal["fast", "accurate"]`
- an instance of `typing.TypeVar` optionally bound by one or more of the above types
- `sd_mecha.StateDict[T]` where `T` is any of the above types

When a parameter is of type `sd_mecha.StateDict[T]`, instead of receiving a tensor from an input state dict, it will be an instance of a subclass of `sd_mecha.StateDict`.
`StateDict` is a dict-like interface that will load tensors from disk or other sources on demand.
When the config of a state dict parameter is the same as the target config, we can simply use `state_dict[kwargs["key"]]` to materialize the target key from the state dict.
Use `state_dict.keys()` to see the input keys selected by the key map for this call.

Use `Return(Literal["fast", "accurate"])` to declare the strings a method can produce. Wrap the literal type in `Optional` to include `None`, or in `StateDict` to return values for several keys.

Literal inputs accept a constant, per-key values, or another recipe. The recipe’s declared output choices must be a subset of the choices allowed by the input. sd-mecha checks compatibility when it builds or finalizes a recipe and checks returned values during the merge. Plain `str` parameters are deprecated.

## Relate one parameter to another

Use `param_ref("a").model_config` when an input or output must use the same key format as `a`. A method can set `allow_conversion=True` on one input. sd-mecha checks for a direct match first, then looks for a unique shortest conversion path to the required config and merge space.

```python
import torch
from sd_mecha import Parameter, Return, merge_method, param_ref

@merge_method
def apply_delta(
    a: Parameter(torch.Tensor, "weight"),
    delta: Parameter(
        torch.Tensor,
        "delta",
        model_config=param_ref("a").model_config,
        allow_conversion=True,
    ),
) -> Return(torch.Tensor, "weight", model_config=param_ref("a").model_config):
    return a + delta
```

Use `param_ref("a").map(callback)` if you need to derive the allowed configs. The callback receives a `ParameterInfo` with `model_config` and `merge_spaces`. In a `model_config=` argument, the callback returns configs or config IDs. When used as a return config, it must resolve to one config.

A standalone method can use a mapped reference as a parameter type. Here the choices come from the components of input `a`.

```python
from typing import Literal
from sd_mecha import Parameter, param_ref

def component_type(info):
    return Literal[tuple(info.model_config.components())]

Component = Parameter(param_ref("a").map(component_type), "param")
```

Use `Component` as the annotation of a parameter in a method with an input named `a`. sd-mecha resolves the choices during recipe construction or finalization, once the source metadata is known.

## Register implementations of a shared signature

Use `@merge_method(is_interface=True)` on a function with an empty body to declare a signature. Register each implementation with `@merge_method(implements=interface)`. sd-mecha uses the input configs and merge spaces to select an implementation.

An implementation must declare the same `Literal` input choices as its interface. Use those values inside the implementation to choose an algorithm. The implementation can declare a subset of the interface's output choices.

## Declare key dependencies

A function with tensor inputs normally reads the same key from each input. Use a class with a static or class method `map_keys(builder)` to read different keys. sd-mecha requires a key map for methods that declare different configs or return multiple keys.

Combine required inputs with `&` and alternatives with `|`. For example, `(builder.a.keys["old"] | builder.a.keys["new"]) & builder.b.keys["scale"]` requires a scale and either spelling of the first input. Use `builder.a.keys[()]` when an alternative can run without any keys from `a`.

Call `sd_mecha.skip_key(key)` when a particular output cannot be produced at run time. A fallback recipe can supply the result for that key.

## Custom Blocks Config

Block merging basics are covered in [Typical Use Cases > Blocks Merging (MBW)](1-typical-use-cases.md#blocks-merging-mbw).

To define a custom block config, two things need to be defined:

1. The name of each block, in a model config. This is required to enable conversion and serialization
2. A conversion method that maps each block to a key in the target model config

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
from typing import TypeVar
from sd_mecha import Parameter, Return, StateDict, merge_method

T = TypeVar("T")
sdxl = model_configs.resolve("sdxl-sgm")
target_key = "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.attn2.to_v.weight"

@merge_method(is_conversion=True, reuse_outputs=False)
class convert_my_blocks_to_sdxl:
    @staticmethod
    def map_keys(b):
        for key in sdxl.keys():
            block_key = "the_key_we_want" if key == target_key else "rest"
            b[key] = b.blocks_dict.keys[block_key]

    def __call__(
        self,
        blocks_dict: Parameter(StateDict[T], model_config="sdxl-my_blocks"),
        **kwargs,
    ) -> Return(T, model_config="sdxl-sgm"):
        block_key, = kwargs["key_relation"].inputs["blocks_dict"]
        return blocks_dict[block_key]


# 3. use the conversion method
from sd_mecha import model, convert, literal, weighted_sum

a = model("path/to/model_a.safetensors", config="sdxl-sgm")
b = model("path/to/model_b.safetensors", config="sdxl-sgm")

blocks = literal({
    "the_key_we_want": 1.0,
    "rest": 0.0,
}, config="sdxl-my_blocks", merge_space="param")
alpha = convert(blocks, "sdxl-sgm")

recipe = weighted_sum(a, b, alpha=alpha)
```

Passing `is_conversion=True` to `@merge_method` registers the method as a possible path for `sd_mecha.convert()`. Here it converts `sdxl-my_blocks` to `sdxl-sgm`.

As an alternative, call the conversion method directly:

```python
alpha = convert_my_blocks_to_sdxl(blocks)
recipe = weighted_sum(a, b, alpha=alpha)
```

Register a custom config before loading a saved recipe that refers to its ID.

## Cache work across merges

Set `cache_factory` on `@merge_method` to provide a constructor for the method's cache. `create_cache(*args, **kwargs)` forwards its arguments to that factory. Give factory parameters defaults so callers can create a cache with no arguments. Create a cache for each node you want to reuse, then pass the mapping to `merge()`. The cache belongs to the caller, so the method must decide when cached data is still valid.

```python
from sd_mecha import merge
from sd_mecha.recipe_nodes import MergeRecipeNode

cache = {
    node: node.merge_method.create_cache()
    for node in recipe
    if isinstance(node, MergeRecipeNode)
}
merge(recipe, cache=cache)
```

For rotation, rank or Kronecker truncation, and LoRA or LoKr extraction, pass `device=` or `dtype=` to `node.merge_method.create_cache()`. The factories store both settings in the cache dictionary and default to CPU and `torch.float16`. When cached factors have a different device or dtype, the method recomputes them and stores them with the requested settings.

Within one merge, sd-mecha can reuse outputs that have multiple consumers. Set `reuse_outputs=False` for cheap operations whose outputs sd-mecha can recompute.

Next: [Builtin Merge Methods](3-merge-methods-listing.md)
