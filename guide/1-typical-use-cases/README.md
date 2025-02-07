# Typical Use Cases

Different tasks can be performed with sd-mecha. Here are typical use cases.

## Merge Models

A traditional merge method is `weighted_sum`. Let's use it as an illustrative example:

```python
import sd_mecha

a = sd_mecha.model("path/to/model_a.safetensors")
b = sd_mecha.model("path/to/model_b.safetensors")

recipe = sd_mecha.weighted_sum(a, b, alpha=0.5)

sd_mecha.merge_and_save(recipe, "path/to/model_out.safetensors")
```

Here is what happens at each step:

1. `a = sd_mecha.model("path/to/model_a.safetensors")`: creates a handle to a safetensors model on disk. It will not load the model right away, only a reference to it
2. `b = sd_mecha.model("path/to/model_b.safetensors")`: does the same as 1. but with a different model.
3. `recipe = sd_mecha.weighted_sum(a, b, alpha=0.5)`: creates a `weighted_sum` recipe node with `a` and `b` as children. It will not merge the models right away
4. `sd_mecha.merge_and_save(recipe, "path/to/model_out.safetensors")`: merges the recipe graph by streaming one key at a time to disk

The `sd_mecha` module has multiple merge methods builtin (i.e. `sd_mecha.add_difference`, `sd_mecha.train_difference_mask`).
See the [builtin merge methods reference](todo) for a comprehensive list and companion description of the builtin merge methods.

## Convert Models

Some models have multiple different formats in which they can be represented.
For example, SDXL has the SGM format, the diffusers format, and a bunch of others used to represent SDXL PEFT adapters.
This is known to be a very unwieldy thing about dealing with models in the merging community.
It prevents otherwise compatible models from easily being used together in state dict recipes.
There is code to convert between different formats for the same model,
however, all conversion functions are scattered across different repositories.
Furthermore, none of these conversion utilities are remotely efficient when it comes to system resources utilization.

To simplify this process and to make it much less eager to deplete system resources, `sd_mecha` defines a simple conversion mechanism: `sd_mecha.convert()`.
Here's an example of how you can use it. A typical use case is to merge a LoRA to a base model:

```python
import sd_mecha

base = sd_mecha.model("path/to/base.safetensors")
lora = sd_mecha.model("path/to/lora.safetensors")
diff = sd_mecha.convert(lora, base)
recipe = base + diff

sd_mecha.merge_and_save(recipe, "path/to/model_out.safetensors")
```

Here is what happens at each step (skipping `merge_and_save` which was covered in [Merge Models](#merge-models)):

- `base = sd_mecha.model("path/to/base.safetensors")`: creates a handle to a base model
- `lora = sd_mecha.model("path/to/lora.safetensors")`: creates a handle to a LoRA adapter (or any other model that is compatible with the base model up to conversion)
- `sd_mecha.convert(lora, base)`: finds the shortest path of pre-registered conversion functions that converts from the model config of `lora` to that of `base`, then sequentially composes a recipe node from these conversion functions
- `recipe = base + diff`: creates a recipe node that will add the decompressed lora to the base model. `+` is a shorthand for `sd_mecha.add_difference` with `alpha=1.0`

### Blocks Merging (MBW)

Another very common use case is to associate a weight to each "block" of a model, and then merge each block according to these (also known as [Merge Block Weighted (MBW)](https://note.com/kohya_ss/n/n9a485a066d5b)).
For this to be possible, we need to define what a "block" is for a given model.
This is because there is not always a true canonical way to neatly fit all keys of a model into categories.
For example, sometimes a small group of keys could belong to more than one block.

For convenience, sd-mecha predefines block configs for models that have some block conventions that the community uses often.
Some of these include:

- SDXL supermerger blocks (`sdxl-supermerger_blocks`)
- SD1 supermerger blocks (`sd1-supermerger_blocks`)

This is how block-weighted merging works in sd-mecha:

```python
import sd_mecha

a = sd_mecha.model("path/to/model_a.safetensors")
b = sd_mecha.model("path/to/model_b.safetensors")

blocks = {
    "BASE": 0.0,
    "IN00": 0.5,
    "IN01": 0.25,
    "IN02": 0.75,
    # ...
}
blocks = sd_mecha.convert(blocks, a)
blocks = blocks | 1.0  # optional

recipe = sd_mecha.weighted_sum(a, b, alpha=blocks)

sd_mecha.merge_and_save(recipe, "path/to/model_out.safetensors")
```

Step by step:

1. `blocks = {`: The blocks are created as a literal state dictionary of block weights.
The keys of the blocks are specific to each architecture.
However, in the case of supermerger, they are one of `INxx`, `OUTxx`, (here lower case `x` stands for any digit between `0` and `9`) `M00` and `BASE`. (and `VAE` for SDXL)
2. `blocks = sd_mecha.convert(blocks, a)`: this converts the blocks to a full state dict of weights compatible with the input models.
See [Convert Models](#convert-models) for an explanation about `sd_mecha.convert`.
3. `blocks = blocks | 1.0`: specifies that for any key that is missing a corresponding block, the fallback value is `1.0`.
Note that this is optional if all block weights have been explicitly specified in the literal state dict.

Defining custom blocks is a tad more involved, see [User-Defined Merge Methods > Custom Blocks Config](../2-user-defined-merge-methods#custom-blocks-config) for more info.
