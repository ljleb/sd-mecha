# Typical Use Cases

Different tasks can be performed with sd-mecha. Here are typical use cases.

## Merge Models

A traditional merge method is `weighted_sum`. Let's use it as an illustrative example:

```python
import sd_mecha

a = sd_mecha.model("path/to/model_a.safetensors")  # 1.
b = sd_mecha.model("path/to/model_b.safetensors")  # 2.

recipe = sd_mecha.weighted_sum(a, b, alpha=0.5)  # 3.

sd_mecha.merge(recipe, output="path/to/model_out.safetensors")  # 4.
```

Here is what happens at each step:

1. `sd_mecha.model("path/to/model_a.safetensors")`: creates a handle to a safetensors model on disk. It will not load the model right away, only a reference to it
2. `sd_mecha.model("path/to/model_b.safetensors")`: does the same as 1. but with a different model
3. `sd_mecha.weighted_sum(a, b, alpha=0.5)`: creates a `weighted_sum` recipe node with `a` and `b` as children. It will not merge the models right away
4. `sd_mecha.merge(recipe, output="path/to/model_out.safetensors")`: merges the recipe graph by streaming the output tensors to disk

The `sd_mecha` module has multiple merge methods builtin (i.e. `sd_mecha.add_difference`, `sd_mecha.train_difference_mask`).
See the [builtin merge methods listing](3-merge-methods-listing.md) for a comprehensive list of builtin merge methods and their associated descriptions.

Omit `output` to get a state dict in memory. To fill missing output keys, use a fallback such as `recipe | a`.

## Merge individual keys differently

Each key names one tensor in a checkpoint. Pass a map of keys to blend weights as `alpha` to control how each tensor is merged. Use `|` to supply a default for the other keys.

```python
import sd_mecha

config = "sd1-kohya"
a = sd_mecha.model("model_a.safetensors", config=config)
b = sd_mecha.model("model_b.safetensors", config=config)
key = "vae.post_quant_conv.bias"

# Give b 80% of the weight for this key and 50% everywhere else.
alpha = sd_mecha.literal({key: 0.8}, config=config, merge_space="param") | 0.5
blend = sd_mecha.weighted_sum(a, b, alpha=alpha)

# Use a geometric blend for this key and keep the other keys from blend.
geometric = sd_mecha.geometric_sum(a, b, alpha=0.5)
override = sd_mecha.literal({key: geometric}, config=config, merge_space="weight")
recipe = override | blend
```

## Convert Models

Different key formats can name or store the same weights differently. `convert()` composes registered conversion methods between key formats. Pass a config identifier or a recipe whose model config defines the target key format.

```python
import sd_mecha

model = sd_mecha.model("moss_comfyui.safetensors", config="moss_image-comfyui")
recipe = sd_mecha.convert(model, "moss_image-openmoss")

sd_mecha.merge(recipe, output="moss_openmoss.safetensors")
```

See [Merge Models](#merge-models) for how the merge runs.

To compose a difference, subtract two checkpoints with `target - base`. Then add the difference to a checkpoint with `base + diff`. The latter calls the difference merge method with `alpha=1.0`.

## Apply a LyCORIS adapter

Use `apply_lycoris()` to apply an adapter to a checkpoint. sd-mecha converts the checkpoint to the adapter's base key format when needed and keeps base weights for keys the adapter does not target.

```python
import sd_mecha

base = sd_mecha.model("base.safetensors")
adapter = sd_mecha.model("adapter.safetensors")
recipe = sd_mecha.apply_lycoris(base, adapter)

sd_mecha.merge(recipe, output="merged.safetensors")
```

You can pass the adapter config explicitly if needed. For example, `kolors-diffusers_lycoris_lora` targets the standalone Kolors UNet. To scale an additive adapter, pass it to `add_difference(base, adapter, alpha=...)`. sd-mecha converts the adapter to a delta. Use `apply_lycoris()` for BOFT, which needs the base weights.

## Extract an adapter

Use `sd_mecha.lycoris.resolve()` to select an extractor. LoRA, LoKr, and Norm take a difference. BOFT takes both the original and target weights.

```python
import sd_mecha

base = sd_mecha.model("base.safetensors")
target = sd_mecha.model("target.safetensors")

extract = sd_mecha.lycoris.resolve("lora")
recipe = extract(target - base, rank=16)
sd_mecha.merge(recipe, output="difference_lora.safetensors")
```

For LoKr, pass `dims=` to choose the factor sizes. Create the per-key dimensions with `sd_mecha.lokr_dims()` or `sd_mecha.lokr_dims_lycoris()`.

```python
extract_boft = sd_mecha.lycoris.resolve("boft")
recipe = extract_boft(base, target, block_size=10, stages=3, solver="svd_sweeps")
sd_mecha.merge(recipe, output="difference_boft.safetensors")
```

The default BOFT solver uses four sweeps. Use `solver="lbfgs"` to refine that fit. `stages` limits how many butterfly stages the adapter uses per weight.

## Blocks Merging (MBW)

Another very common use case is to associate a weight to each "block" of a model, and then merge each "block" according to these weights (also known as [Merge Block Weighted (MBW)](https://note.com/kohya_ss/n/n9a485a066d5b)).
For this to be possible, we need to define what a "block" is for a given model.
This is because there is not always a true canonical way to neatly fit all keys of a model into fixed categories.
For example, sometimes a small group of keys could belong to more than one block.

The built-in block configs are:

- SDXL supermerger blocks (`sdxl-supermerger_blocks`)
- SD1 supermerger blocks (`sd1-supermerger_blocks`)
- Anima (`anima-blocks`), with keys such as `DIFFUSER00` and `LLM_ADAPTER00`.
- MOSS-Image (`moss_image-blocks`), with keys such as `DIFFUSER00` and `VAE`.

This is how block-weighted merging works in sd-mecha:

```python
import sd_mecha

a = sd_mecha.model("path/to/model_a.safetensors", config="sdxl-sgm")
b = sd_mecha.model("path/to/model_b.safetensors", config="sdxl-sgm")

blocks = sd_mecha.literal({
    "BASE": 0.0,
    "IN00": 0.5,
    "IN01": 0.25,
    "IN02": 0.75,
    # ...
}, config="sdxl-supermerger_blocks", merge_space="param")
blocks = sd_mecha.convert(blocks, a)
blocks = blocks | 1.0  # optional

recipe = sd_mecha.weighted_sum(a, b, alpha=blocks)

sd_mecha.merge(recipe)
```

Step by step:

1. `blocks` is a literal with one value per block in `param` space.
The keys of the blocks are specific to each architecture.
However, in the case of supermerger, each key name is one of `INxx`, `OUTxx`, (here lower case `x` stands for any digit between `0` and `9`) `M00` and `BASE`. (and `VAE` for SDXL)
2. `blocks = sd_mecha.convert(blocks, a)` returns a recipe that broadcasts the block weights to the keys of `a`'s model config. Each key gets one block weight, which it shares with the other keys in the block.
See [Convert Models](#convert-models) for an overview of `sd_mecha.convert`.
3. `blocks = blocks | 1.0`: specifies that for any key that is missing a corresponding block, the fallback value is `1.0`.
Note that this is optional if all block weights have been explicitly specified in the literal state dict.

Defining custom blocks is a tad more involved, see [User-Defined Merge Methods > Custom Blocks Config](2-user-defined-merge-methods.md#custom-blocks-config) for more info.

## Replace Model Components

The keys of model configs are always distributed into different *components*. For example, SDXL configs include `diffuser`, `clip_l`, `clip_g`, and `vae` components.
Sometimes, all we really want is to replace one self-contained component of a model with a different one. Here is how to achieve this:

```python
import sd_mecha

a = sd_mecha.model("path/to/model_a.safetensors")
b = sd_mecha.model("path/to/model_b.safetensors")

a_vae = sd_mecha.pick_component(a, "vae")  # 1.
recipe = a_vae | b  # 2.

sd_mecha.merge(recipe)
```

Explanation:

1. `sd_mecha.pick_component(a, "vae")`: picks the `vae` component of model `a`. It discards all keys from other components.
2. `a_vae | b`: replaces all missing keys (so keys that are not from the vae) with keys from `b`. The `|` operator is a shorthand for `sd_mecha.fallback`.

## Save and load a recipe

A `.mecha` file stores the operations and file references needed to run a recipe again. Keep the referenced checkpoint files available. To serialize a recipe that uses in-memory tensors, save those tensors to Safetensors and use file inputs.

```python
from pathlib import Path
import sd_mecha

path = Path("merge.mecha")
sd_mecha.serialize(recipe, output=path)
restored = sd_mecha.deserialize_path(path)
sd_mecha.merge(restored, output="merged.safetensors")
```

By default, `serialize()` finalizes the recipe when it needs to resolve file or literal nodes. Pass `finalize=False` to write the recipe as supplied.

Use `merge_device=` to set the device for input tensors and `output_device=` to set the output device. A device set by `cast(..., device=...)` or `recipe.to(device=...)` is part of the recipe. Saving an explicit device cast emits a warning.

```python
restored = sd_mecha.deserialize_path(path, remap_devices={"cuda:1": "cuda:0"})
restored = sd_mecha.deserialize_path(path, remap_devices="cpu")
restored = sd_mecha.deserialize_path(path, remap_devices=None)
```

A dictionary remaps matching saved device names, including per-key literals. A string replaces all explicit device casts. `None` clears device placement and retains dtype casts. The loader applies the override before validating devices on the current machine.

Next: [User-Defined Merge Methods](2-user-defined-merge-methods.md)
