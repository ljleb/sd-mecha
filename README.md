# sd-mecha

sd-mecha is a memory-efficient general-purpose model merger. It can merge any model architecture given appropriate configuration. (i.e. diffusion models, LLMs, Depth models, etc.)

## Features

- Memory efficient model merging -- merge a very large number of models at the same time
- Mecha recipes as a textual and interpretable format (.mecha)
- Extension API through python:
    - add new architectures (i.e. Stable Cascade, Stable Diffusion 3, etc.)
    - add new model types (i.e. OFT networks, LoKr, etc.)
    - add new merge methods
- Recipe variables for general recipe templates
- Compose recipe templates to create mega recipes
- SD1.5 and SDXL supported
- Merge loras together and to checkpoints
- Block-wise hyperparameters for precise control of blocks
- Class-wise hyperparameters for precise control of layer types

Coming soon:

- Save lora from delta space model

## Install

```commandline
pip install sd-mecha torch
```

sd-mecha depends additionally on:

- `torch>=2.0.1`

The pypi package does not ship with `torch` so that you can install the appropriate version for your system.

## Usage

### Get Model-Specific Information

The interface for merge hyperparameters requires prior knowledge of the blocks and classes of the architecture being merged.
The command `info` was made to discover the names of the blocks and/or classes to use.

To show the registered model architectures:

```commandline
python -m sd_mecha info
```

Mecha has builtin support for the SD1.x and the SDXL architectures:

```
Available architectures:
- sd1
- sdxl
```

To view the available blocks and classes of an architecture, specify the architecture:

```commandline
python -m sd_mecha info sd1
```
```
Component "txt":
  Blocks:
  - in0
  - in1
  - in2
  ...
  Classes:
  - final_layer_norm
  - layer_norm1
  - layer_norm2
  - mlp_fc1
  ...
Component "unet":
  Blocks:
  ...
  Classes:
  ...
```

If run as verbose, it also shows the keys that are associated with each block/class:

```commandline
python -m sd_mecha info sd1 -v
```
```
Component "txt":
  Blocks:
    in0:
    - model.diffusion_model.input_blocks.0.0.bias
    - model.diffusion_model.input_blocks.0.0.weight
    in1:
    - model.diffusion_model.input_blocks.1.0.emb_layers.1.bias
    - model.diffusion_model.input_blocks.1.0.emb_layers.1.weight
    - model.diffusion_model.input_blocks.1.0.in_layers.0.bias
    - model.diffusion_model.input_blocks.1.0.in_layers.0.weight
    ...
  ...
...
```

### Merge models

To merge models, mecha needs a recipe as input. There are multiple ways to provide a recipe:
- using the python merging API
- using the CLI with .mecha recipes

#### Using the python merging API

Here's an example simple sum-twice merge setup:

```python
import sd_mecha

# create a simple weighted sum recipe
# all builtin merge methods are direct properties of the `sd_mecha` package for convenience
recipe = sd_mecha.weighted_sum(
    sd_mecha.weighted_sum(
        "ghostmix_v20Bakedvae",
        "deliberate_v2",
        alpha=0.5,
    ),
    "dreamshaper_332BakedVaeClipFix",
    alpha=0.33,
)

# merger contains default parameters
merger = sd_mecha.RecipeMerger(
    models_dir=r"E:\sd\models\Stable-diffusion",
)

# perform the entire merge plan and save to output path
merger.merge_and_save(recipe, output="basic_merge")
```

See the [examples](/examples) directory for more examples.

#### Using the CLI with .mecha recipes

It is alternatively possible to merge recipes previously serialized to `.mecha`.
This is only possible if the recipe is concrete. (i.e. all potential parameters have been replaced with actual models)

```commandline
python -m sd_mecha merge path/to/recipe.mecha
```

For more information:

```commandline
python -m sd_mecha merge --help
```

### Compose recipes

It is possible to compose recipes together to create more complex recipes.
For this to work, the base recipe must be general: (i.e. the parameters to replace must exist in the base recipe)

```commandline
python -m sd_mecha compose path/to/recipe.mecha [options]
```

For example, here we compose the recipe [incompatible_fusion.mecha](examples/recipes/incompatible_fusion.mecha)
with another recipe for parameter "a" and
SD1.5 base for parameter "c":

```commandline
python -m sd_mecha compose examples/recipes/incompatible_fusion.mecha \
  -p a examples/recipes/weighted_sum.mecha \
  -p c v1-5-pruned.safetensors
```

For more information:

```commandline
python -m sd_mecha merge --help
```

## Motivation

Keeping track of full merge recipes has always been annoying.
I needed something that allows to store merge recipes in a readable format while also being executable.
I also needed something that allows to fully merge an entire tree of models without having to save intermediate models to disk.

Typically, mergers load all models in memory before initiating the merge process.
This can be very inefficient when the merge focuses on each key individually:

![image of typical merge graph](/media/memory-gone.PNG)

sd-mecha doesn't have this problem as it saves keys as soon as it can:

![image of sd-mecha merge graph](/media/did-you-see-something.PNG)

This allows to merge a very large number of models simultaneously on low-end hardware.
