# sd-mecha

[![PyPI version](https://badge.fury.io/py/sd-mecha.svg)](https://badge.fury.io/py/sd-mecha)
[![Discord Server](https://dcbadge.vercel.app/api/server/2EPaw6fxxm?style=flat)](https://discord.gg/invite/2EPaw6fxxm)

sd-mecha is a memory-efficient general-purpose model merger. It can merge any model architecture given appropriate configuration:
- diffusion models
- LLMs
- Depth models
- Scorers
- ...

## Features

- Memory efficient model merging -- merge a very large number of models at the same time
- Mecha recipes as a textual and interpretable format (.mecha)
- Extension API through python:
  - add new architectures (i.e. Stable Cascade, Stable Diffusion 3, etc.)
  - add new model types (i.e. OFT networks, LoKr, etc.)
  - add new merge methods
- Recipe variables for general recipe templates
- Compose recipe templates to create mega recipes
- Builtin support for popular model architectures:
  - SD1.5
  - SDXL
  - SD3
- Merge LoRAs together and to checkpoints
- Block-wise hyperparameters for precise control of blocks
- Class-wise hyperparameters for precise control of layer types
- Support arbitrary model architectures and types using the `sd_mecha.extensions` module
- Merge SDXL LoRAs to models and with other LoRAs

## Install

```commandline
pip install sd-mecha torch
```

sd-mecha depends additionally on:

- `torch>=2.0.1`

The pypi package does not ship with `torch` so that you can install the appropriate version for your system.

## Usage

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

### Get Model-Specific Information

The interface for block/class hyperparameters requires prior knowledge of the blocks and classes of the architecture being merged.
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

Given this information, it is possible to set i.e. the value of block `in2` in the `txt` component specifically:

```python
import sd_mecha
recipe = sd_mecha.weighted_sum(
    "ghostmix_v20Bakedvae",
    "dreamshaper_332BakedVaeClipFix",
    alpha=(
      sd_mecha.default("sd1", "txt", 0.33) |
      sd_mecha.blocks("sd1", "txt", in2=0.75)
    ),
)
```

See the [merging API section](#using-the-python-merging-api) above for more info.

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

```shell
python -m sd_mecha compose --help
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
