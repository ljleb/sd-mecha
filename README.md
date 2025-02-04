# sd-mecha

[![PyPI version](https://badge.fury.io/py/sd-mecha.svg)](https://badge.fury.io/py/sd-mecha)
[![Discord Server](https://dcbadge.vercel.app/api/server/2EPaw6fxxm?style=flat)](https://discord.gg/invite/2EPaw6fxxm)

```python
import sd_mecha

# create the merge plan
model_a = sd_mecha.model("model_a.safetensors")
model_b = sd_mecha.model("model_b.safetensors")
recipe = sd_mecha.weighted_sum(model_a, model_b, alpha=0.5)

# initialize merger
merger = sd_mecha.RecipeMerger(models_dir="/path/to/models")

# merge!
merger.merge_and_save(recipe)
```

sd-mecha is a general memory-efficient model merging library. It can merge *any* model:
- Diffusion models
- LLMs
- VLMs
- Aesthetic scorers
- etc.

## Features

- Memory efficient model merging: merge a very large number of models in a single execution
- Textual and interpretable format for storage and execution (.mecha)
- Extensible library interface:
  - add custom models
  - add custom merge methods
- Builtin support for popular diffusion models:
  - Stable Diffusion 1.5
  - Stable Diffusion XL
  - Stable Diffusion 3
  - FLUX Schnell/Dev
- Merge LyCORIS networks together and to checkpoints
- Block-wise hyperparameters for precise control of blocks (aka MBW)

## Install

```commandline
pip install sd-mecha
```

Make sure to install the appropriate release of [`torch`](https://pytorch.org/get-started/locally/) to get the best performance.

## Usage

### Merge models

To merge models, mecha uses recipes.
A recipe is a list of instructions that describes the exact steps needed to obtain the final merged model.

#### Using python

Here's an example script that merges three Stable Diffusion 1.5 models:

```python
from sd_mecha import model, clamp, RecipeMerger

# create a simple add difference recipe
# all builtin merge methods are direct properties of the `sd_mecha` package for convenience
a = model("ghostmix_v20Bakedvae.safetensors")
b = model("deliberate_v2.safetensors")
c = model("dreamshaper_332BakedVaeClipFix.safetensors")
recipe = clamp(a + (b - c), a, b)

# merger contains default parameters
merger = RecipeMerger(
    models_dir="/path/to/models",
)

# perform the entire merge plan and save to output path
merger.merge_and_save(recipe, output="basic_merge.safetensors")
```

See the [examples](/examples) directory for more examples.

### Get Model-Specific Information

To specify block weights, we need to know the name of the blocks.

This information can be discovered using the `extensions.model_configs` submodule.

Mecha has builtin support for Stable Diffusion 1.X, Stable Diffusion XL, Stable Diffusion 3 and FLUX Schnell/Dev:

```python
from sd_mecha.extensions import model_configs

all_configs = model_configs.get_all()

print([config.identifier for config in all_configs])
# ["sd1-ldm-base", "sdxl-sgm-base", "sd3-sgm-base", ...]
```

To view the available components of a model:

```python
from sd_mecha.extensions import model_configs

config = model_configs.resolve("sd1-ldm")
for component_id, component in config.components.items():
      # block_keys contains the state dict keys that the block controls
      print(f"{component_id}")

# this prints:
#   clip_l
#   vae
#   diffusers
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
