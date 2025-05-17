# sd-mecha

[![PyPI version](https://badge.fury.io/py/sd-mecha.svg)](https://badge.fury.io/py/sd-mecha)
[![Discord Server](https://dcbadge.vercel.app/api/server/2EPaw6fxxm?style=flat)](https://discord.gg/invite/2EPaw6fxxm)

```python
import sd_mecha

# create the merge plan
a = sd_mecha.model("path/to/model_a.safetensors")
b = sd_mecha.model("path/to/model_b.safetensors")
recipe = sd_mecha.weighted_sum(a, b, alpha=0.5)

# merge!
sd_mecha.merge(recipe, output="path/to/model_out.safetensors")
```

sd-mecha is a general memory-efficient model merging library. It can merge any model:
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
  - FLUX Schnell/Dev
- Merge LyCORIS networks together and to checkpoints
- Block-wise hyperparameters for precise control of blocks (aka MBW)

## Install

```commandline
pip install sd-mecha
```

Make sure to install the appropriate release of [`torch`](https://pytorch.org/get-started/locally/) to get the best performance.

## Usage

See the [mecha guide](docs/guide) for an in-depth exploration of how to use the library, and to decide whether it is appropriate for your purposes.

### Merge models

To merge models, mecha uses recipes.
A recipe is a list of instructions that describes the exact steps needed to derive a state dict from inputs.

Here's an example script that merges three models:

```python
import sd_mecha

# create the merge plan
model_a = sd_mecha.model("path/to/model_a.safetensors")
model_b = sd_mecha.model("path/to/model_b.safetensors")
recipe = sd_mecha.weighted_sum(model_a, model_b, alpha=0.5)

# merge!
sd_mecha.merge(recipe, output="path/to/model_out.safetensors")
```

See the [examples](/examples) directory for more examples.

### Get Model-Specific Information

The library uses a "model config" to designate any specific set of keys of a certain shape.

It is possible to list all available model configs through the `sd_mecha.extensions.model_configs` module:

```python
from sd_mecha.extensions import model_configs

all_configs = model_configs.get_all()

print([config.identifier for config in all_configs])
# ["sd1-ldm-base", "sdxl-sgm-base", "sd3-sgm-base", ...]
```

A *component* of a model config is a subset of keys of the config that belong to the same logical group.
For example, all keys starting with "first_stage_model." in Stable Diffusion models belong to the component "vae".

It is possible to query the different components of a model config:

```python
from sd_mecha.extensions import model_configs

config = model_configs.resolve("sd1-ldm")
for component_id, component in config.components().items():
      # component.keys contains the state dict keys that the component owns
      print(f"{component_id}")

# this prints:
#   clip_l
#   vae
#   diffuser
```

## Motivation

Keeping track of full merge recipes has always been a problem for me.
I needed something that allows to store merge recipes in a readable format while also being executable.
I also needed something that allows to fully merge an entire tree of models without having to save intermediate models to disk.

Typically, mergers load all models in memory before initiating the merge process.
This can be very inefficient when the merge focuses on each key individually:

![image of typical merge graph](/media/memory-gone.PNG)

sd-mecha doesn't have this problem as it saves keys as soon as it can:

![image of sd-mecha merge graph](/media/did-you-see-something.PNG)

This allows to merge a very large number of models simultaneously on low-end hardware.
