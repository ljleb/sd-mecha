# sd-mecha

sd-mecha is a stable diffusion recipe merger:

```python
import sd_mecha

# create a simple weighted sum recipe
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
merger = sd_mecha.MergeScheduler(
    base_dir=r"E:\sd\models\Stable-diffusion",
)

# perform the entire merge plan and save to output path
merger.merge_and_save(recipe, output_path="basic_merge")
```

See the [examples](/examples) directory for other examples.

## Features

- Memory efficient model merging -- merge a very large number of models at the same time
- Mecha recipes as a textual and interpretable format (.mecha)
- Custom merge method programming interface for experiments
- Recipe variables for general recipe templates
- Compose recipe templates to create mega recipes
- SD1.5 and SDXL supported
- Merge loras (SD1.5)

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

### Merge models

You can merge models following a recipe. Make sure the recipe does not contain any parameters:

```shell
python -m sd_mecha merge <path/to/recipe.mecha> [options]
```

For more information:

```shell
python -m sd_mecha merge --help
```

### Compose recipes

You can compose recipes together to create more complex recipes.
For this to work, the base recipe must contain free parameters:

```shell
python -m sd_mecha compose <path/to/recipe.mecha> [options]
```

For exampl, here we compose the recipe in `examples/recipes/incompatible_fusion.mecha`
with another recipe for parameter "a" and
the sd1.5 base model for parameter "c":

```shell
python -m sd_mecha compose examples/recipes/incompatible_fusion.mecha \
  -p a examples/recipes/weighted_sum.mecha \
  -p c pure/v1-5-pruned
```

For more information:

```shell
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
