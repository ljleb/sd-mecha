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

# scheduler contains default parameters
scheduler = sd_mecha.MergeScheduler(
    base_dir=r"E:\sd\models\Stable-diffusion",
)

# perform the entire merge plan and save to output path
scheduler.merge_and_save(recipe, output_path="basic_merge")
```

See the [examples](/examples) directory for other examples.

## Features

- Memory efficient model merging -- merge a very large number of models at the same time
- Mecha recipes as a storable and executable format
- Custom merge method programming interface for experiments

Coming soon:

- Recipe variables for general recipe templates
- Compose recipe templates to create mega recipes

## Install

```commandline
pip install sd-mecha torch
```

sd-mecha depends additionally on:

- `torch>=2.0.1`

The pypi package does not ship with `torch` so that you can install the appropriate version for your system.

## Usage

### Merge recipes with the CLI

```shell
pip -m sd_mecha merge <recipe.mecha> -o path/to/output.safetensors [options]
```

For more information:

```shell
pip -m sd_mecha merge --help
```

It is also possible to merge recipes from python code using the library. See also the [examples](/examples).

## Motivation

Keeping track of full merge recipes has always been annoying.
I needed something that allows to store merge recipes in a readable format while also being executable.
I also needed something that allows to fully merge an entire tree of models without having to save intermediate models to disk.

Typically, mergers load all models in memory before initiating the merge process.
This can be very inefficient when the merge focuses on each key individually:

![image of typical merge graph](/media/memory-gone.PNG)

sd-mecha doesn't have this problem as it saves keys as soon as it can:

![image of sd-mecha merge graph](/media/did-you-see-something.PNG)

This allows to merge a very large number of models simultaneously on low-end hardware. (i.e. 8+)
