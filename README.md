# sd-mecha

sd-mecha is a stable diffusion recipe merger:

```python
import sd_mecha

# plan a simple weighted sum
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
    device="cuda:0",
    prune=True,
)

# perform the entire merge plan and save to output path
scheduler.merge_and_save(recipe, output_path="basic_merge")
```

See the [examples](/examples) directory for other examples.

## Motivation

Keeping track of full merge recipes has always been annoying.
I needed something that allows to store merge recipes in a readable format while also being executable.
I also needed something that allows to fully merge an entire tree of models without having to save intermediate models to disk.

## Install

```commandline
pip install sd-mecha torch tensordict
```

sd-mecha depends additionally on:

- `torch>=2.0.1`
- `tensordict`

The pypi package does not ship with `torch` nor `tensordict` so that you can install the appropriate version for your system.

## Acknowledgements

This code is heavily based on the [sd-meh](https://github.com/s1dlx/meh) library.
