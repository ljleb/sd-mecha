# sd-mecha

sd-mecha is a configuration-based stable diffusion checkpoint merger:

```python
import sd_mecha

# plan a simple weighted sum
merge = sd_mecha.weighted_sum(
    "ghostmix_v20Bakedvae",
    "dreamshaper_332BakedVaeClipFix",
    alpha=0.5,
)

# scheduler contains default parameters
scheduler = sd_mecha.MergeScheduler(
    base_dir=r"E:\sd\models\Stable-diffusion",
    device="cuda:0",
    prune=True,
)

# perform the entire merge plan and save to output path
scheduler.merge_and_save(merge, output_path="basic_merge")
```

See the [examples](/examples) directory for other examples.

## Motivation

Keeping track of full merge recipes has always been annoying.
I needed something that allows to store merge recipes in a readable format while also being executable.
I also needed something that allows to fully merge an entire tree of models without having to save intermediate models to disk.

## Acknowledgements

This code is heavily based on the [sd-meh](https://github.com/s1dlx/meh) library.
