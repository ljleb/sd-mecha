import sd_mecha
import torch
sd_mecha.set_log_level()


# 5 example models to average together
models = [
    "ghostmix_v20Bakedvae",
    "dreamshaper_332BakedVaeClipFix",
    "deliberate_v2",
    "darkSushi25D25D_v20",
    "CounterfeitV30_v30",
]

recipe = models[0]
for i, model in enumerate(models[1:], start=2):
    # dtype to accommodate precision loss by alpha near 1
    dtype = torch.float16 if i - 2 < 4 else torch.float32 if i - 2 < 16 else torch.float64
    recipe = sd_mecha.weighted_sum(model, recipe, alpha=(i-1)/i, dtype=dtype)

scheduler = sd_mecha.MergeScheduler(base_dir=r"E:\sd\models\Stable-diffusion")
scheduler.merge_and_save(recipe, output_path="n_average")
