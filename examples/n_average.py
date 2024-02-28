import sd_mecha
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
    recipe = sd_mecha.weighted_sum(model, recipe, alpha=(i-1)/i)

merger = sd_mecha.RecipeMerger(models_dir=r"E:\sd\models\Stable-diffusion")
merger.merge_and_save(recipe, output="basic_merge")
