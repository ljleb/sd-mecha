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

recipe = sd_mecha.n_average(*models)
merger = sd_mecha.Defaults(models_dir=r"E:\sd\models\Stable-diffusion")
merger.merge_and_save(recipe)
