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
sd = sd_mecha.merge(recipe)
