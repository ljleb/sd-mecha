import sd_mecha
sd_mecha.set_log_level()


recipe = sd_mecha.weighted_sum(
    "ghostmix_v20Bakedvae",
    "dreamshaper_332BakedVaeClipFix",
)


sd_mecha.serialize_and_save(recipe, "recipes/test_weighted_sum.mecha")
