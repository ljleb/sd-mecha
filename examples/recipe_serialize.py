import sd_mecha
sd_mecha.set_log_level()


text_encoder_recipe = sd_mecha.add_perpendicular(
    "ghostmix_v20Bakedvae",
    "dreamshaper_332BakedVaeClipFix",
    "pure/v1-5-pruned"
)

unet_recipe = sd_mecha.weighted_sum(
    "ghostmix_v20Bakedvae",
    "dreamshaper_332BakedVaeClipFix",
)

recipe = sd_mecha.weighted_sum(
    text_encoder_recipe,
    unet_recipe,
    alpha=(
        sd_mecha.default("sd1", "txt", 0) |
        sd_mecha.default("sd1", "unet", 1)
    ),
)


sd_mecha.serialize_and_save(recipe, "recipes/test_split_unet_text_encoder.mecha")
