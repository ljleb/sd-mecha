import sd_mecha
sd_mecha.set_log_level()


text_encoder_recipe = sd_mecha.add_perpendicular(
    sd_mecha.model("ghostmix_v20Bakedvae.safetensors"),
    sd_mecha.model("dreamshaper_332BakedVaeClipFix.safetensors"),
    sd_mecha.model("pure/v1-5-pruned.safetensors")
)

unet_recipe = sd_mecha.weighted_sum(
    sd_mecha.model("ghostmix_v20Bakedvae.safetensors"),
    sd_mecha.model("dreamshaper_332BakedVaeClipFix.safetensors"),
)

config = unet_recipe.model_config
recipe = sd_mecha.weighted_sum(
    text_encoder_recipe,
    unet_recipe,
    alpha=sd_mecha.convert({"BASE": 0}, "sd1-ldm") | 1
)
sd_mecha.serialize_and_save(recipe, "recipes/test_split_unet_text_encoder.mecha")
