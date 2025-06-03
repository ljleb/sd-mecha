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

recipe = sd_mecha.pick_component(unet_recipe, "diffuser") | text_encoder_recipe
sd_mecha.serialize(recipe, output="recipes/test_split_unet_text_encoder.mecha")
