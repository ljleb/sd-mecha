import sd_mecha
sd_mecha.set_log_level()


recipe = sd_mecha.weighted_sum(
    sd_mecha.model("ghostmix_v20Bakedvae.safetensors"),
    sd_mecha.model("dreamshaper_332BakedVaeClipFix.safetensors"),
)

sd = sd_mecha.merge(recipe)
