import sd_mecha
sd_mecha.set_log_level()


recipe = sd_mecha.add_perpendicular(
    sd_mecha.model("ghostmix_v20Bakedvae.safetensors"),
    sd_mecha.model("dreamshaper_332BakedVaeClipFix.safetensors"),
    sd_mecha.model("pure/v1-5-pruned.safetensors"),
)

sd_mecha.merge(recipe, output="result.safetensors")
