import sd_mecha
sd_mecha.set_log_level()

recipe = sd_mecha.dropout(
    sd_mecha.model("pure/v1-5-pruned.safetensors"),
    sd_mecha.model("ghostmix_v20Bakedvae.safetensors"),
    sd_mecha.model("dreamshaper_332BakedVaeClipFix.safetensors"),
    probability=0.9, alpha=0.5, seed=0,
)
sd = sd_mecha.merge(recipe)
