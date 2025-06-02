import sd_mecha
sd_mecha.set_log_level()


models = [
    sd_mecha.model("ghostmix_v20Bakedvae.safetensors"),
    sd_mecha.model("dreamshaper_332BakedVaeClipFix.safetensors"),
    sd_mecha.model("realisticVisionV20_v20.safetensors"),
    sd_mecha.model("illustrationArtstyleMM_27.safetensors"),
    sd_mecha.model("lyriel_v16.safetensors"),
    sd_mecha.model("Midnight Maple.safetensors"),
    sd_mecha.model("mixproyuki77mi_v10.safetensors"),
]

recipe = sd_mecha.add_difference_ties(sd_mecha.model("pure/v1-5-pruned.safetensors"), *models, alpha=0.5)
sd = sd_mecha.merge(recipe, output_device="cpu")
