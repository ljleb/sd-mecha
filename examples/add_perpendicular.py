import sd_mecha
sd_mecha.set_log_level()


recipe = sd_mecha.add_perpendicular(
    sd_mecha.model("ghostmix_v20Bakedvae.safetensors"),
    sd_mecha.model("dreamshaper_332BakedVaeClipFix.safetensors"),
    sd_mecha.model("pure/v1-5-pruned.safetensors"),
)

merger = sd_mecha.RecipeMerger(
    models_dir=r"E:\sd\models\Stable-diffusion",
)

merger.merge_and_save(recipe)
