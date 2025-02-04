import sd_mecha
sd_mecha.set_log_level()

recipe = sd_mecha.dropout(
    sd_mecha.model("pure/v1-5-pruned.safetensors"),
    sd_mecha.model("ghostmix_v20Bakedvae.safetensors"),
    sd_mecha.model("dreamshaper_332BakedVaeClipFix.safetensors"),
    probability=0.9, alpha=0.5, seed=0,
)
merger = sd_mecha.RecipeMerger(models_dir=r"E:\sd\models\Stable-diffusion")
merger.merge_and_save(recipe)
