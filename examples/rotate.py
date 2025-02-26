import sd_mecha
sd_mecha.set_log_level()


recipe = sd_mecha.rotate(
    sd_mecha.model("ghostmix_v20Bakedvae.safetensors"),
    sd_mecha.model("dreamshaper_332BakedVaeClipFix.safetensors"),
)

merger = sd_mecha.Defaults(
    models_dir=r"E:\sd\models\Stable-diffusion",
)

merger.merge_and_save(recipe)
