import sd_mecha
sd_mecha.set_log_level()


recipe = sd_mecha.rotate(
    "ghostmix_v20Bakedvae",
    "dreamshaper_332BakedVaeClipFix",
)

scheduler = sd_mecha.RecipeMerger(
    base_dir=r"E:\sd\models\Stable-diffusion",
    default_device="cuda",
)

scheduler.merge_and_save(recipe, output_path="basic_merge")
