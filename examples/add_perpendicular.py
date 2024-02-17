import sd_mecha
sd_mecha.set_log_level()


recipe = sd_mecha.add_perpendicular(
    "ghostmix_v20Bakedvae",
    "dreamshaper_332BakedVaeClipFix",
    "pure/v1-5-pruned",
)

merger = sd_mecha.RecipeMerger(
    base_dir=r"E:\sd\models\Stable-diffusion",
)

merger.merge_and_save(recipe, output_path="basic_merge")
