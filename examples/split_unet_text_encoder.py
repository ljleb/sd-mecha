import sd_mecha
sd_mecha.set_log_level()


text_encoder_recipe = sd_mecha.add_perpendicular(
    "ghostmix_v20Bakedvae",
    "dreamshaper_332BakedVaeClipFix",
    "pure/v1-5-pruned"
)

unet_recipe = sd_mecha.weighted_sum(
    "ghostmix_v20Bakedvae",
    "dreamshaper_332BakedVaeClipFix",
)

recipe = sd_mecha.weighted_sum(
    text_encoder_recipe,
    unet_recipe,
    alpha=(
        sd_mecha.sd15_txt_classes(0) |
        sd_mecha.sd15_unet_classes(1)
    ),
)

merger = sd_mecha.RecipeMerger(
    base_dir=r"E:\sd\models\Stable-diffusion",
)

merger.merge_and_save(recipe, output_path="basic_merge")
