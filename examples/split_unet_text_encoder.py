import sd_mecha
sd_mecha.set_log_level()


text_encoder_recipe = sd_mecha.weighted_sum(
    "ghostmix_v20Bakedvae",
    "dreamshaper_332BakedVaeClipFix",
)

unet_recipe = sd_mecha.add_perpendicular(
    "ghostmix_v20Bakedvae",
    "dreamshaper_332BakedVaeClipFix",
    "pure/v1-5-pruned"
)

recipe = sd_mecha.weighted_sum(
    text_encoder_recipe,
    unet_recipe,
    alpha=(
        sd_mecha.txt15_blocks(default=0) |
        sd_mecha.unet15_blocks(default=1)
    ),
)

scheduler = sd_mecha.MergeScheduler(
    base_dir=r"E:\sd\models\Stable-diffusion",
)

scheduler.merge_and_save(recipe, output_path="basic_merge")
