import sd_mecha
sd_mecha.set_log_level()


text_encoder_recipe = sd_mecha.add_perpendicular(
    "js2prony_v10",
    "juggernautXL_v9Rundiffusionphoto2",
    "pure/sdxl_base",
    alpha=-2.0,
)

unet_recipe = sd_mecha.weighted_sum(
    "js2prony_v10",
    "juggernautXL_v9Rundiffusionphoto2",
    alpha=-2.0,
)

recipe = sd_mecha.weighted_sum(
    text_encoder_recipe,
    unet_recipe,
    alpha=(
        sd_mecha.sdxl_unet_blocks(in00=-1, in01=-1, in02=-1, in03=-1, default=0) |
        sd_mecha.sdxl_txt_blocks(0) |
        sd_mecha.sdxl_txt_g14_blocks(0)
    ),
)

merger = sd_mecha.RecipeMerger(
    models_dir=r"E:\sd\models\Stable-diffusion",
)

merger.merge_and_save(recipe, output="basic_merge")
