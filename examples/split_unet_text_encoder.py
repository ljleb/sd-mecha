import sd_mecha
sd_mecha.set_log_level()


text_encoder_recipe = sd_mecha.add_perpendicular(
    sd_mecha.model("js2prony_v10", "sdxl"),
    sd_mecha.model("juggernautXL_v9Rundiffusionphoto2", "sdxl"),
    sd_mecha.model("pure/sdxl_base", "sdxl"),
)

unet_recipe = sd_mecha.weighted_sum(
    sd_mecha.model("js2prony_v10", "sdxl"),
    sd_mecha.model("juggernautXL_v9Rundiffusionphoto2", "sdxl"),
)

recipe = sd_mecha.weighted_sum(
    text_encoder_recipe,
    unet_recipe,
    alpha=(
        sd_mecha.blocks("sdxl", "txt") |
        sd_mecha.blocks("sdxl", "txt2") |
        sd_mecha.default("sdxl", "unet", 1)
    ),
)

merger = sd_mecha.RecipeMerger(
    models_dir=r"E:\sd\models\Stable-diffusion",
)

merger.merge_and_save(recipe, output="basic_merge")
