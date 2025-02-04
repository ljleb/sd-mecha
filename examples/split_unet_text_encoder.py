import sd_mecha
sd_mecha.set_log_level()


text_encoder_recipe = sd_mecha.add_perpendicular(
    sd_mecha.model("js2prony_v10.safetensors"),
    sd_mecha.model("juggernautXL_v9Rundiffusionphoto2.safetensors"),
    sd_mecha.model("pure/sdxl_base.safetensors"),
)

unet_recipe = sd_mecha.weighted_sum(
    sd_mecha.model("js2prony_v10.safetensors"),
    sd_mecha.model("juggernautXL_v9Rundiffusionphoto2.safetensors"),
)

recipe = sd_mecha.weighted_sum(
    text_encoder_recipe,
    unet_recipe,
    alpha=sd_mecha.convert({"BASE": 0}, "sdxl-sgm") | 1
)

merger = sd_mecha.RecipeMerger(
    models_dir=r"E:\sd\models\Stable-diffusion",
)

merger.merge_and_save(recipe)
