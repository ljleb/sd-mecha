import sd_mecha
sd_mecha.set_log_level()


recipe = sd_mecha.weighted_sum(
    sd_mecha.model("animagineXLV3_v30.safetensors"),
    sd_mecha.model("juggernautXL_v9Rundiffusionphoto2.safetensors"),
)

merger = sd_mecha.RecipeMerger(
    models_dir=r"E:\sd\models\Stable-diffusion",
)

merger.merge_and_save(recipe)
