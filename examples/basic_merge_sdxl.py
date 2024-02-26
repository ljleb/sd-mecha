import sd_mecha
sd_mecha.set_log_level()


recipe = sd_mecha.weighted_sum(
    sd_mecha.model("animagineXLV3_v30", "sdxl"),
    sd_mecha.model("juggernautXL_v9Rundiffusionphoto2", "sdxl"),
)

merger = sd_mecha.RecipeMerger(
    models_dir=r"E:\sd\models\Stable-diffusion",
)

merger.merge_and_save(recipe, output="test_a")
