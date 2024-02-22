import sd_mecha
sd_mecha.set_log_level()


recipe = sd_mecha.weighted_sum(
    "animagineXLV3_v30",
    "juggernautXL_v9Rundiffusionphoto2",
)

merger = sd_mecha.RecipeMerger(
    models_dir=r"E:\sd\models\Stable-diffusion",
)

merger.merge_and_save(recipe, output="basic_merge")
