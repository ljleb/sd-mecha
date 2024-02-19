import sd_mecha
sd_mecha.set_log_level()


recipe = sd_mecha.weighted_sum(
    "animagineXLV3_v30",
    "juggernautXL_v9Rundiffusionphoto2",
)

merger = sd_mecha.RecipeMerger(
    base_dir=r"E:\sd\models\Stable-diffusion",
    default_device="cuda",
)

merger.merge_and_save(recipe, output_path="basic_merge")
