import sd_mecha
sd_mecha.set_log_level()


recipe = sd_mecha.add_difference(
    sd_mecha.model(r"Stable-diffusion\animagineXLV3_v30", "sdxl"),
    sd_mecha.lora(r"Lora\add-detail-xl", "sdxl"),
)

merger = sd_mecha.RecipeMerger(
    models_dir=r"E:\sd\models",
)

merger.merge_and_save(recipe, output=r"Stable-diffusion\test_a")
