import sd_mecha
sd_mecha.set_log_level()


recipe = sd_mecha.add_difference(
    "Stable-diffusion/ghostmix_v20Bakedvae",
    sd_mecha.lora("Lora/head-mounted display3-000007.safetensors"),
    alpha=0.5,
)

scheduler = sd_mecha.MergeScheduler(
    base_dir=r"E:\sd\models",
)

scheduler.merge_and_save(recipe, output_path="Stable-diffusion/basic_merge")
