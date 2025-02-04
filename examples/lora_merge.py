import sd_mecha
sd_mecha.set_log_level()


base = sd_mecha.model("ghostmix_v20Bakedvae.safetensors")
lora = sd_mecha.convert(sd_mecha.model("head-mounted display3-000007.safetensors"), "sd1-ldm")
recipe = sd_mecha.add_difference(base, lora, alpha=1.0)

merger = sd_mecha.RecipeMerger(
    models_dir=[r"E:\sd\models\Stable-diffusion", r"E:\sd\models\Lora"],
)

merger.merge_and_save(recipe)
