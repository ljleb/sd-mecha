import sd_mecha
sd_mecha.set_log_level()


# plan a simple weighted sum
recipe = sd_mecha.weighted_sum(
    "ghostmix_v20Bakedvae",
    "dreamshaper_332BakedVaeClipFix",
)

# scheduler contains default parameters
merger = sd_mecha.RecipeMerger(
    base_dir=r"E:\sd\models\Stable-diffusion",
    default_device="cuda",
)

# perform the entire merge plan and save to output path
merger.merge_and_save(
    recipe,
    output_path="basic_merge",
    threads=2,
)
