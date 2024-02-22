import sd_mecha
sd_mecha.set_log_level()


# plan a simple weighted sum
recipe = sd_mecha.weighted_sum(
    "ghostmix_v20Bakedvae",
    "dreamshaper_332BakedVaeClipFix",
)

# scheduler provides global defaults for methods
merger = sd_mecha.RecipeMerger(
    models_dir=r"E:\sd\models\Stable-diffusion",
)

# perform the entire merge plan and save to output path
merger.merge_and_save(
    recipe,
    output="basic_merge",
)
