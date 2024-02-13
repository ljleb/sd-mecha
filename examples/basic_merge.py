import sd_mecha
sd_mecha.set_log_level()


# plan a simple weighted sum
merge = sd_mecha.weighted_sum(
    "ghostmix_v20Bakedvae",
    "dreamshaper_332BakedVaeClipFix",
    alpha=0.5,
)

# scheduler contains default parameters
scheduler = sd_mecha.MergeScheduler(
    base_dir=r"E:\sd\models\Stable-diffusion",
    device="cuda:0",
)

# perform the entire merge plan and save to output path
scheduler.merge_and_save(merge, output_path="basic_merge")
