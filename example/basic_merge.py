import sd_mecha


merge = sd_mecha.weighted_sum(
    "ghostmix_v20Bakedvae",
    "dreamshaper_332BakedVaeClipFix",
    alpha=0.5,
)

scheduler = sd_mecha.MergeScheduler(
    base_dir=r"E:\sd\models\Stable-diffusion",
    device="cuda:0",
    prune=True,
)

scheduler.merge_and_save(merge, output_path="merge_test")
