import sd_mecha


# 5 example models to average together
models = [
    "ghostmix_v20Bakedvae",
    "dreamshaper_332BakedVaeClipFix",
    "deliberate_v2",
    "darkSushi25D25D_v20",
    "CounterfeitV30_v30",
]

merge = models[0]
for i, model in enumerate(models[1:], start=2):
    merge = sd_mecha.weighted_sum(merge, model, alpha=1/i)

scheduler = sd_mecha.MergeScheduler(
    base_dir=r"E:\sd\models\Stable-diffusion",
    device="cuda:0",
)

scheduler.merge_and_save(merge, output_path="n_average_test")
