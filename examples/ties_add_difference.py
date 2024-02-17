import sd_mecha
sd_mecha.set_log_level()


models = [
    "ghostmix_v20Bakedvae",
    "dreamshaper_332BakedVaeClipFix",
    "realisticVisionV20_v20",
    "illustrationArtstyleMM_27",
    "lyriel_v16",
    "Midnight Maple",
    "mixproyuki77mi_v10",
]


recipe = sd_mecha.add_difference_ties("pure/v1-5-pruned", *models, alpha=0.5)

scheduler = sd_mecha.RecipeMerger(
    base_dir=r"E:\sd\models\Stable-diffusion",
    default_device="cpu",
)

scheduler.merge_and_save(recipe, output_path="basic_merge")
