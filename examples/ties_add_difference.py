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
#recipe_ties_soup = sd_mecha.add_difference_ties("pure/v1-5-pruned", *models, alpha=1.0, k=1.0, vote_sgn=1.0)

merger = sd_mecha.RecipeMerger(
    models_dir=r"E:\sd\models\Stable-diffusion",
    default_device="cpu",
)

merger.merge_and_save(recipe, output="basic_merge")
