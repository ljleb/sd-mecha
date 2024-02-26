import sd_mecha
sd_mecha.set_log_level()


# plan a simple weighted sum
recipe = sd_mecha.weighted_sum("ghostmix_v20Bakedvae", "dreamshaper_332BakedVaeClipFix")

# merger provides global defaults for methods
merger = sd_mecha.RecipeMerger(models_dir=r"E:\sd\models\Stable-diffusion")

state_dict = {}
merger.merge_and_save(recipe, output=state_dict)
