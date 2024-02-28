import sd_mecha
sd_mecha.set_log_level()


recipe = sd_mecha.weighted_sum("ghostmix_v20Bakedvae", "dreamshaper_332BakedVaeClipFix")
merger = sd_mecha.RecipeMerger(models_dir=r"E:\sd\models\Stable-diffusion")

state_dict = {}
merger.merge_and_save(recipe, output=state_dict)
