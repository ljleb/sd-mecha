import sd_mecha
sd_mecha.set_log_level()


recipe = sd_mecha.binomial_dropout("ghostmix_v20Bakedvae", "dreamshaper_332BakedVaeClipFix", p=0.5, l=1.0)
merger = sd_mecha.RecipeMerger(models_dir=r"E:\sd\models\Stable-diffusion")
merger.merge_and_save(recipe, output="basic_merge")
