import sd_mecha
sd_mecha.set_log_level()

recipe = sd_mecha.dropout("ghostmix_v20Bakedvae", "dreamshaper_332BakedVaeClipFix", p=0.9, l=0.5, seed=0)
merger = sd_mecha.RecipeMerger(models_dir=r"E:\sd\models\Stable-diffusion")
merger.merge_and_save(recipe, output="basic_merge")
