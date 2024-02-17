import sd_mecha
sd_mecha.set_log_level()


recipe = sd_mecha.add_difference(
    "pure/v1-5-pruned",
    sd_mecha.ties_sum(
        sd_mecha.subtract("ghostmix_v20Bakedvae", "pure/v1-5-pruned"),
        sd_mecha.subtract("dreamshaper_332BakedVaeClipFix", "pure/v1-5-pruned"),
        sd_mecha.subtract("realisticVisionV20_v20", "pure/v1-5-pruned"),
        sd_mecha.subtract("darkSushi25D25D_v20", "pure/v1-5-pruned"),
        sd_mecha.subtract("illustrationArtstyleMM_27", "pure/v1-5-pruned"),
        sd_mecha.subtract("lyriel_v16", "pure/v1-5-pruned"),
        sd_mecha.subtract("Midnight Maple", "pure/v1-5-pruned"),
        sd_mecha.subtract("mixproyuki77mi_v10", "pure/v1-5-pruned"),
        alpha=1.0,
    ),
)

scheduler = sd_mecha.MergeScheduler(
    base_dir=r"E:\sd\models\Stable-diffusion",
)

scheduler.merge_and_save(recipe, output_path="basic_merge")
