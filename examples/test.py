import sd_mecha
sd_mecha.set_log_level()


recipe = sd_mecha.rotate(
    "animagineXLV3_v30",
    "juggernautXL_v9Rundiffusionphoto2",
    alpha=(
        sd_mecha.sdxl_unet_blocks(
            in00=1.0,
            in01=0.22433768037666652,
            in02=0.9410822844867236,
            in03=0.5215865072802894,
            in04=0.7296995363182784,
            in05=0.8470249787923061,
            in06=0.5184139321353828,
            in07=0.21574537505056232,
            in08=1.0,
            mid00=0.37754973978285744,
            out00=1.0,
            out01=0.9198311296844665,
            out02=0.5326383121636705,
            out03=1.0,
            out04=0.21152291927503236,
            out05=0.33753802934627164,
            out06=0.4385880279690535,
            out07=0.38610522199778274,
            out08=0.902798403755789
        ) |
        sd_mecha.sdxl_txt_blocks(default=1.0) |
        sd_mecha.sdxl_txt_g14_blocks(default=1.0)
    ),
    beta=(
        sd_mecha.sdxl_unet_blocks(
            in00=0.13611656877000583,
            in01=0.3978682065523139,
            in02=0.045707845052389874,
            in03=0.0,
            in04=0.19059007051287766,
            in05=0.5278322694876812,
            in06=0.46020493750920655,
            in07=0.3656028335744898,
            in08=0.7184282463142058,
            mid00=0.21357560794886013,
            out00=0.0418222423706616,
            out01=1.0,
            out02=0.867712668545755,
            out03=0.506517621479329,
            out04=0.2788185763771121,
            out05=0.023123873191855028,
            out06=0.12142054533840675,
            out07=0.2491099371622733,
            out08=1.0
        ) |
        sd_mecha.sdxl_txt_blocks(default=0.16027571563113877) |
        sd_mecha.sdxl_txt_g14_blocks(default=0.16027571563113877)
    )
)

merger = sd_mecha.RecipeMerger(
    base_dir=r"E:\sd\models\Stable-diffusion",
    default_device="cuda",
)

merger.merge_and_save(recipe, output_path="hoaxbdzr7mecha")
