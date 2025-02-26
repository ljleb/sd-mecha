import sd_mecha
sd_mecha.set_log_level()


text_encoder_recipe = sd_mecha.add_perpendicular(
    sd_mecha.model("js2prony_v10.safetensors"),
    sd_mecha.model("furry-xl-4.0.safetensors"),
    sd_mecha.model("pure/sdxl_base.safetensors"),
)

unet_recipe = sd_mecha.weighted_sum(
    sd_mecha.model("js2prony_v10.safetensors"),
    sd_mecha.model("furry-xl-4.0.safetensors"),
)

recipe = sd_mecha.pick_component(unet_recipe, "diffuser") | text_encoder_recipe

merger = sd_mecha.Defaults(
    model_dirs=r"E:\sd\models\Stable-diffusion",
)

merger.merge_and_save(recipe, output={})
