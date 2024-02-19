import sd_mecha
sd_mecha.set_log_level()


a = "A"
b = "B"
c = "C"


recipe = sd_mecha.rotate(
    sd_mecha.add_difference(a, b, c, clip_to_ab=True),
    a,
    device="cuda"
)


sd_mecha.serialize_and_save(recipe, output_path="recipes/incompatible_fusion")
