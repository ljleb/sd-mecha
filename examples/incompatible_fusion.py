import torch
import sd_mecha
sd_mecha.set_log_level()

a = sd_mecha.parameter("a")
b = sd_mecha.parameter("b")
c = sd_mecha.parameter("c")

recipe = sd_mecha.rotate(
    sd_mecha.add_perpendicular(a, b, c), a,
    alpha=1.0,
    device="cuda", dtype=torch.float64,
)

sd_mecha.serialize_and_save(recipe, "recipes/incompatible_fusion")
