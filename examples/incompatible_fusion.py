import torch
import sd_mecha
sd_mecha.set_log_level()

a = sd_mecha.parameter("a")
b = sd_mecha.parameter("b")
c = sd_mecha.parameter("c")

recipe = sd_mecha.rotate(
    sd_mecha.add_perpendicular(a, b, c),
    a,
    alpha=0.99,
    cache={},
    device="cuda", dtype=torch.float64,
)

merger = sd_mecha.RecipeMerger(
    models_dir=r"E:\sd\models\Stable-diffusion",
)

merger.merge_and_save(recipe)
