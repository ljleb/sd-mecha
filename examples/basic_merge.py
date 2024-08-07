import sd_mecha
from sd_mecha import hypers

sd_mecha.set_log_level()


config = sd_mecha.extensions.model_config.resolve("sd1-ldm-base")


a = sd_mecha.model("ghostmix_v20Bakedvae", "sd1-ldm-base")
b = sd_mecha.model("dreamshaper_332BakedVaeClipFix", "sd1-ldm-base")


# plan a simple weighted sum
recipe = sd_mecha.weighted_sum(
    a,
    b,
    alpha=(
        hypers.blocks("sd1-ldm-base", "unet", in0=0, in1=1, in2=2, in3=3) |
        hypers.default("sd1-ldm-base", value=0.5)
    ),
)

# merger provides global defaults for methods
merger = sd_mecha.RecipeMerger(models_dir=r"E:\sd\models\Stable-diffusion")

# perform the entire merge plan and save to output path
merger.merge_and_save(recipe, output="basic_merge")
