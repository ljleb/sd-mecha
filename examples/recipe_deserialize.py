import pathlib
import sd_mecha
sd_mecha.set_log_level()


recipe_path = pathlib.Path(__file__).parent / 'recipes' / "test_split_unet_text_encoder.mecha"
with open(recipe_path) as f:
    recipe = sd_mecha.deserialize(f.readlines())


merger = sd_mecha.Defaults(models_dir=r"E:\sd\models\Stable-diffusion")
merger.merge_and_save(recipe)
