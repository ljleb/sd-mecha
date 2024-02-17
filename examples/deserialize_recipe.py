import pathlib
import sd_mecha
sd_mecha.set_log_level()


recipe_path = pathlib.Path(__file__).parent / 'recipes' / "test_weighted_sum.mecha"
with open(recipe_path) as f:
    recipe = sd_mecha.deserialize(f.readlines())


merger = sd_mecha.RecipeMerger(
    base_dir=r"E:\sd\models\Stable-diffusion",
)

# perform the entire merge plan and save to output path
merger.merge_and_save(recipe, output_path="basic_merge")
