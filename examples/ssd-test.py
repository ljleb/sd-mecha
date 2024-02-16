import pathlib
import safetensors.torch

models = []

# Iterate over all files with the .safetensors extension in the directory and its subdirectories
for file in pathlib.Path(r"E:\sd\models\Stable-diffusion").glob('**/*.safetensors'):
    try:
        with safetensors.safe_open(file, framework="pt") as f:
            models.append(file.absolute())  # Add the absolute path to the list
    except:
        pass


import sd_mecha
import torch
sd_mecha.set_log_level()


recipe = models[0]
for i, model in enumerate(models[1:], start=2):
    # dtype to accommodate precision loss by alpha near 1
    dtype = torch.float16 if i - 2 < 4 else torch.float32 if i - 2 < 16 else torch.float64
    recipe = sd_mecha.weighted_sum(model, recipe, alpha=(i-1)/i, dtype=dtype)

scheduler = sd_mecha.MergeScheduler(base_dir=r"E:\sd\models\Stable-diffusion")
scheduler.merge_and_save(recipe, output_path="n_average2")
