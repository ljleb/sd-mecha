import pathlib
from sd_mecha.extensions.model_version import register_model_version

models_dir = pathlib.Path(__file__).parent / "models"
register_model_version(models_dir / "sd1_ldm.yaml", "sd1")
