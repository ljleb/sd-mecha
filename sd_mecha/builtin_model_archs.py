import pathlib
from sd_mecha.extensions.model_arch import register_model_arch

models_dir = pathlib.Path(__file__).parent / "models"
register_model_arch(models_dir / "sd1_ldm.yaml", "sd1")
register_model_arch(models_dir / "sdxl_sgm.yaml", "sdxl")
register_model_arch(models_dir / "sd3_sgm.yaml", "sd3")
register_model_arch(models_dir / "flux_flux.yaml", "flux")
