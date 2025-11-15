import pathlib
from sd_mecha.extensions import model_configs


yaml_directory = pathlib.Path(__file__).parent


def _register_configs():
    for yaml in yaml_directory.glob("*.yaml"):
        config = model_configs.YamlModelConfig(yaml)
        model_configs.register(config)


_register_configs()
from .convert_sdxl_kohya_to_original import convert_sdxl_kohya_to_original
from .convert_sdxl_diffusers_unet_to_original import convert_sdxl_diffusers_unet_to_original
from .convert_sd1_kohya_to_original import convert_sd1_kohya_to_original
from . import convert_sdxl_blocks
from . import convert_sd1_blocks
from . import convert_sdxl_sgm_split
