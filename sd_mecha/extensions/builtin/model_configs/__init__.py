import pathlib
from sd_mecha import extensions
from sd_mecha.extensions.model_config import LazyModelConfig


yaml_directory = pathlib.Path(__file__).parent


def _register_configs():
    for yaml in reversed(sorted(yaml_directory.glob("*.yaml"))):
        config = LazyModelConfig(yaml)
        extensions.model_config.register(config)


_register_configs()
from .convert_sdxl_diffusers_to_original import convert_sdxl_diffusers_to_original
from .convert_sd1_diffusers_to_original import convert_sd1_diffusers_to_original
