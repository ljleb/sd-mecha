import pathlib
from sd_mecha import extensions
from sd_mecha.extensions.model_config import YamlModelConfig


yaml_directory = pathlib.Path(__file__).parent


def _register_configs():
    for yaml in yaml_directory.glob("*.yaml"):
        config = YamlModelConfig(yaml)
        extensions.model_config.register(config)


_register_configs()
from .convert_sdxl_kohya_to_original import convert_sdxl_kohya_to_original
from .convert_sd1_kohya_to_original import convert_sd1_kohya_to_original
