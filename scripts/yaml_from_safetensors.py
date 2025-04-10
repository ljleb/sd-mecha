import argparse
import pathlib
from collections import OrderedDict
from sd_mecha.extensions import model_configs
from sd_mecha.extensions.model_configs import ModelConfigImpl, ModelComponent, KeyMetadata
from sd_mecha.streaming import InSafetensorsDict


def create_model_config(config_id: str, path_to_model: pathlib.Path):
    header = OrderedDict(
        (k, KeyMetadata(v.shape, v.dtype))
        for k, v in InSafetensorsDict(path_to_model, 0).metadata()
    )
    config = ModelConfigImpl(config_id, {
        "diffuser": ModelComponent(header)
    })
    model_config_str = model_configs.to_yaml(config)
    path_to_config = pathlib.Path.cwd() / f"{config_id}.yaml"
    with open(path_to_config, "w") as f:
        f.write(model_config_str)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a model config YAML from a safetensors model file.",
    )
    parser.add_argument(
        "config_id",
        help="Model config identifier, usually `<arch_id>-<implementation_details>`.",
    )
    parser.add_argument(
        "path_to_model",
        help="Path to the safetensors model file to be used as reference.",
    )
    args = parser.parse_args()
    create_model_config(args.config_id, pathlib.Path(args.path_to_model))


if __name__ == "__main__":
    main()
