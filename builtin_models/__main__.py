from builtin_models.paths import target_yaml_directory
from builtin_models.script_runner import get_model_configs
from sd_mecha.extensions import model_config


def main():
    for config in get_model_configs():
        yaml_config = model_config.to_yaml(config)
        with open(target_yaml_directory / f"{config.identifier}.yaml", "w") as f:
            f.write(yaml_config)


if __name__ == "__main__":
    main()
