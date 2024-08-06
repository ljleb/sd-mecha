import pathlib
import sys
from builtin_models.paths import get_target_yaml_file, get_script_module
from sd_mecha.extensions.model_config import to_yaml


def run_script(script_path: pathlib.Path):
    module = get_script_module(script_path)

    if not hasattr(module, 'create_config'):
        raise RuntimeError(f"Function `create_config` not found in script {script_path}")

    model_config = module.create_config()
    yaml_config = to_yaml(model_config)
    with open(get_target_yaml_file(model_config.identifier), "w") as f:
        f.write(yaml_config)


if __name__ == "__main__":
    run_script(pathlib.Path(sys.argv[1]))
