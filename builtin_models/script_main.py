import importlib.util
import pathlib
import sys
from builtin_models.paths import target_yaml_dir
from sd_mecha.extensions.model_config import to_yaml


def run_script(script_path: pathlib.Path):
    module_name = script_path.stem
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, 'create_config'):
        model_config = module.create_config()
    else:
        raise RuntimeError(f"Function `create_config` not found in script {script_path}")

    yaml_config = to_yaml(model_config)
    with open(target_yaml_dir / f"{model_config.identifier}.yaml", "w") as f:
        f.write(yaml_config)


if __name__ == "__main__":
    run_script(pathlib.Path(sys.argv[1]))
