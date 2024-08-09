import pathlib
import sys

import torch

from model_configs.disable_init import MetaTensorMode, DisableInitialization
from model_configs.paths import get_target_yaml_file, get_script_module, extra_path, target_yaml_dir
from model_configs.script_venvs import get_venv_configs
from sd_mecha.extensions.model_config import to_yaml


def run_script(script_path: pathlib.Path):
    module = get_script_module(script_path)

    if not hasattr(module, "create_configs"):
        raise RuntimeError(f"Function `create_configs` not found in script {script_path}")

    extra_sys_paths = []
    if hasattr(module, "get_venv"):
        venv_configs = get_venv_configs()
        venv_config = venv_configs[module.get_venv()]
        extra_sys_paths.extend(str(path.resolve()) for path in venv_config.sys_paths)

    with extra_path(*extra_sys_paths), DisableInitialization(), MetaTensorMode(), torch.no_grad():
        model_configs = module.create_configs()

    for model_config in model_configs:
        yaml_config = to_yaml(model_config)
        target_yaml_file = get_target_yaml_file(model_config.identifier)
        target_yaml_dir.mkdir(parents=True, exist_ok=True)
        with open(target_yaml_file, "w") as f:
            f.write(yaml_config)


if __name__ == "__main__":
    run_script(pathlib.Path(sys.argv[1]))
