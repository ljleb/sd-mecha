import pathlib
import sys
from builtin_models.disable_init import MetaTensorMode, DisableInitialization
from builtin_models.paths import get_target_yaml_file, get_script_module, extra_path
from builtin_models.script_venvs import get_venv_configs
from sd_mecha.extensions.model_config import to_yaml


def run_script(script_path: pathlib.Path):
    module = get_script_module(script_path)

    if not hasattr(module, "create_config"):
        raise RuntimeError(f"Function `create_config` not found in script {script_path}")

    extra_sys_paths = []
    if hasattr(module, "get_venv"):
        venv_configs = get_venv_configs()
        venv_config = venv_configs[module.get_venv()]
        extra_sys_paths.extend(str(path.resolve()) for path in venv_config.sys_paths)

    with extra_path(*extra_sys_paths), DisableInitialization(), MetaTensorMode():
        model_config = module.create_config()

    yaml_config = to_yaml(model_config)
    target_yaml_file = get_target_yaml_file(model_config.identifier)
    target_yaml_file.parent.mkdir(parents=True, exist_ok=True)
    with open(target_yaml_file, "w") as f:
        f.write(yaml_config)


if __name__ == "__main__":
    run_script(pathlib.Path(sys.argv[1]))
