import importlib.util
import os
import pathlib
import sys
from contextlib import contextmanager
from types import ModuleType


module_dir = pathlib.Path(__file__).parent
repositories_dir = module_dir / "repositories"
configs_dir = module_dir / "configs"
scripts_dir = module_dir / "scripts"

sd_mecha_dir = module_dir.parent / "sd_mecha"
target_yaml_dir = sd_mecha_dir / "model_configs"

venv_dir = module_dir.parent / "venv"
script_venvs_dir = module_dir / "venvs"


@contextmanager
def extra_path(*paths):
    original_sys_path = sys.path.copy()
    sys.path[:0] = [str(path) for path in paths]
    yield
    sys.path[:] = original_sys_path


def get_target_yaml_file(identifier: str) -> pathlib.Path:
    return target_yaml_dir / f"{identifier}.yaml"


def get_script_venv(identifier: str) -> pathlib.Path:
    return script_venvs_dir / identifier


def get_script_module(script_path: pathlib.Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_executable(new_venv_dir: pathlib.Path):
    if os.name == "nt":
        return new_venv_dir / "Scripts" / "python.exe"
    else:
        return new_venv_dir / "bin" / "python"
