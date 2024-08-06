import os
import pathlib
import shutil
import subprocess
import tempfile
from concurrent.futures import as_completed, ProcessPoolExecutor
from builtin_models.paths import get_script_module, get_target_yaml_file, module_dir, scripts_dir, venv_dir
from types import ModuleType
from typing import List


def generate_model_configs():
    script_paths = scripts_dir.glob('*.py')
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(generate_model_config, path) for path in script_paths]
        for future in as_completed(futures):
            if future.exception() is not None:
                for future_to_cancel in futures:
                    future_to_cancel.cancel()
                raise future.exception()
            future.result()


def generate_model_config(script_path: pathlib.Path):
    module = get_script_module(script_path)
    config_identifier = get_config_identifier(module)
    if get_target_yaml_file(config_identifier).exists():
        return

    requirements = get_script_requirements(module)
    with tempfile.TemporaryDirectory("venv") as new_venv_dir:
        if requirements:
            new_venv_dir = pathlib.Path(new_venv_dir).resolve()
            copy_venv_to(new_venv_dir)
            install_packages(new_venv_dir, requirements)
        else:
            new_venv_dir = venv_dir
        run_script(new_venv_dir, script_path)


def get_config_identifier(module: ModuleType):
    if not hasattr(module, 'get_identifier'):
        raise RuntimeError(f"Function `get_identifier` is not defined in script {module.__file__}")
    return module.get_identifier()


def get_script_requirements(module: ModuleType):
    if not hasattr(module, 'get_requirements'):
        return []
    return module.get_requirements()


def copy_venv_to(new_venv_dir: pathlib.Path):
    shutil.copytree(venv_dir, new_venv_dir, dirs_exist_ok=True)


def install_packages(new_venv_dir: pathlib.Path, requirements: List[str]):
    args = ["-m", "pip", "install", "--upgrade-strategy", "only-if-needed"] + requirements
    subprocess.run([str(get_executable(new_venv_dir))] + args, check=True)


def run_script(new_venv_dir: pathlib.Path, script_path: pathlib.Path):
    args = [str(module_dir / "script_main.py"), str(script_path)]
    subprocess.run([str(get_executable(new_venv_dir))] + args, cwd=os.getcwd(), check=True)


def get_executable(new_venv_dir: pathlib.Path):
    if os.name == "nt":
        return new_venv_dir / "Scripts" / "python.exe"
    else:
        return new_venv_dir / "bin" / "python"
