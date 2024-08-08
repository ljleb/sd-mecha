import os
import pathlib
import subprocess
from concurrent.futures import as_completed, ProcessPoolExecutor
from builtin_models.paths import get_script_module, get_executable, get_script_venv, module_dir, scripts_dir
from types import ModuleType


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
    script_venv_dir = get_module_venv(module)
    run_script(script_venv_dir, script_path)


def get_module_venv(module: ModuleType) -> pathlib.Path:
    if not hasattr(module, "get_venv"):
        raise RuntimeError(f"Function `get_venv` is not defined in script {module.__file__}")
    return get_script_venv(module.get_venv())


def run_script(new_venv_dir: pathlib.Path, script_path: pathlib.Path):
    args = [str(module_dir / "script_main.py"), str(script_path)]
    subprocess.run([str(get_executable(new_venv_dir))] + args, cwd=os.getcwd(), check=True)
