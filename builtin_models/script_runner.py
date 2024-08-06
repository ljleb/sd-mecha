import importlib.util
import multiprocessing
import pathlib
from builtin_models.paths import scripts_dir


def get_model_configs():
    script_paths = scripts_dir.glob('*.py')
    with multiprocessing.Pool(1) as pool:
        return pool.map(run_create_config_script, script_paths)


def run_create_config_script(script_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, 'create_config'):
        return module.create_config()
    else:
        raise RuntimeError("function `create_config()` not found")
