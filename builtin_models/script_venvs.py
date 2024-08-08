import dataclasses
import functools
import pathlib
import shutil
import subprocess
from builtin_models.paths import get_executable, get_script_venv, repositories_dir, venv_dir
from typing import List, Dict


def create_venvs():
    for venv_name, venv_config in get_venv_configs().items():
        script_venv_dir = get_script_venv(venv_name)
        if script_venv_dir.exists():
            continue

        cli_requirements = [
            cli_requirement
            for requirement in venv_config.requirements
            for cli_requirement in (
                (requirement,)
                if isinstance(requirement, str)
                else ("-r", str(requirement))
            )
        ]
        create_new_venv(script_venv_dir, cli_requirements)


def create_new_venv(new_venv_dir: pathlib.Path, requirements: List[str]):
    copy_venv_to(new_venv_dir)
    install_packages(new_venv_dir, requirements)


def copy_venv_to(new_venv_dir: pathlib.Path):
    shutil.copytree(venv_dir, new_venv_dir, dirs_exist_ok=True)


def install_packages(new_venv_dir: pathlib.Path, requirements: List[str]):
    args = ["-m", "pip", "install", "--upgrade-strategy", "only-if-needed"] + requirements
    subprocess.run([str(get_executable(new_venv_dir))] + args, check=True)


@dataclasses.dataclass
class VenvConfig:
    requirements: List[str | pathlib.Path]
    sys_paths: List[pathlib.Path]


@functools.cache
def get_venv_configs() -> Dict[str, VenvConfig]:
    ldm_path = repositories_dir / "stability-ai-stable-diffusion"

    return {
        "ldm": VenvConfig(
            requirements=[ldm_path / "requirements.txt"],
            sys_paths=[ldm_path],
        ),
    }
