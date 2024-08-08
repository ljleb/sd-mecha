import dataclasses
import functools
import pathlib
import shutil
import subprocess
from builtin_models.paths import get_executable, get_script_venv, repositories_dir, shared_venv_dir
from typing import List, Dict


def create_venvs():
    for venv_name, venv_config in get_venv_configs().items():
        script_venv_dir = get_script_venv(venv_name)
        if script_venv_dir.exists():
            continue

        create_new_venv(script_venv_dir, venv_config.requirements)


def create_new_venv(new_venv_dir: pathlib.Path, requirements: List[str]):
    copy_venv_to(new_venv_dir)
    install_packages(new_venv_dir, requirements)


def copy_venv_to(new_venv_dir: pathlib.Path):
    shutil.copytree(shared_venv_dir, new_venv_dir, dirs_exist_ok=True)


def install_packages(new_venv_dir: pathlib.Path, requirements: List[str]):
    args = ["-m", "pip", "install", "--upgrade-strategy", "only-if-needed"] + requirements
    subprocess.run([str(get_executable(new_venv_dir))] + args)


@dataclasses.dataclass
class VenvConfig:
    requirements: List[str | pathlib.Path]
    sys_paths: List[pathlib.Path] = dataclasses.field(default_factory=list)


@functools.cache
def get_venv_configs() -> Dict[str, VenvConfig]:
    ldm_path = repositories_dir / "stability-ai-stable-diffusion"
    sgm_path = repositories_dir / "stability-ai-generative-models"

    return {
        "ldm": VenvConfig(
            requirements=read_requirements_file(ldm_path / "requirements.txt"),
            sys_paths=[ldm_path],
        ),
        "sgm": VenvConfig(
            requirements=[
                requirement
                for requirement in read_requirements_file(sgm_path / "requirements" / "pt2.txt")
                if not requirement.startswith(("triton", "torch"))
            ],
            sys_paths=[sgm_path],
        )
    }


def read_requirements_file(requirements_file: pathlib.Path):
    return [
        requirement.strip()
        for requirement in requirements_file.read_text().split("\n")
        if requirement.strip() and not requirement.strip().startswith("#")
    ]
