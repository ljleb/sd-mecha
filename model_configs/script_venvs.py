import dataclasses
import functools
import pathlib
import shutil
import subprocess
from model_configs.paths import get_executable, get_script_venv, repositories_dir, shared_venv_dir
from typing import List, Dict


def create_venvs():
    for venv_name, venv_config in get_venv_configs().items():
        script_venv_dir = get_script_venv(venv_name)
        if not script_venv_dir.exists():
            create_new_venv(script_venv_dir, venv_config.requirements)


def create_new_venv(new_venv_dir: pathlib.Path, requirements: List[str]):
    if not requirements:
        return

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
    return {
        "ldm": VenvConfig(
            requirements=[
                "omegaconf==2.1.1",
                "pytorch-lightning==1.4.2",
                "torchmetrics==0.6",
                "kornia==0.6",
                "transformers==4.19.2",
                "open-clip-torch==2.7.0",
            ],
            sys_paths=[repositories_dir / "stability-ai-stable-diffusion"],
        ),
        "sgm": VenvConfig(
            requirements=[
                "omegaconf>=2.3.0",
                "pytorch-lightning==2.0.1",
                "kornia==0.6.9",
                "open-clip-torch>=2.20.0",
                "transformers==4.19.1",
            ],
            sys_paths=[repositories_dir / "stability-ai-generative-models"],
        )
    }
