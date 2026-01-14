import pathlib
from typing import List


_registry: List[pathlib.Path] = []


def get_all() -> List[pathlib.Path]:
    return _registry.copy()


def add_path(path: str | pathlib.Path):
    if isinstance(path, str):
        path = pathlib.Path(path)
    _registry.append(path)
