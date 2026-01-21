import pathlib
from typing import List


_registry: List[pathlib.Path] = []


def get_all() -> List[pathlib.Path]:
    return _registry.copy()


def add_path(path: str | pathlib.Path):
    if isinstance(path, str):
        path = pathlib.Path(path)
    _registry.append(path)


def normalize_path(path: pathlib.Path) -> pathlib.Path:
    if not path.is_absolute():
        for base_dir in _registry:
            path_attempt = base_dir / path
            if path_attempt.exists():
                path = path_attempt
                break
    return path.resolve()
