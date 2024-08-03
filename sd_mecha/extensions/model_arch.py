import dataclasses
import fuzzywuzzy.process
import pathlib
import re
import traceback
import yaml
from typing import Set, Dict, List, Tuple


WILDCARD = "*"


def discover_blocks(config: dict, arch_identifier: str):
    blocks = {}

    for component, component_config in config["merge"].items():
        prefix = component_config["prefix"]
        for block_id_pattern, patterns in component_config["blocks"].items():
            if isinstance(patterns, str):
                patterns = [patterns]

            for replacement in enumerate_block_id_replacements(block_id_pattern):
                block_id = re.sub(r"\[.*?]", replacement, block_id_pattern) if replacement else block_id_pattern
                block_key = f"{arch_identifier}_{component}_block_{block_id}"
                blocks[block_key] = re.compile(rf"^{re.escape(prefix)}\.(?:" + "|".join(
                    re.escape(pattern.replace(WILDCARD, replacement) if replacement else pattern)
                    for pattern in patterns
                ) + ")")

    return blocks


def enumerate_block_id_replacements(block_id_pattern: str):
    if "[" not in block_id_pattern:
        yield ""
        return

    replacement_exprs = re.split(r"[\[\]]", block_id_pattern)[1].split(",")
    for replacement_expr in replacement_exprs:
        if ".." in replacement_expr:
            low, high = replacement_expr.split("..")
            for i in range(int(low.strip()), int(high.strip()) + 1):
                yield str(i)
        else:
            yield replacement_expr.strip()


@dataclasses.dataclass
class ModelArch:
    identifier: str
    components: Set[str]
    passthrough_prefixes: Tuple[str]
    blocks: Dict[str, re.Pattern]
    location: str

    def hyper_keys(self) -> List[str]:
        return list(self.blocks.keys())


def register_model_arch(
    yaml_config_path: str | pathlib.Path,
    identifier: str,
):
    stack_frame = traceback.extract_stack(None, 2)[0]

    if identifier in _model_archs_registry:
        existing_location = _model_archs_registry[identifier].location
        raise ValueError(f"Extension '{identifier}' is already registered at {existing_location}.")

    if isinstance(yaml_config_path, str):
        yaml_config_path = pathlib.Path(yaml_config_path)

    with open(yaml_config_path, "r") as f:
        config = yaml.safe_load(f.read())

    passthrough_prefixes = tuple(config["passthrough"])
    blocks = discover_blocks(config, identifier)
    components = set(config["merge"].keys())

    location = f"{stack_frame.filename}:{stack_frame.lineno}"
    _model_archs_registry[identifier] = ModelArch(identifier, components, passthrough_prefixes, blocks, location)


_model_archs_registry = {}


def resolve(identifier: str) -> ModelArch:
    try:
        return _model_archs_registry[identifier]
    except KeyError as e:
        suggestion = fuzzywuzzy.process.extractOne(str(e), _model_archs_registry.keys())[0]
        raise ValueError(f"unknown model architecture: {e}. Nearest match is '{suggestion}'")


def get_all() -> List[str]:
    return list(_model_archs_registry.keys())
