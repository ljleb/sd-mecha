import dataclasses
import functools
import operator
import fuzzywuzzy.process
import pathlib
import re
import traceback
import yaml
from typing import Set, Dict, List


WILDCARD = "*"


def discover_block_prefixes(keys: Set[str], config: dict, arch_identifier: str):
    discovered_blocks = {}

    for component, component_config in config["merge"].items():
        prefix = component_config["prefix"]
        for shorthand, patterns in component_config["blocks"].items():
            if isinstance(patterns, str):
                patterns = [patterns]

            first_pattern_re = re.escape(patterns[0]).replace(re.escape(WILDCARD), r'(\w+)')
            pattern = re.compile(rf"^{re.escape(prefix)}\.{first_pattern_re}")

            for key in keys:
                match = pattern.match(key)
                if match:
                    block_id = match.groups()[0] if WILDCARD in patterns[0] else ""
                    block_key = (arch_identifier + "_" + component + "_block_" + shorthand.replace(WILDCARD, block_id))
                    if block_key not in discovered_blocks:
                        discovered_blocks[block_key] = {
                            "patterns": [re.compile(p) for p in sorted((
                                re.escape(f"{prefix}.{p.replace(WILDCARD, block_id)}") + r"(?:\.|$)"
                                for p in patterns
                            ), key=lambda s: len(s.split(".")), reverse=True)],
                        }

    return discovered_blocks


def discover_blocks(keys, discovered_block_prefixes, arch_identifier: str):
    blocks = {}
    for key in keys:
        for block, prefixes in discovered_block_prefixes.items():
            if any(prefix.match(key) for prefix in prefixes["patterns"]):
                blocks.setdefault(key, set()).add(block)
                break

    return blocks


@dataclasses.dataclass
class ModelArch:
    identifier: str
    components: Set[str]
    keys: Set[str]
    keys_to_forward: Set[str]
    keys_to_merge: Set[str]
    blocks: Dict[str, Set[str]]
    location: str

    def user_keys(self) -> Set[str]:
        return functools.reduce(operator.or_, self.blocks.values(), set())


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

    keys = config["keys"]
    if isinstance(keys, str):
        with open(yaml_config_path.parent / keys, "r") as f:
            keys = [k.strip() for k in f.readlines()]

    prefixes_to_forward = tuple(config["passthrough"])
    keys_to_forward = set(key for key in keys if key.startswith(prefixes_to_forward))
    keys_to_merge = set(key for key in keys if key not in keys_to_forward)
    block_prefixes = discover_block_prefixes(keys_to_merge, config, identifier)
    blocks = discover_blocks(keys_to_merge, block_prefixes, identifier)
    components = set(config["merge"].keys())

    location = f"{stack_frame.filename}:{stack_frame.lineno}"
    _model_archs_registry[identifier] = ModelArch(identifier, components, keys, keys_to_forward, keys_to_merge, blocks, location)


_model_archs_registry = {}


def resolve(identifier: str) -> ModelArch:
    try:
        return _model_archs_registry[identifier]
    except KeyError as e:
        suggestion = fuzzywuzzy.process.extractOne(str(e), _model_archs_registry.keys())[0]
        raise ValueError(f"unknown model architecture: {e}. Nearest match is '{suggestion}'")


def get_all() -> List[str]:
    return list(_model_archs_registry.keys())
