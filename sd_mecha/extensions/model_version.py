import dataclasses
import functools
import operator

import fuzzywuzzy.process
import pathlib
import re
import traceback
import yaml
from typing import Set, Dict

WILDCARD = "*"
PADDING = "#"


def discover_block_prefixes(keys: Set[str], config: dict, version_id: str):
    discovered_blocks = {}

    for module, module_config in config["merge"].items():
        prefix = module_config["prefix"]
        for block_patterns, shorthand in module_config["blocks"].items():
            patterns = block_patterns.split(" ")
            first_pattern_re = re.escape(patterns[0]).replace(re.escape(WILDCARD), r'(\w+)').replace(re.escape(PADDING), r'\w+')
            pattern = re.compile(rf"^{prefix}\.{first_pattern_re}")

            for key in keys:
                match = pattern.match(key)
                if match:
                    block_id = match.groups()[0] if WILDCARD in patterns[0] else ""
                    block_shorthand = (version_id + "_" + module + "_block_" + shorthand.replace(WILDCARD, "{}")).format(block_id)
                    if block_shorthand not in discovered_blocks:
                        discovered_blocks[block_shorthand] = {
                            "patterns": [
                                re.compile(re.escape(f"{prefix}.{p.replace(WILDCARD, block_id)}.").replace(re.escape(PADDING), r'\w+'))
                                for p in patterns
                            ],
                            "module": module,
                        }

    return discovered_blocks


def discover_blocks(keys, discovered_block_prefixes, version_id):
    blocks = {}
    classes = {}
    for key in keys:
        for block, prefixes in discovered_block_prefixes.items():
            if any((match := prefix.match(key)) for prefix in prefixes["patterns"]):
                blocks.setdefault(key, set()).add(block)
                clazz = key[len(match.group(0)):]
                while clazz in ["weight", "bias"] or clazz[:1].isnumeric():
                    part = key[:-len(clazz)-1].split(".")[-1]
                    clazz = part + "." + clazz
                clazz = clazz.replace(".weight", "").replace(".bias", "").replace(".", "_")
                classes[key] = {version_id + "_" + prefixes["module"] + "_class_" + clazz}
                break

    return blocks, classes


@dataclasses.dataclass
class ModelVersion:
    identifier: str
    components: Set[str]
    keys: Set[str]
    keys_to_forward: Set[str]
    keys_to_merge: Set[str]
    blocks: Dict[str, Set[str]]
    classes: Dict[str, Set[str]]
    location: str

    def user_keys(self) -> Set[str]:
        return (
            functools.reduce(operator.or_, self.blocks.values(), set()) |
            functools.reduce(operator.or_, self.classes.values(), set())
        )


def register_model_version(
    yaml_config_path: str | pathlib.Path,
    identifier: str,
):
    stack_frame = traceback.extract_stack(None, 2)[0]

    if identifier in _model_versions_registry:
        existing_location = _model_versions_registry[identifier].location
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
    blocks, classes = discover_blocks(keys_to_merge, block_prefixes, identifier)
    components = set(config["merge"].keys())

    location = f"{stack_frame.filename}:{stack_frame.lineno}"
    _model_versions_registry[identifier] = ModelVersion(identifier, components, keys, keys_to_forward, keys_to_merge, blocks, classes, location)


_model_versions_registry = {}


def resolve(identifier: str) -> ModelVersion:
    try:
        return _model_versions_registry[identifier]
    except KeyError as e:
        suggestion = fuzzywuzzy.process.extractOne(str(e), _model_versions_registry.keys())[0]
        raise ValueError(f"unknown merge method: {e}. Nearest match is '{suggestion}'")
