from typing import List, Dict, Set, Tuple
import fuzzywuzzy.process


class MergeSpace:
    def __init__(self, identifier: str):
        self.identifier = identifier

    def __eq__(self, other):
        if isinstance(other, str):
            other = MergeSpace(other)
        return self.identifier == other.identifier

    def __hash__(self):
        return hash(self.identifier)


class MergeSpaceSymbol:
    def __init__(self, *merge_spaces: Tuple[str | MergeSpace, ...]):
        self.merge_spaces = {
            MergeSpace(merge_space) if isinstance(merge_space, str) else merge_space
            for merge_space in merge_spaces
        }


AnyMergeSpace = Set[MergeSpace] | MergeSpaceSymbol


def register_merge_space(identifier: str):
    if identifier in _merge_space_registry:
        raise ValueError(f"merge space {identifier} already exists")

    merge_space = MergeSpace(identifier)
    _merge_space_registry[identifier] = merge_space
    return merge_space


def resolve(identifier: str) -> MergeSpace:
    try:
        return _merge_space_registry[identifier]
    except KeyError as e:
        suggestion = fuzzywuzzy.process.extractOne(str(e), _merge_space_registry.keys())[0]
        raise ValueError(f"unknown merge space: {e}. Nearest match is '{suggestion}'")


def get_identifiers(merge_space: AnyMergeSpace) -> List[str]:
    if isinstance(merge_space, Set):
        return [merge_space.identifier for merge_space in merge_space]
    elif isinstance(merge_space, MergeSpaceSymbol):
        return get_identifiers(merge_space.merge_spaces)
    else:
        raise TypeError(f"expected {MergeSpaceSymbol.__name__} or Tuple[{MergeSpace.__name__}, ...], got {type(merge_space)}")


def get_all() -> Set[MergeSpace]:
    return set(_merge_space_registry.values())


_merge_space_registry: Dict[str, MergeSpace] = {
    merge_space: register_merge_space(merge_space)
    for merge_space in ("weight", "delta", "param")
}
