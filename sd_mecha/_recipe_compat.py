"""Migrate deprecated extraction method calls in saved recipes until sd-mecha 1.3.0."""

import re
import warnings
from copy import copy

from .extensions import merge_methods


def deserialize_merge_call(identifier, args, kwargs):
    try:
        method = merge_methods.resolve(identifier)
    except KeyError:
        recipe = _deserialize_legacy_extraction(identifier, args, kwargs)
        if recipe is None:
            raise
        replacement = recipe.merge_method.get_identifier()
        warnings.warn(
            f"Recipe merge method {identifier!r} is deprecated and will be removed in sd-mecha 1.3.0. "
            f"Save the loaded recipe to update it to {replacement!r}.",
            FutureWarning,
            stacklevel=4,
        )
        return recipe
    return method(*args, **kwargs)


def _deserialize_legacy_extraction(identifier, args, kwargs):
    if match := re.fullmatch(r"extract_(lora|lokr)_'([^']+)'", identifier):
        algorithm, config = match.groups()
        method = merge_methods.resolve(f"extract_{algorithm}_{config}")
    elif match := re.fullmatch(r"extract_(lycoris|kohya)_(lora|lokr)", identifier):
        adapter_format, algorithm = match.groups()
        method = copy(merge_methods.resolve(f"extract_{algorithm}"))
        method.interface = copy(method.interface)
        method.interface.candidates = [
            (candidate, signature)
            for candidate, signature in method.interface.candidates
            if getattr(signature.return_annotation.data.model_config, "lycoris_identifier", None) == adapter_format
        ]
    else:
        return None

    if algorithm == "lokr":
        return _deserialize_legacy_lokr(method, *args, **kwargs)
    return method(*args, **kwargs)


def _deserialize_legacy_lokr(method, base, kronecker_ratio=0.5):
    from .extensions.builtin.merge_methods.kronecker import lokr_dims

    return method(base, dims=lokr_dims(base, kronecker_ratio))
