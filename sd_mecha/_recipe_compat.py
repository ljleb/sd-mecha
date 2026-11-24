"""Resolve deprecated model config IDs and recipe method calls until sd-mecha 1.3.0."""

import re
import warnings

from .extensions import merge_methods


_ANIMA_ADAPTER_CONFIG_ID = "anima-diffusion_pipe_lycoris_lora"
_DEPRECATED_MODEL_CONFIG_IDS = {
    "anima-diffuser": "anima-diffusion_pipe",
    "anima-diffuser_kohya_lora": _ANIMA_ADAPTER_CONFIG_ID,
    "anima-diffuser_kohya_lokr": _ANIMA_ADAPTER_CONFIG_ID,
    "anima-diffuser_kohya_loha": _ANIMA_ADAPTER_CONFIG_ID,
}


def _canonical_config(identifier: str) -> str:
    if identifier in _DEPRECATED_MODEL_CONFIG_IDS:
        return _DEPRECATED_MODEL_CONFIG_IDS[identifier]
    if identifier.endswith("_kohya_lora"):
        return identifier.removesuffix("_kohya_lora") + "_lycoris_lora"
    return identifier


def deprecated_model_config_alias(identifier: str) -> str | None:
    replacement = _canonical_config(identifier)
    if replacement == identifier:
        return None

    warnings.warn(
        f"Model config alias {identifier!r} is deprecated and will be removed in sd-mecha 1.3.0. "
        f"Use {replacement!r} instead.",
        FutureWarning,
        stacklevel=3,
    )
    return replacement


def deserialize_merge_call(identifier, args, kwargs):
    if identifier == "rotate" and (
        len(args) > 4 or {"alignment", "stiefel_eps", "stiefel_max_iters"}.intersection(kwargs)
    ):
        return _deserialize_legacy_rotate(*args, **kwargs)
    try:
        method = merge_methods.resolve(identifier)
    except KeyError:
        recipe = _deserialize_legacy_merge_call(identifier, args, kwargs)
        if recipe is None:
            raise
        method = getattr(recipe, "merge_method", None)
        update = "Save the loaded recipe to update it."
        if method is not None:
            update = f"Save the loaded recipe to update it to {method.get_identifier()!r}."
        warnings.warn(
            f"Recipe merge method {identifier!r} is deprecated and will be removed in sd-mecha 1.3.0. {update}",
            FutureWarning,
            stacklevel=4,
        )
        return recipe
    return method(*args, **kwargs)


def _deserialize_legacy_merge_call(identifier, args, kwargs):
    if match := re.fullmatch(r"convert_'([^']+)'_to_'([^']+)'", identifier):
        source, target = match.groups()
        if source != target and (source.endswith("_kohya_lora") or target.endswith("_kohya_lora")):
            if _canonical_config(source) == _canonical_config(target):
                return _identity(*args, **kwargs)

    if match := re.fullmatch(r"convert_'([^']+)'_to_base", identifier):
        config = _canonical_config(match[1])
        if config == match[1]:
            return None
        if "lora" in kwargs and "lyco" not in kwargs:
            kwargs = dict(kwargs)
            kwargs["lyco"] = kwargs.pop("lora")
        return merge_methods.resolve(f"convert_'{config}'_to_base")(*args, **kwargs)

    if match := re.fullmatch(r"apply_lycoris_(.+_kohya_lora)", identifier):
        config = _canonical_config(match[1])
        return merge_methods.resolve(f"apply_lycoris_{config}")(*args, **kwargs)

    legacy_lokr = False
    if match := re.fullmatch(r"extract_(lora|lokr)_'([^']+)'", identifier):
        algorithm, config = match.groups()
        method = merge_methods.resolve(f"extract_{algorithm}_{_canonical_config(config)}")
        legacy_lokr = algorithm == "lokr"
    elif match := re.fullmatch(r"extract_(?:lycoris|kohya)_(lora|lokr)", identifier):
        algorithm = match[1]
        legacy_lokr = algorithm == "lokr"
        method = merge_methods.resolve(f"extract_{algorithm}")
    elif match := re.fullmatch(r"(extract_.+)_kohya_lora", identifier):
        try:
            method = merge_methods.resolve(f"{match[1]}_lycoris_lora")
        except KeyError:
            return None
    else:
        return None

    return _deserialize_legacy_lokr(method, *args, **kwargs) if legacy_lokr else method(*args, **kwargs)


def _identity(adapter):
    return adapter


def _deserialize_legacy_rotate(a, b, alignment=1.0, alpha=0.0, centralization=1.0, stiefel_eps=1e-8, stiefel_max_iters=100):
    warnings.warn(
        "The rotate alignment parameter has no effect in sd-mecha 1.2.0. "
        "The last version where it affected rotation was 1.1.8. "
        "Loading recipes with legacy alignment arguments will raise an error in sd-mecha 1.3.0. "
        "Save this recipe again to migrate it to the current rotate signature.",
        FutureWarning,
        stacklevel=3,
    )
    return merge_methods.resolve("rotate")(a, b, alpha, centralization)


def _deserialize_legacy_lokr(method, base, kronecker_ratio=0.5):
    from .extensions.builtin.merge_methods.kronecker import lokr_dims

    return method(base, dims=lokr_dims(base, kronecker_ratio))
