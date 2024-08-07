import re
import fuzzywuzzy.process
from typing import Dict, Optional, List
from sd_mecha import extensions
from sd_mecha.extensions.model_arch import ModelArch


Hyper = int | float | Dict[str, int | float]


def get_hyper(hyper: Hyper, key: str, model_arch: ModelArch) -> int | float:
    if isinstance(hyper, (float, int)):
        return hyper
    elif isinstance(hyper, dict):
        block_keys = model_arch.blocks[key]
        result = 0.0
        total = 0
        for block_key in block_keys:
            value = hyper.get(block_key)
            if value is not None:
                result += value
                total += 1

        if total > 0:
            return result / total

        for block_key in block_keys:
            for component in model_arch.components:
                if f"_{component}_" not in block_key:
                    continue

                try:
                    return hyper[model_arch.identifier + "_" + component + "_default"]
                except KeyError:
                    continue

        return 0
    else:
        raise TypeError(f"Hyperparameter must be a float or a dictionary, not {type(hyper).__name__}")


def validate_hyper(hyper: Hyper, model_arch: Optional[ModelArch]) -> Hyper:
    if isinstance(hyper, dict):
        if model_arch is None:
            raise ValueError("Abstract recipes (with recipe parameters) cannot specify component-wise hyperparameters")

        user_keys = model_arch.user_keys()
        for key in hyper.keys():
            if key not in user_keys and not key.endswith("_default"):
                suggestion = fuzzywuzzy.process.extractOne(key, user_keys)[0]
                raise ValueError(f"Unsupported dictionary key '{key}'. Nearest match is '{suggestion}'")
    elif isinstance(hyper, (int, float)):
        return hyper
    else:
        raise TypeError(f"Hyperparameter must be a float or a dictionary, not {type(hyper).__name__}")


def blocks(model_arch: str | ModelArch, model_component: str, *args, strict: bool = True, **kwargs) -> Hyper:
    """
    Generate block hyperparameters for a model version.
    Either positional arguments or keyword arguments can be used, but not both at the same time.
    If positional blocks are used, they must all be specified, unless strict=False.
    The CLI has a command to help determine the names of the blocks without guessing:

    ```
    python -m sd_mecha info <model_arch>
    ```

    where `<model_arch>` is the version identifier of the model used. (i.e. "sd1", "sdxl", etc.)

    :param model_arch: the architecture of the model for which to generate block hyperparameters
    :param model_component: the model component for which the block-wise hyperparameters are intended for (i.e. "unet", "txt", "txt2", etc.)
    :param strict: if blocks are passed to *args, determines whether the number of blocks must match exactly the maximum number of blocks for the selected model component
    :param args: positional block hyperparameters
    :param kwargs: keyword block hyperparameters
    :return: block-wise hyperparameters
    """
    if isinstance(model_arch, str):
        model_arch = extensions.model_arch.resolve(model_arch)

    if args and kwargs:
        raise ValueError("`args` and `kwargs` cannot be used at the same time. Either use positional or keyword arguments, but not both.")
    if args:
        identifiers = list(
            k for k in model_arch.user_keys()
            if "_" + model_component + "_block_" in k
        )
        if strict and len(args) != len(identifiers):
            raise ValueError(f"blocks() got {len(args)} block{'s' if len(args) > 1 else ''} but {len(identifiers)} are expected. Use keyword arguments to pass only a few (i.e. 'in0=1, out3=0.5, ...') or pass strict=False.")

        identifiers.sort(key=natural_sort_key)
        return dict(zip(identifiers, args))
    return {
        model_arch.identifier + "_" + model_component + "_block_" + k: v
        for k, v in kwargs.items()
    }


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def default(model_arch: str | ModelArch, model_components: Optional[str | List[str]] = None, value: int | float = 0) -> Hyper:
    if isinstance(model_arch, str):
        model_arch = extensions.model_arch.resolve(model_arch)

    if not model_components:
        model_components = model_arch.components
    elif isinstance(model_components, str):
        model_components = [model_components]

    return {
        model_arch.identifier + "_" + model_component + "_default": value
        for model_component in model_components
    }
