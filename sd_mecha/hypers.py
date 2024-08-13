import re
import fuzzywuzzy.process
from typing import Dict, Optional, List
from sd_mecha import extensions
from sd_mecha.extensions.model_config import ModelConfig


Hyper = int | float | Dict[str, int | float]


def get_hyper(hyper: Hyper, key: str, model_config: ModelConfig, default_value: Optional[int | float]) -> int | float:
    if isinstance(hyper, (int, float)):
        return hyper
    elif isinstance(hyper, dict):
        result = 0.0
        total = 0
        for component_id, component in model_config.components.items():
            for block_id, block in component.blocks.items():
                if key not in block.keys_to_merge:
                    continue

                hyper_id = model_config.get_hyper_block_key(component_id, block_id)
                value = hyper.get(hyper_id)
                if value is not None:
                    result += value
                    total += 1

        if total > 0:
            return result / total

        for component_id, component in model_config.components.items():
            if key in component.keys_to_merge:
                try:
                    return hyper[model_config.get_hyper_default_key(component_id)]
                except KeyError:
                    continue

        if default_value is not None:
            return default_value

        raise ValueError(f"Key {key} does not have a value")
    else:
        raise TypeError(f"Hyperparameter must be a float or a dictionary, not {type(hyper).__name__}")


def validate_hyper(hyper: Hyper, model_config: Optional[ModelConfig]) -> Hyper:
    if isinstance(hyper, dict):
        if model_config is None:
            raise ValueError("Abstract recipes (with recipe parameters) cannot specify component-wise hyperparameters")

        hyper_keys = list(model_config.hyper_keys())
        for key in hyper.keys():
            if key not in hyper_keys and not key.endswith("_default"):
                suggestion = fuzzywuzzy.process.extractOne(key, hyper_keys)[0]
                raise ValueError(f"Unsupported dictionary key '{key}'. Nearest match is '{suggestion}'")
    elif isinstance(hyper, (int, float)):
        return hyper
    else:
        raise TypeError(f"Hyperparameter must be a float or a dictionary, not {type(hyper).__name__}")


def blocks(model_config: str | ModelConfig, model_component: str, *args, strict: bool = True, **kwargs) -> Hyper:
    """
    Generate block hyperparameters for a model version.
    Either positional arguments or keyword arguments can be used, but not both at the same time.
    If positional blocks are used, they must all be specified, unless strict=False.
    The CLI has a command to help determine the names of the blocks without guessing:

    ```
    python -m sd_mecha info <model_arch>
    ```

    where `<model_arch>` is the version identifier of the model used. (i.e. "sd1", "sdxl", etc.)

    :param model_config: the configuration of the model for which to generate block hyperparameters
    :param model_component: the model component for which the block-wise hyperparameters are intended for (i.e. "unet", "txt", "txt2", etc.)
    :param strict: if blocks are passed to *args, determines whether the number of blocks must match exactly the maximum number of blocks for the selected model component
    :param args: positional block hyperparameters
    :param kwargs: keyword block hyperparameters
    :return: block-wise hyperparameters
    """
    if isinstance(model_config, str):
        model_config = extensions.model_config.resolve(model_config)

    if args and kwargs:
        raise ValueError("`args` and `kwargs` cannot be used at the same time. Either use positional or keyword arguments, but not both.")
    if args:
        identifiers = list(
            model_config.get_hyper_block_key(model_component, block_id)
            for block_id in model_config.components[model_component].blocks
        )
        if strict and len(args) != len(identifiers):
            raise ValueError(f"blocks() got {len(args)} block{'s' if len(args) > 1 else ''} but {len(identifiers)} are expected. Use keyword arguments to pass only a few (i.e. 'in0=1, out3=0.5, ...') or pass strict=False.")

        identifiers.sort(key=natural_sort_key)
        return dict(zip(identifiers, args))
    return {
        model_config.get_hyper_block_key(model_component, block_id): v
        for block_id, v in kwargs.items()
    }


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def default(model_config: str | ModelConfig, model_components: Optional[str | List[str]] = None, value: int | float = 0) -> Hyper:
    if isinstance(model_config, str):
        model_config = extensions.model_config.resolve(model_config)

    if not model_components:
        model_components = model_config.components
    elif isinstance(model_components, str):
        model_components = [model_components]

    return {
        f"{model_config.identifier}_{model_component}_default": value
        for model_component in model_components
    }
