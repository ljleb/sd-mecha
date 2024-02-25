import re
import fuzzywuzzy.process
from typing import Dict, Optional, List
from sd_mecha import extensions
from sd_mecha.extensions.model_version import ModelVersion


Hyper = int | float | Dict[str, int | float]


def get_hyper(hyper: Hyper, key: str, model_version: ModelVersion) -> int | float:
    if isinstance(hyper, (float, int)):
        return hyper
    elif isinstance(hyper, dict):
        hyper_ids = model_version.classes[key] | model_version.blocks[key]
        result = 0.0
        total = 0
        for hyper_id in hyper_ids:
            value = hyper.get(hyper_id)
            if value is not None:
                result += value
                total += 1

        if total > 0:
            return result / total

        for hyper_id in hyper_ids:
            for component in model_version.components:
                if f"_{component}_" not in hyper_id:
                    continue

                try:
                    return hyper[model_version.identifier + "_" + component + "_default"]
                except KeyError:
                    continue

        return 0
    else:
        raise TypeError(f"Hyperparameter must be a float or a dictionary, not {type(hyper)}")


def validate_hyper(hyper: Hyper, model_version: ModelVersion) -> Hyper:
    if isinstance(hyper, dict):
        user_keys = model_version.user_keys()
        for key in hyper.keys():
            if key not in user_keys and not key.endswith("_default"):
                suggestion = fuzzywuzzy.process.extractOne(key, user_keys)[0]
                raise ValueError(f"Unsupported dictionary key '{key}'. Nearest match is '{suggestion}'.")
    elif isinstance(hyper, (int, float)):
        return hyper
    else:
        raise TypeError(f"Hyperparameter must be a float or a dictionary, not {type(hyper)}")


def blocks(model_version: str | ModelVersion, model_component: str, *args, **kwargs) -> Hyper:
    """
    Generate block hyperparameters for a model version.
    Either positional arguments or keyword arguments can be used, but not both at the same time.
    If positional blocks are used, they must all be specified.
    The CLI has a command to help determine the names of the blocks without guessing:

    ```
    python -m sd_mecha info <model_version> blocks
    ```

    where `<model_version>` is the version identifier of the model used. (i.e. "sd1", "sdxl", etc.)

    :param model_version: the version of the model for which to generate block hyperparameters
    :param model_component: the model component for which the block-wise hyperparameters are intended for (i.e. "unet", "txt", "txt2", etc.)
    :param args: block hyperparameters by value
    :param kwargs: block hyperparameters by name
    :return: block-wise hyperparameters
    """
    if isinstance(model_version, str):
        model_version = extensions.model_version.resolve(model_version)

    if args and kwargs:
        raise ValueError("`args` and `kwargs` cannot be used at the same time. Either use positional or keyword arguments, but not both.")
    if args:
        identifiers = list(
            k for k in model_version.user_keys()
            if "_" + model_component + "_block_" in k
        )
        if len(args) != len(identifiers):
            raise ValueError(f"blocks() got {len(args)} block{'s' if len(args) > 1 else ''} but {len(identifiers)} are expected. Use keyword arguments to pass only a few. (i.e. 'in0=1, out3=0.5, ...')")

        identifiers.sort(key=natural_sort_key)
        return dict(zip(identifiers, args))
    return {
        model_version.identifier + "_" + model_component + "_block_" + k: v
        for k, v in kwargs.items()
    }


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def classes(model_version: str | ModelVersion, model_component: str, **kwargs) -> Hyper:
    """
    Generate class hyperparameters for a model version.
    The CLI has a command to help determine the names of the classes for a given model version without guessing:

    ```
    python -m sd_mecha info <model_version> classes
    ```

    where `<model_version>` is the version identifier of the model used. (i.e. "sd1", "sdxl", etc.)

    :param model_version: the version of the model for which to generate class hyperparameters
    :param model_component: the model component for which the class-wise hyperparameters are intended for (i.e. "unet", "txt", "txt2", etc.)
    :param kwargs: class hyperparameters by name
    :return: class-wise hyperparameters
    """
    if isinstance(model_version, str):
        model_version = extensions.model_version.resolve(model_version)

    return {
        model_version.identifier + "_" + model_component + "_class_" + k: v
        for k, v in kwargs.items()
    }


def default(model_version: str | ModelVersion, model_components: Optional[str | List[str]] = None, value: int | float = 0) -> Hyper:
    if isinstance(model_version, str):
        model_version = extensions.model_version.resolve(model_version)

    if not model_components:
        model_components = model_version.components
    elif isinstance(model_components, str):
        model_components = [model_components]

    return {
        model_version.identifier + "_" + model_component + "_default": value
        for model_component in model_components
    }
