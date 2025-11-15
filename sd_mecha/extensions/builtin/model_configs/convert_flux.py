from torch import Tensor
from sd_mecha import merge_method, Parameter, Return, StateDict, StateDictKeyError


@merge_method(is_conversion=True)
def convert_flux_to_backbone_only(
    flux: Parameter(StateDict[Tensor], model_config="flux-flux"),
    **kwargs,
) -> Return(Tensor, model_config="flux-flux_diffuser_only"):
    diffuser_only_key = kwargs["key"]
    full_key = f"model.diffusion_model.{diffuser_only_key}"
    return flux[full_key]


@merge_method(is_conversion=True)
def convert_flux_backbone_to_full(
    flux_diffuser: Parameter(StateDict[Tensor], model_config="flux-flux_diffuser_only"),
    **kwargs,
) -> Return(Tensor, model_config="flux-flux"):
    full_key = kwargs["key"]
    if not full_key.startswith("model.diffusion_model."):
        raise StateDictKeyError(full_key)

    diffuser_only_key = full_key[len("model.diffusion_model."):]
    return flux_diffuser[diffuser_only_key]
