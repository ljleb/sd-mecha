from torch import Tensor
from sd_mecha.extensions.merge_methods import merge_method, StateDict, Return, Parameter
from sd_mecha.extensions import model_configs
from .convert_sdxl_diffusers_unet_to_original import convert_unet_key
from .convert_sdxl_diffusers_clip_g_to_original import convert_clip_g, convert_clip_g_key
from .convert_huggingface_sd_vae_to_original import convert_vae, convert_vae_key

sdxl_kohya_config = model_configs.resolve('sdxl-kohya')
sdxl_sgm_config = model_configs.resolve('sdxl-sgm')


@merge_method(
    identifier=f"convert_'{sdxl_kohya_config.identifier}'_to_'{sdxl_sgm_config.identifier}'",
    is_conversion=True,
)
class convert_sdxl_kohya_to_original:
    @staticmethod
    def input_keys_for_output(sgm_key: str, *_args, **_kwargs):
        if sgm_key.startswith("model.diffusion_model."):
            return (sgm_key.replace("model.diffusion_model.", "unet."),)
        elif sgm_key.startswith("conditioner.embedders.0."):
            return (sgm_key.replace("conditioner.embedders.0.transformer.", "te1."),)
        elif sgm_key.startswith("conditioner.embedders.1."):
            return convert_clip_g_key(sgm_key)[0]
        elif sgm_key.startswith("first_stage_model."):
            return (sgm_key,)
        else:
            return (sgm_key,)

    def __call__(
        self,
        kohya_sd: Parameter(StateDict[Tensor], model_config=sdxl_kohya_config),
        **kwargs,
    ) -> Return(Tensor, model_config=sdxl_sgm_config):
        sgm_key = kwargs["key"]

        if sgm_key.startswith("model.diffusion_model."):
            kohya_key = sgm_key.replace("model.diffusion_model.", "unet.")
            return kohya_sd[kohya_key]
        elif sgm_key.startswith("conditioner.embedders.0."):
            kohya_key = sgm_key.replace("conditioner.embedders.0.transformer.", "te1.")
            return kohya_sd[kohya_key]
        elif sgm_key.startswith("conditioner.embedders.1."):
            return convert_clip_g(kohya_sd, sgm_key)
        elif sgm_key.startswith("first_stage_model."):
            return kohya_sd[sgm_key]
        else:
            return kohya_sd[sgm_key]


sdxl_kohya_diffusers_config = model_configs.resolve('sdxl-kohya_but_diffusers')


@merge_method(
    identifier=f"convert_'{sdxl_kohya_diffusers_config.identifier}'_to_'{sdxl_sgm_config.identifier}'",
    is_conversion=True,
)
class convert_sdxl_kohya_but_diffusers_to_original:
    @staticmethod
    def input_keys_for_output(sgm_key: str, *_args, **_kwargs):
        if sgm_key.startswith("model.diffusion_model."):
            return (convert_unet_key(sgm_key, prefix="unet."),)
        elif sgm_key.startswith("conditioner.embedders.0."):
            return (sgm_key.replace("conditioner.embedders.0.transformer.", "te1."),)
        elif sgm_key.startswith("conditioner.embedders.1."):
            return convert_clip_g_key(sgm_key)[0]
        elif sgm_key.startswith("first_stage_model."):
            return (sgm_key,)
        else:
            return (sgm_key,)

    def __call__(
        self,
        kohya_sd: Parameter(StateDict[Tensor], model_config=sdxl_kohya_diffusers_config),
        **kwargs,
    ) -> Return(Tensor, model_config=sdxl_sgm_config):
        sgm_key = kwargs["key"]

        if sgm_key.startswith("model.diffusion_model."):
            kohya_key = convert_unet_key(sgm_key, prefix="unet.")
            return kohya_sd[kohya_key]
        elif sgm_key.startswith("conditioner.embedders.0."):
            kohya_key = sgm_key.replace("conditioner.embedders.0.transformer.", "te1.")
            return kohya_sd[kohya_key]
        elif sgm_key.startswith("conditioner.embedders.1."):
            return convert_clip_g(kohya_sd, sgm_key)
        elif sgm_key.startswith("first_stage_model."):
            return kohya_sd[sgm_key]
        else:
            return kohya_sd[sgm_key]
