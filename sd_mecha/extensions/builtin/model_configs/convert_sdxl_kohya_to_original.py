from torch import Tensor
from sd_mecha.extensions.merge_methods import merge_method, StateDict, Return, Parameter
from sd_mecha.extensions import model_configs
from .convert_n_to_1 import convert_n_to_1, convert_1_to_n
from .convert_sdxl_diffusers_unet_to_original import convert_unet_key
from .convert_sdxl_diffusers_clip_g_to_original import convert_clip_g_key


sdxl_sgm_config = model_configs.resolve('sdxl-sgm')
sdxl_kohya_config = model_configs.resolve('sdxl-kohya')
sdxl_kohya_diffusers_config = model_configs.resolve('sdxl-kohya_but_diffusers')


@merge_method(
    identifier=f"convert_'{sdxl_kohya_config.identifier}'_to_'{sdxl_sgm_config.identifier}'",
    is_conversion=True,
    reuse_outputs=False,
)
class convert_sdxl_kohya_to_original:
    @staticmethod
    def map_keys(b):
        for output_key in sdxl_sgm_config.keys():
            needs_transpose = False
            if output_key.startswith("model.diffusion_model."):
                input_keys = output_key.replace("model.diffusion_model.", "unet.")
            elif output_key.startswith("conditioner.embedders.0."):
                input_keys = output_key.replace("conditioner.embedders.0.transformer.", "te1.")
            elif output_key.startswith("conditioner.embedders.1."):
                input_keys, needs_transpose = convert_clip_g_key(output_key)
            elif output_key.startswith("first_stage_model."):
                input_keys = output_key
            else:
                input_keys = output_key
            try:
                b[output_key] = b.keys[input_keys] @ needs_transpose
            except ValueError:
                pass

    def __call__(
        self,
        kohya_sd: Parameter(StateDict[Tensor], model_config=sdxl_kohya_config),
        **kwargs,
    ) -> Return(Tensor, model_config=sdxl_sgm_config):
        relation = kwargs["key_relation"]
        return convert_n_to_1(kohya_sd, relation.inputs["kohya_sd"], relation.meta)


@merge_method(
    identifier=f"convert_'{sdxl_kohya_diffusers_config.identifier}'_to_'{sdxl_sgm_config.identifier}'",
    is_conversion=True,
    reuse_outputs=False,
)
class convert_sdxl_kohya_but_diffusers_to_original:
    @staticmethod
    def map_keys(b):
        needs_transpose = False
        for output_key in sdxl_sgm_config.keys():
            if output_key.startswith("model.diffusion_model."):
                input_keys = convert_unet_key(output_key, prefix="unet.")
            elif output_key.startswith("conditioner.embedders.0."):
                input_keys = output_key.replace("conditioner.embedders.0.transformer.", "te1.")
            elif output_key.startswith("conditioner.embedders.1."):
                input_keys, needs_transpose = convert_clip_g_key(output_key)
            elif output_key.startswith("first_stage_model."):
                input_keys = output_key
            else:
                input_keys = output_key
            try:
                b[output_key] = b.keys[input_keys] @ needs_transpose
            except ValueError:
                pass

    def __call__(
        self,
        kohya_sd: Parameter(StateDict[Tensor], model_config=sdxl_kohya_diffusers_config),
        **kwargs,
    ) -> Return(Tensor, model_config=sdxl_sgm_config):
        relation = kwargs["key_relation"]
        return convert_n_to_1(kohya_sd, relation.inputs["kohya_sd"], relation.meta)


@merge_method(
    identifier=f"convert_'{sdxl_sgm_config.identifier}'_to_'{sdxl_kohya_config.identifier}'",
    is_conversion=True,
    reuse_outputs=False,
)
class convert_sdxl_kohya_to_original:
    @staticmethod
    def map_keys(b):
        for output_key in sdxl_sgm_config.keys():
            needs_transpose = False
            if output_key.startswith("model.diffusion_model."):
                input_keys = output_key.replace("model.diffusion_model.", "unet.")
            elif output_key.startswith("conditioner.embedders.0."):
                input_keys = output_key.replace("conditioner.embedders.0.transformer.", "te1.")
            elif output_key.startswith("conditioner.embedders.1."):
                input_keys, needs_transpose = convert_clip_g_key(output_key)
            elif output_key.startswith("first_stage_model."):
                input_keys = output_key
            else:
                input_keys = output_key
            try:
                b[input_keys] = b.keys[output_key] @ needs_transpose
            except ValueError:
                pass

    def __call__(
        self,
        sgm_sd: Parameter(StateDict[Tensor], model_config=sdxl_sgm_config),
        **kwargs,
    ) -> Return(StateDict[Tensor], model_config=sdxl_kohya_config):
        relation = kwargs["key_relation"]
        sgm_key = relation.inputs["sgm_sd"][0]
        return convert_1_to_n(sgm_sd[sgm_key], relation.outputs, relation.meta)


@merge_method(
    identifier=f"convert_'{sdxl_sgm_config.identifier}'_to_'{sdxl_kohya_diffusers_config.identifier}'",
    is_conversion=True,
    reuse_outputs=False,
)
class convert_sdxl_kohya_but_diffusers_to_original:
    @staticmethod
    def map_keys(b):
        needs_transpose = False
        for output_key in sdxl_sgm_config.keys():
            if output_key.startswith("model.diffusion_model."):
                input_keys = convert_unet_key(output_key, prefix="unet.")
            elif output_key.startswith("conditioner.embedders.0."):
                input_keys = output_key.replace("conditioner.embedders.0.transformer.", "te1.")
            elif output_key.startswith("conditioner.embedders.1."):
                input_keys, needs_transpose = convert_clip_g_key(output_key)
            elif output_key.startswith("first_stage_model."):
                input_keys = output_key
            else:
                input_keys = output_key
            try:
                b[input_keys] = b.keys[output_key] @ needs_transpose
            except ValueError:
                pass

    def __call__(
        self,
        sgm_sd: Parameter(StateDict[Tensor], model_config=sdxl_sgm_config),
        **kwargs,
    ) -> Return(Tensor, model_config=sdxl_kohya_diffusers_config):
        relation = kwargs["key_relation"]
        sgm_key = relation.inputs["sgm_sd"][0]
        return convert_1_to_n(sgm_sd[sgm_key], relation.outputs, relation.meta)
