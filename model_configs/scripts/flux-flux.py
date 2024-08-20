import torch.nn
from model_configs.nn_module_config import create_config_from_module, Component
from model_configs.stable_diffusion_components import create_clip_l_component, create_t5xxl_component, create_vae_component, list_blocks
from sd_mecha.extensions.model_config import ModelConfig
from typing import Iterable


def get_venv() -> str:
    return "flux"


def create_configs() -> Iterable[ModelConfig]:
    dev_model = EntireFluxModel("flux-dev")
    schnell_model = EntireFluxModel("flux-schnell")

    dev_components = (
        create_clip_l_component(dev_model.text_encoders.clip_l),
        create_t5xxl_component(dev_model.text_encoders.t5xxl),
        create_vae_component(dev_model.vae),
        create_unet_component(dev_model.model.diffusion_model),
    )
    schnell_components = (
        create_clip_l_component(schnell_model.text_encoders.clip_l),
        create_t5xxl_component(schnell_model.text_encoders.t5xxl),
        create_vae_component(schnell_model.vae),
        create_unet_component(schnell_model.model.diffusion_model),
    )

    return [
        create_config_from_module(
            identifier="flux_dev-flux",
            model=dev_model,
            components=dev_components,
        ),
        create_config_from_module(
            identifier="flux_dev_unet_only-flux",
            model=dev_model.model.diffusion_model,
            components=(
                create_unet_component(dev_model.model.diffusion_model),
            ),
        ),
        create_config_from_module(
            identifier="flux_schnell-flux",
            model=schnell_model,
            components=schnell_components,
        ),
        create_config_from_module(
            identifier="flux_schnell_unet_only-flux",
            model=schnell_model.model.diffusion_model,
            components=(
                create_unet_component(schnell_model.model.diffusion_model),
            ),
        ),
    ]


class EntireFluxModel(torch.nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        from flux.util import load_ae
        self.text_encoders = FluxTextEncoders()
        self.vae = load_ae(model_id, hf_download=False)
        self.model = FluxDiffusionModel(model_id)


class FluxDiffusionModel(torch.nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        from flux.util import load_flow_model
        self.diffusion_model = load_flow_model(model_id, hf_download=False)


class FluxTextEncoders(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import CLIPTextModel, T5EncoderModel
        self.clip_l = TransformerWrapper(CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14"))
        self.t5xxl = TransformerWrapper(T5EncoderModel.from_pretrained("google/t5-v1_1-xxl"))


class TransformerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.transformer = model


def create_unet_component(unet: torch.nn.Module) -> Component:
    component = Component("unet", unet, [
        *list_blocks("double", unet.double_blocks.children()),
        *list_blocks("single", unet.single_blocks.children()),
    ])

    component.blocks[0].modules_to_merge += [unet.img_in, unet.txt_in]
    component.blocks[-1].modules_to_merge += [unet.final_layer]
    for block in component.blocks:
        block.modules_to_merge += [unet.time_in, unet.vector_in]
        if unet.params.guidance_embed:
            block.modules_to_merge.append(unet.guidance_in)

    return component
