import torch.nn
import warnings
from model_configs.nn_module_config import create_config_from_module, Block, Component
from model_configs.stable_diffusion_components import create_clip_l_component, list_blocks
from sd_mecha.extensions.model_config import ModelConfig
from typing import Iterable


def get_venv() -> str:
    return "huggingface"


def create_configs() -> Iterable[ModelConfig]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline = create_model_pipeline()

    model = SD1Diffusers(pipeline)

    components = (
        create_clip_l_component(model.text_encoder),
        create_vae_component(model.vae),
        create_unet_component(model.unet),
    )

    return [
        create_config_from_module(
            identifier="sd1-diffusers",
            merge_space="weight",
            model=model,
            components=components,
        ),
    ]


def create_unet_component(unet: torch.nn.Module):
    component = Component("unet", unet, [
        *list_blocks("in", unet.down_blocks.children()),
        Block("mid", [unet.mid_block]),
        *list_blocks("out", unet.up_blocks.children()),
    ])
    component.blocks[0].modules_to_merge += [unet.conv_in]
    component.blocks[-1].modules_to_merge += [unet.conv_norm_out, unet.conv_out]
    for i, block in enumerate(component.blocks):
        block.modules_to_merge += [unet.time_embedding]

    return component


def create_vae_component(vae: torch.nn.Module) -> Component:
    component = Component("vae", vae, [
        *list_blocks("in", [*vae.encoder.down_blocks.children()] + [vae.encoder.mid_block], copy=True),
        *list_blocks("out", [vae.decoder.mid_block] + [*vae.decoder.up_blocks.children()], copy=True),
    ], copy_only=True)
    component.blocks[0].modules_to_copy += [vae.encoder.conv_in]
    component.blocks[4].modules_to_copy += [vae.encoder.conv_norm_out, vae.encoder.conv_out]
    if hasattr(vae, "quant_conv"):
        component.blocks[4].modules_to_copy.append(vae.quant_conv)
    component.blocks[5].modules_to_copy += [vae.decoder.conv_in]
    if hasattr(vae, "post_quant_conv"):
        component.blocks[5].modules_to_copy.append(vae.post_quant_conv)
    component.blocks[-1].modules_to_copy += [vae.decoder.conv_norm_out, vae.decoder.conv_out]

    return component


def create_model_pipeline():
    import diffusers
    import transformers

    unet_config = diffusers.UNet2DConditionModel.load_config("runwayml/stable-diffusion-v1-5", subfolder="unet")
    vae_config = diffusers.AutoencoderKL.load_config("runwayml/stable-diffusion-v1-5", subfolder="vae")
    unet = diffusers.UNet2DConditionModel.from_config(unet_config)
    vae = diffusers.AutoencoderKL.from_config(vae_config)
    text_encoder = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = transformers.CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    scheduler = diffusers.PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    return diffusers.StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    )


class SD1Diffusers(torch.nn.Module):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.text_encoder = pipeline.text_encoder
        self.tokenizer = pipeline.tokenizer
        self.vae = pipeline.vae
        self.unet = pipeline.unet
        self.scheduler = pipeline.scheduler
