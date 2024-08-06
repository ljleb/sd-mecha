import pathlib
from builtin_models.disable_init import DisableInitialization, MetaTensorMode
from builtin_models.model_config_autodetect import create_config_autodetect, Component, list_blocks
from builtin_models.paths import extra_path, configs_dir, repositories_dir
from sd_mecha.extensions.model_config import ModelConfig
from typing import List


ldm_path = repositories_dir / "stability-ai-stable-diffusion"


def get_requirements() -> List[str | pathlib.Path]:
    return [
        "-r", ldm_path / "requirements.txt",
    ]


def get_identifier() -> str:
    return "sd1-ldm-base"


def create_config() -> ModelConfig:
    with (
        extra_path(ldm_path),
        DisableInitialization(),
        MetaTensorMode()
    ):
        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config

        config = str(configs_dir / "v1-inference.yaml")
        config = OmegaConf.load(config).model
        model = instantiate_from_config(config)

        return create_config_autodetect(
            identifier=get_identifier(),
            merge_space="weight",
            model=model,
            components=(
                Component("txt", txt := model.cond_stage_model.transformer.text_model, {
                    **list_blocks("in", txt.encoder.layers.children()),
                }),
                Component("unet", unet := model.model.diffusion_model, {
                    **list_blocks("in", unet.input_blocks.children()),
                    "mid": unet.middle_block,
                    **list_blocks("out", unet.output_blocks.children()),
                }),
            ),
        )
