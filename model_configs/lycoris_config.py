import torch.nn
import lycoris


def create_lycoris_configs(
    identifier: str,
    model: torch.nn.Module,
):
    for algo in lycoris.wrapper.network_module_dict:
        lycoris_wrapper = lycoris.kohya.create_network(
            1.0,
            network_dim=16,
            network_alpha=1.0,
            vae=model.vae,
            text_encoder=[model.text_encoders.clip_l, model.text_encoders.t5xxl],
            unet=model.model.diffusion_model,
            train_norm=True,
        )
        lycoris_wrapper.apply_to([model.text_encoders.clip_l, model.text_encoders.t5xxl], model.model.diffusion_model, apply_text_encoder=True, apply_unet=True)
        state_dict = lycoris_wrapper.state_dict()
        lycoris_wrapper.restore()
